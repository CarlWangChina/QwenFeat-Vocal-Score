import json
import os
import random
import sys
import torch
import librosa
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
import logging
import math
from collections import defaultdict
from scipy.stats import pearsonr

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

from audioscore.dataset import AudioDataset_pkl_base, AudioDataset_fullaudio, AudioDataset_tensor
import audioscore.audio_cut

def masked_average(tensor, mask):
    """
    tensor: [16, 854, 5]
    mask: [16, 854] (0/1 mask)
    返回: [16, 5]
    """
    # 扩展mask维度以匹配输入张量
    mask_expanded = mask.unsqueeze(-1)  # [16, 854, 1]
    
    # 应用mask（将无效位置置零）
    masked_tensor = tensor * mask_expanded  # [16, 854, 5]
    
    # 计算有效元素数量（避免除零）
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [16, 1]
    
    # 求和并计算平均值
    sum_result = masked_tensor.sum(dim=1)  # [16, 5]
    avg_result = sum_result / valid_counts  # [16, 5]
    
    return avg_result

class Trainer:
    def __init__(self, model, train_json, val_json=None, device=None, local_rank=-1, world_size=-1, data_type="use_audio_feat"):
        """
        音频评分模型训练器（支持多卡训练）
        Args:
            train_json: 训练集路径
            val_json: 验证集路径
            device: 训练设备 (自动选择GPU如果可用)
            local_rank: 分布式训练的本地进程编号
        """
        self.model = model
        self.local_rank = local_rank
        self.device = device
        self.world_size = world_size
        self.sub_batch_size = 32
        self.contact_batch = False
        self.use_pitch_feature = False
        self.data_type = data_type
        self.grl = None
        
        # self.model.to(self.device)

        # 创建数据集
        if self.data_type=="use_full_audio":
            self.train_dataset = AudioDataset_fullaudio(train_json[0], train_json[1], use_same_data=True)
            self.val_dataset = AudioDataset_fullaudio(val_json[0], val_json[1]) if val_json else None
        elif self.data_type=="use_audio_feat":
            self.train_dataset = AudioDataset_pkl_base(train_json[0], train_json[1], use_same_data=True)
            self.val_dataset = AudioDataset_pkl_base(val_json[0], val_json[1]) if val_json else None
        elif self.data_type=="use_tensor":
            self.train_dataset = AudioDataset_tensor(train_json[0], train_json[1], use_same_data=True)
            self.val_dataset = AudioDataset_tensor(val_json[0], val_json[1]) if val_json else None
        
        # 训练参数默认值
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 10
        self.save_dir = "./checkpoints"
        
        self.criterion = CrossEntropyLoss()
        self.metric_name = "Accuracy"

        if self.local_rank in [-1, 0]:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)  # 设置最低日志级别
            
            # 创建文件处理器（FileHandler）
            file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # 处理器级别
            
            # 创建格式化器（Formatter）
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 将格式化器绑定到处理器
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            self.logger = logger
            self.logger.info("start")
    
    def create_dataloaders(self):
        """创建训练和验证数据加载器（支持分布式采样器）"""
        train_sampler = DistributedSampler(self.train_dataset, shuffle=False, rank=self.local_rank, num_replicas=self.world_size)
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
        return train_loader, val_loader, train_sampler
    
    def preprocess_feat(self, ppg, vec, pit):
        
        ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
        ppg = torch.FloatTensor(ppg)
        # ppg = torch.zeros_like(ppg)

        vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
        vec = torch.FloatTensor(vec)
        # vec = torch.zeros_like(vec)

        delta = random.uniform(0.66, 1.49)
        pit = torch.FloatTensor(pit) * delta
        
        len_pit = pit.size()[0]
        len_vec = vec.size()[0]
        len_ppg = ppg.size()[0]
        len_min = min(len_pit, len_vec)
        len_min = min(len_min, len_ppg)
        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]
        pit = audioscore.audio_cut.f0_to_coarse(pit)

        ppg = ppg.view(1, -1, 1280)
        vec = vec.view(1, -1, 256)
        pit = pit.view(1, -1)

        return ppg, vec, pit

    def collate_fn(self, batch):
        """批处理函数"""
        if self.data_type=="use_full_audio":
            audio_feats_0_whisper = []
            audio_feats_0_hubert = []
            audio_feats_0_f0 = []
            audio_feats_1_whisper = []
            audio_feats_1_hubert = []
            audio_feats_1_f0 = []
            scores = []

            audio_feats_0_whisper_pad = []
            audio_feats_0_hubert_pad = []
            audio_feats_0_f0_pad = []
            audio_feats_1_whisper_pad = []
            audio_feats_1_hubert_pad = []
            audio_feats_1_f0_pad = []

            mask = []

            max_len = 0
            for data_bags, score, file_path in batch:
                feature_0 = data_bags["res_ori"]
                feature_1 = data_bags["res_user"]
                w, h, f = self.preprocess_feat(feature_0["whisper"], feature_0["hubert"], feature_0["f0"])
                audio_feats_0_whisper.append(w)
                audio_feats_0_hubert.append(h)
                audio_feats_0_f0.append(f)
                w, h, f = self.preprocess_feat(feature_1["whisper"], feature_1["hubert"], feature_1["f0"])
                audio_feats_1_whisper.append(w)
                audio_feats_1_hubert.append(h)
                audio_feats_1_f0.append(f)
                scores.append(score)

            for i in range(len(audio_feats_0_whisper)):
                max_len = max(max_len, audio_feats_0_whisper[i].shape[1])

            for i in range(len(audio_feats_0_whisper)):
                audio_feats_0_whisper_pad.append(torch.cat([audio_feats_0_whisper[i], torch.zeros(1, max_len - audio_feats_0_whisper[i].shape[1], 1280)], dim=1))
                audio_feats_0_hubert_pad.append(torch.cat([audio_feats_0_hubert[i], torch.zeros(1, max_len - audio_feats_0_hubert[i].shape[1], 256)], dim=1))
                audio_feats_0_f0_pad.append(torch.cat([audio_feats_0_f0[i], torch.zeros(1, max_len - audio_feats_0_f0[i].shape[1])], dim=1))
                audio_feats_1_whisper_pad.append(torch.cat([audio_feats_1_whisper[i], torch.zeros(1, max_len - audio_feats_1_whisper[i].shape[1], 1280)], dim=1))
                audio_feats_1_hubert_pad.append(torch.cat([audio_feats_1_hubert[i], torch.zeros(1, max_len - audio_feats_1_hubert[i].shape[1], 256)], dim=1))
                audio_feats_1_f0_pad.append(torch.cat([audio_feats_1_f0[i], torch.zeros(1, max_len - audio_feats_1_f0[i].shape[1])], dim=1))

                mask_local = torch.cat([torch.ones((1, audio_feats_0_whisper[i].shape[1])), torch.zeros((1, max_len - audio_feats_0_whisper[i].shape[1]))], dim=1)
                mask.append(mask_local)

            # print("scores", scores)

            return([{
                "audio_feats_0_whisper": torch.cat(audio_feats_0_whisper_pad, dim=1),
                "audio_feats_0_hubert": torch.cat(audio_feats_0_hubert_pad, dim=1),
                "audio_feats_0_f0": torch.cat(audio_feats_0_f0_pad, dim=1).long(),
                "audio_feats_1_whisper": torch.cat(audio_feats_1_whisper_pad, dim=1),
                "audio_feats_1_hubert": torch.cat(audio_feats_1_hubert_pad, dim=1),
                "audio_feats_1_f0": torch.cat(audio_feats_1_f0_pad, dim=1).long(),
                "masks": torch.cat(mask, dim=1),
                "scores": torch.LongTensor(scores)
            }])
        elif self.data_type=="use_audio_feat":

            audio_feats_0_whisper = []
            audio_feats_0_hubert = []
            audio_feats_0_f0 = []
            audio_feats_1_whisper = []
            audio_feats_1_hubert = []
            audio_feats_1_f0 = []
            scores = []
            
            audio_feat_0_f0_pitch = []
            audio_feat_0_f0_pitch_volume = []
            audio_feat_0_overtone = []
            audio_feat_1_f0_pitch = []
            audio_feat_1_f0_pitch_volume = []
            audio_feat_1_overtone = []

            result = []

            # print(batch)

            for data_bags, score, file_path in batch:
                for segment in data_bags:
                    feature_0 = segment["原唱音频特征"]
                    feature_1 = segment["用户音频特征"]

                    w, h, f = self.preprocess_feat(feature_0["whisper"], feature_0["hubert"], feature_0["f0"])
                    audio_feats_0_whisper.append(w)
                    audio_feats_0_hubert.append(h)
                    audio_feats_0_f0.append(f)

                    w, h, f = self.preprocess_feat(feature_1["whisper"], feature_1["hubert"], feature_1["f0"])
                    audio_feats_1_whisper.append(w)
                    audio_feats_1_hubert.append(h)
                    audio_feats_1_f0.append(f)
                        
                    audio_feat_0_overtone.append(torch.tensor(segment["原唱前九泛音音量线"]).view(1, -1, 9))
                    audio_feat_1_overtone.append(torch.tensor(segment["用户前九泛音音量线"]).view(1, -1, 9))
                    audio_feat_0_f0_pitch.append(torch.tensor(segment["原唱音高线"]).view(1, -1))
                    audio_feat_1_f0_pitch.append(torch.tensor(segment["用户音高线"]).view(1, -1))
                    audio_feat_0_f0_pitch_volume.append(torch.tensor(segment["原唱音量线"]).view(1, -1))
                    audio_feat_1_f0_pitch_volume.append(torch.tensor(segment["用户音量线"]).view(1, -1))

                    scores.append(score)

            sub_batch_index = 0
            # 以sub_batch_size为步长，将数据切分成多个子批次， 并生成对应的mask
            for i in range(0, len(audio_feats_0_whisper), self.sub_batch_size):
                sub_batch_size = min(self.sub_batch_size, len(audio_feats_0_whisper) - i)
                audio_feats_0_whisper_sub = audio_feats_0_whisper[i:i+sub_batch_size]
                audio_feats_0_hubert_sub = audio_feats_0_hubert[i:i+sub_batch_size]
                audio_feats_0_f0_sub = audio_feats_0_f0[i:i+sub_batch_size]
                audio_feats_1_whisper_sub = audio_feats_1_whisper[i:i+sub_batch_size]
                audio_feats_1_hubert_sub = audio_feats_1_hubert[i:i+sub_batch_size]
                audio_feats_1_f0_sub = audio_feats_1_f0[i:i+sub_batch_size]

                audio_feat_0_overtone_sub = audio_feat_0_overtone[i:i+sub_batch_size]
                audio_feat_1_overtone_sub = audio_feat_1_overtone[i:i+sub_batch_size]
                audio_feat_0_f0_pitch_sub = audio_feat_0_f0_pitch[i:i+sub_batch_size]
                audio_feat_1_f0_pitch_sub = audio_feat_1_f0_pitch[i:i+sub_batch_size]
                audio_feat_0_f0_pitch_volume_sub = audio_feat_0_f0_pitch_volume[i:i+sub_batch_size]
                audio_feat_1_f0_pitch_volume_sub = audio_feat_1_f0_pitch_volume[i:i+sub_batch_size]

                if not self.contact_batch:
                    score_sub = scores[i:i+sub_batch_size]
                    # 填充pad并生成mask
                    max_len_feature = max([audio_feats_0_whisper_item.shape[1] for audio_feats_0_whisper_item in audio_feats_0_whisper_sub])
                    max_len_pitch = max([audio_feat_0_f0_pitch_item.shape[1] for audio_feat_0_f0_pitch_item in audio_feat_0_f0_pitch_sub])

                    mask = torch.cat([
                        torch.cat((
                            torch.ones((1, audio_feats_0_whisper_item.shape[1])), 
                            torch.zeros((1, max_len_feature - audio_feats_0_whisper_item.shape[1]))
                        ), dim=1) for audio_feats_0_whisper_item in audio_feats_0_whisper_sub], dim=0)

                    # mask = torch.cat([
                    #     torch.cat((
                    #         torch.ones((1, audio_feats_0_whisper_item.shape[1])), 
                    #         torch.zeros((1, max_len_feature - audio_feats_0_whisper_item.shape[1]))
                    #     ), dim=1) for audio_feats_0_whisper_item in audio_feats_0_whisper_sub], dim=0)

                    mack_pitch = torch.cat([
                        torch.cat((
                            torch.ones((1, audio_feat_0_f0_pitch_item.shape[1])), 
                            torch.zeros((1, max_len_pitch - audio_feat_0_f0_pitch_item.shape[1]))
                        ), dim=1) for audio_feat_0_f0_pitch_item in audio_feat_0_f0_pitch_sub], dim=0)

                    audio_feats_0_whisper_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1], feat.shape[2]))], dim=1) for feat in audio_feats_0_whisper_sub], dim=0)
                    audio_feats_0_hubert_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1], feat.shape[2]))], dim=1) for feat in audio_feats_0_hubert_sub], dim=0)
                    audio_feats_0_f0_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1]))], dim=1) for feat in audio_feats_0_f0_sub], dim=0)

                    audio_feats_1_whisper_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1], feat.shape[2]))], dim=1) for feat in audio_feats_1_whisper_sub], dim=0)
                    audio_feats_1_hubert_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1], feat.shape[2]))], dim=1) for feat in audio_feats_1_hubert_sub], dim=0)
                    audio_feats_1_f0_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1]))], dim=1) for feat in audio_feats_1_f0_sub], dim=0)

                    audio_feat_0_overtone_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1]))], dim=1) for feat in audio_feat_0_overtone_sub], dim=0)
                    audio_feat_1_overtone_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_feature - feat.shape[1]))], dim=1) for feat in audio_feat_1_overtone_sub], dim=0)
                    audio_feat_0_f0_pitch_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_pitch - feat.shape[1]))], dim=1) for feat in audio_feat_0_f0_pitch_sub], dim=0)
                    audio_feat_1_f0_pitch_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_pitch - feat.shape[1]))], dim=1) for feat in audio_feat_1_f0_pitch_sub], dim=0)
                    audio_feat_0_f0_pitch_volume_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_pitch - feat.shape[1]))], dim=1) for feat in audio_feat_0_f0_pitch_volume_sub], dim=0)
                    audio_feat_1_f0_pitch_volume_padded = torch.cat([torch.cat([feat, torch.zeros((1, max_len_pitch - feat.shape[1]))], dim=1) for feat in audio_feat_1_f0_pitch_volume_sub], dim=0)


                    score = torch.LongTensor(score_sub)
                else:
                    score_sub = [scores[i]]
                    # print(score_sub, scores, sub_batch_index)
                    audio_feats_0_whisper_padded = torch.cat(audio_feats_0_whisper_sub, dim=1)
                    audio_feats_0_hubert_padded = torch.cat(audio_feats_0_hubert_sub, dim=1)
                    audio_feats_0_f0_padded = torch.cat(audio_feats_0_f0_sub, dim=1)

                    audio_feats_1_whisper_padded = torch.cat(audio_feats_1_whisper_sub, dim=1)
                    audio_feats_1_hubert_padded = torch.cat(audio_feats_1_hubert_sub, dim=1)
                    audio_feats_1_f0_padded = torch.cat(audio_feats_1_f0_sub, dim=1)

                    audio_feat_0_overtone_padded = torch.cat(audio_feat_0_overtone_sub, dim=1)
                    audio_feat_1_overtone_padded = torch.cat(audio_feat_1_overtone_sub, dim=1)
                    audio_feat_0_f0_pitch_padded = torch.cat(audio_feat_0_f0_pitch_sub, dim=1)
                    audio_feat_1_f0_pitch_padded = torch.cat(audio_feat_1_f0_pitch_sub, dim=1)
                    audio_feat_0_f0_pitch_volume_padded = torch.cat(audio_feat_0_f0_pitch_volume_sub, dim=1)
                    audio_feat_1_f0_pitch_volume_padded = torch.cat(audio_feat_1_f0_pitch_volume_sub, dim=1)
                    mack_pitch = torch.ones_like(audio_feat_0_f0_pitch_padded, dtype=torch.float32)

                    # print("score_sub, score", score_sub, score)

                    score = torch.LongTensor(score_sub)
                    mask = torch.ones_like(audio_feats_0_f0_padded, dtype=torch.float32)

                if torch.isnan(audio_feats_0_whisper_padded).any().item():
                    print("NaN in audio_feats_0_whisper_padded")
                if torch.isnan(audio_feats_0_hubert_padded).any().item():
                    print("NaN in audio_feats_0_hubert_padded")
                if torch.isnan(audio_feats_0_f0_padded).any().item():
                    print("NaN in audio_feats_0_f0_padded")
                if torch.isnan(audio_feats_1_whisper_padded).any().item():
                    print("NaN in audio_feats_1_whisper_padded")
                if torch.isnan(audio_feats_1_hubert_padded).any().item():
                    print("NaN in audio_feats_1_hubert_padded")
                if torch.isnan(audio_feats_1_f0_padded).any().item():
                    print("NaN in audio_feats_1_f0_padded")
                if torch.isnan(mask).any().item():
                    print("NaN in mask")
                if torch.isnan(score).any().item():
                    print("NaN in score")

                # print(audio_feats_0_whisper_padded.shape)
                # print(audio_feats_0_hubert_padded.shape)
                # print(audio_feats_0_f0_padded.shape)
                # print(audio_feats_1_whisper_padded.shape)
                # print(audio_feats_1_hubert_padded.shape)
                # print(audio_feats_1_f0_padded.shape)
                # print(mask.shape, score.shape)
                # print("score.long()", score, score.long())

                result.append({
                    "audio_feats_0_whisper": audio_feats_0_whisper_padded,
                    "audio_feats_0_hubert": audio_feats_0_hubert_padded,
                    "audio_feats_0_f0": audio_feats_0_f0_padded.long(),
                    "audio_feats_1_whisper": audio_feats_1_whisper_padded,
                    "audio_feats_1_hubert": audio_feats_1_hubert_padded,
                    "audio_feats_1_f0": audio_feats_1_f0_padded.long(),
                    "audio_feats_0_overtone": audio_feat_0_overtone_padded,
                    "audio_feats_1_overtone": audio_feat_1_overtone_padded,
                    "audio_feats_0_f0_pitch": audio_feat_0_f0_pitch_padded,
                    "audio_feats_1_f0_pitch": audio_feat_1_f0_pitch_padded,
                    "audio_feats_0_f0_pitch_volume": audio_feat_0_f0_pitch_volume_padded,
                    "audio_feats_1_f0_pitch_volume": audio_feat_1_f0_pitch_volume_padded,
                    "masks_pitch": mack_pitch,
                    "masks": mask,
                    "scores": score.long()
                })
                sub_batch_index += 1

            return result
        elif self.data_type == "use_tensor":

            audio_feats_0 = []
            audio_feats_1 = []
            scores = []

            audio_feats_0_pad = []
            audio_feats_1_pad = []

            mask = []

            max_len = 0
            for data_bags, score, file_path in batch:
                feature_0 = data_bags["res_ori"].view(1, -1, 1024)
                feature_1 = data_bags["res_user"].view(1, -1, 1024)
                # print(data_bags["res_ori"].shape, data_bags["res_user"].shape)
                audio_feats_0.append(feature_0)
                audio_feats_1.append(feature_1)
                scores.append(score)

            for i in range(len(audio_feats_0)):
                max_len = max(max_len, audio_feats_0[i].shape[1])

            for i in range(len(audio_feats_0)):
                audio_feats_0_pad.append(torch.cat([audio_feats_0[i], torch.zeros(1, max_len - audio_feats_0[i].shape[1], 1024)], dim=1))
                audio_feats_1_pad.append(torch.cat([audio_feats_1[i], torch.zeros(1, max_len - audio_feats_1[i].shape[1], 1024)], dim=1))

                mask_local = torch.cat([torch.ones((1, audio_feats_0[i].shape[1])), torch.zeros((1, max_len - audio_feats_0[i].shape[1]))], dim=1)
                mask.append(mask_local)

            # print("scores", scores)

            return([{
                "audio_feats_0": torch.cat(audio_feats_0_pad, dim=1),
                "audio_feats_1": torch.cat(audio_feats_1_pad, dim=1),
                "masks": torch.cat(mask, dim=1),
                "scores": torch.LongTensor(scores)
            }])
    
    def train(self):
        """训练主循环（支持多卡训练）"""
        print("start training")
        self.model.to(self.device)
        print(f"create ddp model rank={self.local_rank}, {self.device}")
        self.model = DistributedDataParallel(self.model, broadcast_buffers=False) # 进程数大于2时在此处卡死
        print("create ddp model")
        train_loader, val_loader, train_sampler = self.create_dataloaders()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        best_val_accuracy = float('0')
        os.makedirs(self.save_dir, exist_ok=True)
        if self.local_rank in [-1, 0]:
            print(self.model)
            self.save_model(f"initial_model")
            self.logger.info(f"Initial model saved to {self.save_dir}/initial_model.pt")

        dist.barrier()
        for epoch in range(self.epochs):
            
            # if val_loader and self.local_rank in [-1, 0]:
            #     val_metrics = self.evaluate(val_loader)
            #     print(val_metrics)
            # dist.barrier()

            # 设置分布式采样器的epoch
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # 训练阶段
            self.model.train()
            total_loss = 0.0
            # 只在主进程显示进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            
            for batch in progress_bar:
                
                local_loss = self.train_step(batch, optimizer, epoch)
                # print(f"local_loss={local_loss} rank={self.local_rank}")

                total_loss += local_loss

                dist.barrier()
                progress_bar.set_postfix({"loss": f"{local_loss:.4f}", "rank": f"{self.local_rank}"})
                
                # if self.local_rank in [-1, 0]:
                #     self.logger.info(f"Training loss: {local_loss:.4f}")

                # dist.barrier()
                # print(f"total_loss:{total_loss} rank={self.local_rank}")
            
            # print(f"train epoch done rank={self.local_rank}")

            # 计算平均训练损失（跨所有GPU）
            if self.local_rank != -1:
                total_loss_tensor = torch.tensor(total_loss, device=self.device)
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = total_loss_tensor.item() / dist.get_world_size()
            
            avg_train_loss = total_loss / len(train_loader)

            print(f"avg_train_loss:{avg_train_loss:.4f} rank={self.local_rank}")
            
            if self.local_rank in [-1, 0]:
                self.logger.info(f"Average training loss: {avg_train_loss:.4f}")

            dist.barrier()
            # 验证阶段（只在主进程进行）
            val_metrics = {}
            if val_loader and self.local_rank in [-1, 0]:
                val_avg_loss, val_accuracy, accuracy_range, Val_var, pearson_corr = self.evaluate(val_loader)

                self.logger.info(f"Validation loss: {val_avg_loss:.4f} accuracy: {val_accuracy:.4f} accuracy_range:{accuracy_range:.4f} var: {Val_var:.4f} Pearson: {pearson_corr} Epoch: {epoch+1}")
                
                # 保存最佳模型
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_model(f"best_model_epoch/{epoch+1}")
                    self.logger.info(f"Best model saved to {self.save_dir}/best_model_epoch/{epoch+1}")
            
            # 打印epoch结果（只在主进程）
            if self.local_rank in [-1, 0]:
                print(f"\nEpoch {epoch+1} Results:")
                print(f"Train Loss: {avg_train_loss:.4f}")
                if val_metrics:
                    print(f"Val Loss: {val_metrics['loss']:.4f}")
                    print(f"Val {self.metric_name}: {val_metrics['metric']:.4f}")
                    self.logger.info(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f}, "+\
                                     f"Val Loss: {val_metrics['loss']:.4f}, "+\
                                     f"Val {self.metric_name}: {val_metrics['metric']:.4f}, "+\
                                     f"Acc_dis:{val_metrics['acc_dis']:.4f} "+\
                                     f"Acc_area:{val_metrics['acc_area']:.4f} ")
            
            # 定期保存模型（只在主进程）
            if (epoch + 1) % 8 == 0 and self.local_rank in [-1, 0]:
                self.save_model(f"model_epoch{epoch+1}")
                self.logger.info(f"Model saved to {self.save_dir}/model_epoch{epoch+1}.pt")
    
    def train_step(self, batch, optimizer, epoch):
        self.model.train()
        optimizer.zero_grad()

        # print("len(batch):", len(batch))
    
        sub_batch = batch[epoch % len(batch)]
        # 准备数据
        scores = sub_batch["scores"].to(self.device).long()
        if self.use_pitch_feature:
            mask = sub_batch["masks_pitch"].to(self.device)
        else:
            mask = sub_batch["masks"].to(self.device)

        # print(sub_batch["audio_feats_0_whisper"].shape)
        # print(sub_batch["audio_feats_0_hubert"].shape)
        # print(sub_batch["audio_feats_0_f0"].shape)
        # print(sub_batch["audio_feats_1_whisper"].shape)
        # print(sub_batch["audio_feats_1_hubert"].shape)
        # print(sub_batch["audio_feats_1_f0"].shape)
        # print(mask.shape)
        # print(scores.shape)
        
        # 前向传播
        if self.data_type=="use_tensor":
            outputs = self.model(
                sub_batch["audio_feats_0"].to(self.device),
                sub_batch["audio_feats_1"].to(self.device),
                mask,
            )
        elif self.use_pitch_feature:
            outputs = self.model(
                audio_feats_0_overtone = sub_batch["audio_feats_0_overtone"].float().to(self.device),
                audio_feats_0_f0 = sub_batch["audio_feats_0_f0_pitch"].long().to(self.device),
                audio_feats_0_volume = sub_batch["audio_feats_0_f0_pitch_volume"].float().to(self.device),
                audio_feats_1_overtone = sub_batch["audio_feats_1_overtone"].float().to(self.device),
                audio_feats_1_f0 = sub_batch["audio_feats_1_f0_pitch"].long().to(self.device),
                audio_feats_1_volume = sub_batch["audio_feats_1_f0_pitch_volume"].float().to(self.device),
                mask = mask,
            )
        else:
            outputs = self.model(
                sub_batch["audio_feats_0_whisper"].to(self.device),
                sub_batch["audio_feats_0_hubert"].to(self.device),
                sub_batch["audio_feats_0_f0"].to(self.device),
                sub_batch["audio_feats_1_whisper"].to(self.device),
                sub_batch["audio_feats_1_hubert"].to(self.device),
                sub_batch["audio_feats_1_f0"].to(self.device),
                mask,
            )
        if torch.isnan(outputs).any().item():
            print("NaN in outputs")
        outputs = masked_average(outputs, mask)

        if self.grl is not None:
            # 使用梯度反转层
            outputs_grl = self.grl(sub_batch)
        
        # 计算损失并梯度累积
        loss_current = self.criterion(outputs, scores)
        # dist.barrier()
        # print(f"loss_current {loss_current.item()} rank: {self.local_rank}")
        # dist.barrier()
        loss_current.backward()  # 反向传播（自动梯度累加）
        # dist.barrier()
        
        total_loss = loss_current.item()  # 记录标量值
    
        optimizer.step()  # 参数更新
        # print(f"optimizer.step total_loss={total_loss} rank={self.local_rank}")
        return total_loss  # 返回总损失

    def evaluate(self, dataloader):
        """评估模型性能（只在主进程调用）"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_correct_range = 0
        total_elements = 0
    
        all_preds = []    # 原始预测值集合
        all_scores = []   # 新增：真实标签集合
        all_masked_outputs = []  # 新增：原始输出值（未离散化）
        
        batch_num = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                total_loss_local = 0
                sub_batch_num = 0
                for sub_batch in batch:
                     # 数据准备
                    scores = sub_batch["scores"].to(self.device).long()  # 真实标签
                    # print("scores:", scores)
                    if self.use_pitch_feature:
                        mask = sub_batch["masks_pitch"].to(self.device)
                    else:
                        mask = sub_batch["masks"].to(self.device)
                    
                    # 前向传播
                    # print("run model")
                    if self.data_type=="use_tensor":
                        outputs = self.model(
                            sub_batch["audio_feats_0"].to(self.device),
                            sub_batch["audio_feats_1"].to(self.device),
                            mask,
                        )
                    elif self.use_pitch_feature:
                        outputs = self.model(
                            audio_feats_0_overtone = sub_batch["audio_feats_0_overtone"].float().to(self.device),
                            audio_feats_0_f0 = sub_batch["audio_feats_0_f0_pitch"].long().to(self.device),
                            audio_feats_0_volume = sub_batch["audio_feats_0_f0_pitch_volume"].float().to(self.device),
                            audio_feats_1_overtone = sub_batch["audio_feats_1_overtone"].float().to(self.device),
                            audio_feats_1_f0 = sub_batch["audio_feats_1_f0_pitch"].long().to(self.device),
                            audio_feats_1_volume = sub_batch["audio_feats_1_f0_pitch_volume"].float().to(self.device),
                            mask = mask,
                        )
                    else:
                        outputs = self.model(
                            sub_batch["audio_feats_0_whisper"].to(self.device),
                            sub_batch["audio_feats_0_hubert"].to(self.device),
                            sub_batch["audio_feats_0_f0"].to(self.device),
                            sub_batch["audio_feats_1_whisper"].to(self.device),
                            sub_batch["audio_feats_1_hubert"].to(self.device),
                            sub_batch["audio_feats_1_f0"].to(self.device),
                            mask,
                        )
                    
                    if torch.isnan(outputs).any().item():
                        print("NaN in outputs")
                    outputs = outputs.masked_fill(torch.isnan(outputs), 0)
                    # print("run model done")
                    
                    # 损失计算（保持与训练时相同的处理方式）
                    masked_outputs = masked_average(outputs, mask)
                    all_masked_outputs.append(masked_outputs.cpu())  # 保存原始输出值
                    
                    # 原始预测值（离散化后的分类结果）
                    preds = masked_outputs.argmax(dim=1)
                    all_preds.append(preds.cpu())  # 保存为CPU张量
                    all_scores.append(scores.cpu())  # 保存真实标签
                    
                    # 统计逻辑（保持原有统计不变）
                    total_correct += (preds == scores).sum().item()
                    total_correct_range += ((preds - scores).abs()<=1).sum().item()
                    total_elements += scores.size(0)
                    
                    # 损失计算（保持原有逻辑不变）
                    loss = self.criterion(masked_outputs, scores)
                    total_loss_local += loss.item()
                    sub_batch_num += 1
                    
                total_loss += total_loss_local / sub_batch_num
                batch_num += 1
        
        # 数据拼接与预处理
        all_preds = torch.cat(all_preds).numpy()
        all_scores = torch.cat(all_scores).numpy()
        all_masked_outputs = torch.cat(all_masked_outputs).numpy()
        
        # 计算Pearson相关系数（使用原始输出值）
        def pearsonr(x, y):
            # 手动实现避免依赖scipy
            x_mean = np.mean(x, axis=0)
            y_mean = np.mean(y, axis=0)
            xy_cov = np.sum((x - x_mean) * (y - y_mean))
            x_std = np.sqrt(np.sum((x - x_mean)**2))
            y_std = np.sqrt(np.sum((y - y_mean)**2))
            return xy_cov / (x_std * y_std)
        
        pearson_corr = pearsonr(all_preds, all_scores)  # 核心计算
        
        # 原始指标计算（保持原有逻辑不变）
        accuracy = total_correct / total_elements
        accuracy_range = total_correct_range / total_elements
        avg_loss = total_loss / batch_num
        preds_var = np.var(all_preds)
        
        return avg_loss, accuracy, accuracy_range, preds_var, pearson_corr
    
    def save_model(self, filename):
        """保存模型和处理器（只在主进程调用）"""
        # 如果是DDP模型，获取原始模型
        model_to_save = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        os.makedirs(self.save_dir, exist_ok=True)
        model_to_save.save_model(os.path.join(self.save_dir, filename))
