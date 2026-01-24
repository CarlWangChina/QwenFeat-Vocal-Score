import os
import sys
import traceback
import torch
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
from queue import Empty
import pyworld as pw

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import audioscore.dataset
import audioscore.feature
import audioscore.model
import SongEval.model
import torch
import torchaudio
import pyloudnorm as pyln
import numpy as np
import wespeaker
import json
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# 导入音频打分相关的模块
import glob
from muq import MuQ, MuQMuLan
import librosa
from safetensors.torch import load_file
from omegaconf import OmegaConf

device = 'cuda'

def audio_conv(audio):
    audio = torchaudio.transforms.Resample(16000, 24000)(torch.tensor(audio, dtype=torch.float32)).view(1, -1).cuda()
    return audio

class AudioScorer:
    def __init__(self, checkpoint_path="src/SongEval/ckpt/model.safetensors", use_cpu=False):
        self.device = torch.device('cuda') if (torch.cuda.is_available() and (not use_cpu)) else torch.device('cpu')
        self.checkpoint_path = checkpoint_path
        self.setup_model()
        
    def setup_model(self):
        """设置打分模型"""
        train_config = OmegaConf.load('src/SongEval/config.yaml')
        self.model = SongEval.model.Generator(
            in_features=train_config.generator.in_features,
            ffd_hidden_size=train_config.generator.ffd_hidden_size,
            num_classes=train_config.generator.num_classes,
            attn_layer_num=train_config.generator.attn_layer_num,)
        self.model = self.model.to(self.device).eval()
        state_dict = load_file(self.checkpoint_path, device="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq = self.muq.to(self.device).eval()
    
    @torch.no_grad()
    def score_audio(self, audio_path_or_array):
        """对音频进行打分"""
        if isinstance(audio_path_or_array, str):
            # 输入是文件路径
            wav, sr = librosa.load(audio_path_or_array, sr=24000)
            audio = torch.tensor(wav).unsqueeze(0).to(self.device)
        else:
            # 输入是音频数组
            if isinstance(audio_path_or_array, np.ndarray):
                audio_tensor = torch.tensor(audio_path_or_array, dtype=torch.float32)
            else:
                audio_tensor = audio_path_or_array
                
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio = audio_tensor.to(self.device)
        
        # 提取特征
        output = self.muq(audio, output_hidden_states=True)
        input_feat = output["hidden_states"][6]
        
        # 进行打分
        scores_g = self.model(input_feat).squeeze(0)
        
        scores = {
            'Coherence': round(scores_g[0].item(), 4),
            'Musicality': round(scores_g[1].item(), 4),
            'Memorability': round(scores_g[2].item(), 4),
            'Clarity': round(scores_g[3].item(), 4),
            'Naturalness': round(scores_g[4].item(), 4)
        }
        
        return scores

def process_audio_scores(score_axis):
    # 初始化打分器
    scorer = AudioScorer()
    
    target_file = f"data/train_ds_4_al/denoise/{score_axis}/train_score.json"
    dir_path = "data/processed"
    targets = dict()
    with open(target_file, "r") as f:
        target = json.load(f)
        for item in target:
            audio_id = item["audio"].split("/")[-1].split(".")[0]
            targets[audio_id] = item

    # 创建结果DataFrame
    columns = ['audio_id']
    
    # 添加片段评分列
    score_columns = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']
    for col in score_columns:
        columns.extend([
            f'ori_seg_{col}_mean', f'ori_seg_{col}_std',
            f'user_seg_{col}_mean', f'user_seg_{col}_std',
            f'ori_full_{col}', f'user_full_{col}'
        ])
    
    results_df = pd.DataFrame(columns=columns)
    
    # 遍历目录
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        for file in filenames:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                audio_id = file.split(".")[0].split("对比数据")[0]
                if audio_id in targets:
                    files.append((file_path, audio_id))

    for file_path, audio_id in files:
        print(f"Processing {audio_id}")
        
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            # 存储片段评分
            ori_seg_scores = {col: [] for col in score_columns}
            user_seg_scores = {col: [] for col in score_columns}
            
            audio_ori_segments = []
            audio_user_segments = []
            
            # 处理每个片段
            for i in range(len(data)):
                # 处理原唱音频片段
                ori_audio = data[i]["原唱音频"]
                user_audio = data[i]["用户音频"]
                
                # 保存片段用于后续连接
                audio_ori_segments.append(ori_audio)
                audio_user_segments.append(user_audio)
                
                # 对片段进行打分
                ori_seg_score = scorer.score_audio(ori_audio)
                user_seg_score = scorer.score_audio(user_audio)
                
                # 收集片段评分
                for col in score_columns:
                    ori_seg_scores[col].append(ori_seg_score[col])
                    user_seg_scores[col].append(user_seg_score[col])

            # 连接所有片段形成完整音频
            full_ori_audio = np.concatenate(audio_ori_segments, axis=0)
            full_user_audio = np.concatenate(audio_user_segments, axis=0)
            
            # 对完整音频进行打分
            ori_full_score = scorer.score_audio(full_ori_audio)
            user_full_score = scorer.score_audio(full_user_audio)
            
            # 准备结果行数据
            row_data = {'audio_id': audio_id}
            
            # 添加片段评分的统计信息（平均值和标准差）
            for col in score_columns:
                # 原唱片段统计
                ori_mean = np.mean(ori_seg_scores[col])
                ori_std = np.std(ori_seg_scores[col])
                
                # 用户片段统计
                user_mean = np.mean(user_seg_scores[col])
                user_std = np.std(user_seg_scores[col])
                
                # 完整音频评分
                ori_full = ori_full_score[col]
                user_full = user_full_score[col]
                
                row_data.update({
                    f'ori_seg_{col}_mean': round(ori_mean, 4),
                    f'ori_seg_{col}_std': round(ori_std, 4),
                    f'user_seg_{col}_mean': round(user_mean, 4),
                    f'user_seg_{col}_std': round(user_std, 4),
                    f'ori_full_{col}': ori_full,
                    f'user_full_{col}': user_full
                })
            
            # 添加到DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
                
        except Exception as e:
            print(f"Error processing {audio_id}: {str(e)}")
            traceback.print_exc()
    
    # 保存结果到CSV
    output_csv = f"data/audio_scores_{score_axis}.csv"
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_csv}")
    
    return results_df

# 如果需要单独运行这个函数，可以取消下面的注释
if __name__ == '__main__':
    process_audio_scores(0)