import os
import torch
import json
import pickle as pickle
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import librosa
import audioread
import numpy
import warnings
from einops import rearrange
import pyloudnorm as pyln
import audioscore.audio_cut
from typing import Dict, List, Any
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch.serialization.add_safe_globals([
#     numpy.core.multiarray._reconstruct,
#     numpy.core.multiarray.scalar,
#     numpy.dtype,
#     numpy.ndarray,
#     numpy.dtypes.Float32DType])

class AudioDataset_pkl_base(Dataset):
    def __init__(self, dir_path, target_file, use_same_data=False):
        
        self.target = dict()
        with open(target_file, "r") as f:
            target = json.load(f)
            for item in target:
                audio_id = item["audio"].split("/")[-1].split(".")[0]
                self.target[audio_id] = item["score"]

        # walk dir_path
        self.files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), "rb") as f:
                        # data = pickle.load(f)
                        # self.data.extend(data)
                        audio_id = file.split("/")[-1].split("对比数据")[0]
                        if audio_id in self.target or use_same_data:
                            self.files.append(os.path.join(root, file))


        print(f"Found {len(self.files)} files in {dir_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data_bags = pickle.load(open(file_path, "rb"))
        audio_id = file_path.split("/")[-1].split("对比数据")[0]
        if audio_id in self.target:
            score = self.target[audio_id] - 1 #id要减一
        else:
            score = 4
            for i in range(len(data_bags)):
                data_bags[i]["用户音频特征"] = data_bags[i]["原唱音频特征"]
                data_bags[i]["用户前九泛音音量线"] = data_bags[i]["原唱前九泛音音量线"]
                data_bags[i]["用户音高线"] = data_bags[i]["原唱音高线"]
                data_bags[i]["用户音量线"] = data_bags[i]["原唱音量线"]
        return data_bags, score, file_path
        
class AudioDataset_fullaudio(Dataset):
    def __init__(self, dir_path, target_file, use_same_data=False):
        
        self.target = dict()
        with open(target_file, "r") as f:
            target = json.load(f)
            for item in target:
                audio_id = item["audio"].split("/")[-1].split(".")[0]
                self.target[audio_id] = item["score"]

        # walk dir_path
        self.files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), "rb") as f:
                        # data = pickle.load(f)
                        # self.data.extend(data)
                        audio_id = file.split("/")[-1].split("对比数据")[0]
                        if audio_id in self.target or use_same_data:
                            self.files.append(os.path.join(root, file))


        print(f"Found {len(self.files)} files in {dir_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data_bags = pickle.load(open(file_path, "rb"))
        audio_id = file_path.split("/")[-1].split("对比数据")[0]
        if audio_id in self.target:
            score = self.target[audio_id] - 1 #id要减一
        else:
            score = 4
            data_bags["res_user"] = data_bags["res_ori"]
        return data_bags, score, file_path
        
class AudioDataset_tensor(Dataset):
    def __init__(self, dir_path, target_file, use_same_data=False):
        
        self.target = dict()
        with open(target_file, "r") as f:
            target = json.load(f)
            for item in target:
                audio_id = item["audio"].split("/")[-1].split(".")[0]
                self.target[audio_id] = item["score"]

        # walk dir_path
        self.files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), "rb") as f:
                        # data = pickle.load(f)
                        # self.data.extend(data)
                        audio_id = file.split("/")[-1].split("对比数据")[0]
                        if audio_id in self.target or use_same_data:
                            self.files.append(os.path.join(root, file))


        print(f"Found {len(self.files)} files in {dir_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            data_bags = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
            audio_id = file_path.split("/")[-1].split("对比数据")[0]
            if audio_id in self.target:
                score = self.target[audio_id] - 1 #id要减一
            else:
                score = 4
                data_bags["res_user"] = data_bags["res_ori"]
                if "audio_ori" in data_bags:
                    data_bags["audio_user"] = data_bags["audio_ori"]
                if "wespeaker_ori" in data_bags:
                    data_bags["wespeaker_user"] = data_bags["wespeaker_ori"]
                if "samoye_ori" in data_bags:
                    data_bags["samoye_user"] = data_bags["samoye_ori"]
                if "spk_ori" in data_bags:
                    data_bags["spk_user"] = data_bags["spk_ori"]
            
            return data_bags, score, file_path
        except Exception as e:
            print(file_path)
            print(e)
            raise Exception(f"Error in loading {file_path}")

class AudioDataset_pairwise(Dataset):
    def __init__(self, target_file):
        
        self.target = dict()
        """
        数据集格式
        [
            [
                //一个数组为一组，每一组按顺序排
                {"audio_file":"音频1","muq":"音频1muq"},
            ],
            [],
            ...
        ]
        """
        with open(target_file, "r") as f:
            self.target = json.load(f)
        
        self.pairs = []
        
        for group_idx, group in enumerate(self.target):
            n = len(group)
            # 生成所有有序对（i,j）且i≠j
            ordered_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            # random.shuffle(ordered_pairs)  # 随机打乱顺序
            # print(ordered_pairs)
            
            # 生成标签：1表示(i,j)顺序，0表示(j,i)顺序
            for i, j in ordered_pairs:

                if group[i]["score"]==group[j]["score"]:
                    continue

                label = 1 if i < j else 0  # 按索引顺序定义前后关系
                self.pairs.append({
                    "group_idx": group_idx,
                    "i": i,
                    "j": j,
                    "label": label
                })
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        group = self.target[pair["group_idx"]]
        
        # 加载两个音频的muq特征
        muq_i = torch.load(os.path.join(ROOT_DIR, group[pair["i"]]["muq"]), weights_only=True, map_location="cpu")["muq"].view(-1,1024)
        muq_j = torch.load(os.path.join(ROOT_DIR, group[pair["j"]]["muq"]), weights_only=True, map_location="cpu")["muq"].view(-1,1024)

        # print(muq_i.shape, muq_j.shape)
        
        return muq_i, muq_j, group[pair["i"]]["score"], group[pair["j"]]["score"], pair["label"]

    @staticmethod
    def collate_fn(batch):
        muq_a = []
        muq_b = []
        labels = []
        a_lengths = []  # 记录muq_a的原始长度
        b_lengths = []  # 记录muq_b的原始长度
        scores_i = []
        scores_j = []
        
        for item in batch:
            muq_i, muq_j, score_i, score_j, label = item
            muq_a.append(muq_i)
            muq_b.append(muq_j)
            labels.append(label)
            a_lengths.append(muq_i.size(0))  # 记录序列长度
            b_lengths.append(muq_j.size(0))
            scores_i.append(score_i)
            scores_j.append(score_j)
        
        # 填充变长序列
        muq_a_padded = torch.nn.utils.rnn.pad_sequence(muq_a, batch_first=True)
        muq_b_padded = torch.nn.utils.rnn.pad_sequence(muq_b, batch_first=True)
        
        # 生成mask
        max_len_a = muq_a_padded.size(1)
        max_len_b = muq_b_padded.size(1)
        
        mask_a = torch.arange(max_len_a).expand(len(muq_a), max_len_a) < torch.tensor(a_lengths).unsqueeze(1)
        mask_b = torch.arange(max_len_b).expand(len(muq_b), max_len_b) < torch.tensor(b_lengths).unsqueeze(1)
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            "muq_a": muq_a_padded,       # 形状 [B, max_seq_a, 1024]
            "muq_b": muq_b_padded,       # 形状 [B, max_seq_b, 1024]
            "mask_a": mask_a,            # 形状 [B, max_seq_a]
            "mask_b": mask_b,            # 形状 [B, max_seq_b]
            "labels": labels,             # 形状 [B]
            "score_i": scores_i,
            "score_j": scores_j
        }

class AudioDataset_hybrid_multifile(Dataset):
    #pkl和mert一起使用的数据集
    def __init__(self, target_file, use_same_data=False):
        self.target = dict()

        smy_dir_path = os.path.join(ROOT_DIR, "data", "processed_contact")
        muq_dir_path = os.path.join(ROOT_DIR, "data", "processed_muq")
        mert_dir_path = os.path.join(ROOT_DIR, "data", "processed_mert")

        print(f"Loading target from {target_file}")

        with open(target_file, "r") as f:
            target = json.load(f)
            for item in target:
                audio_id = item["audio"].split("/")[-1].split(".")[0]
                self.target[audio_id] = item["score"]
        print(f"Found {len(self.target)} items in {target_file}")
        # walk dir_path
        self.files_samoye = dict()
        for file in self.walk_dir(smy_dir_path):
            if file.endswith(".pkl"):
                audio_id = file.split("/")[-1].split("对比数据")[0]
                if audio_id in self.target or use_same_data:
                    self.files_samoye[audio_id] = file
        print(f"Found {len(self.files_samoye)} files in {smy_dir_path}")

        self.files_muq = dict()
        for file in self.walk_dir(muq_dir_path):
            if file.endswith(".pkl"):
                audio_id = file.split("/")[-1].split("对比数据")[0]
                if audio_id in self.target or use_same_data and file in self.files_samoye:
                    self.files_muq[audio_id] = [self.files_samoye[audio_id], file]
        print(f"Found {len(self.files_muq)} files in {muq_dir_path}")

        self.files_mert = dict()
        for file in self.walk_dir(mert_dir_path):
            if file.endswith(".pkl"):
                audio_id = file.split("/")[-1].split("对比数据")[0]
                if audio_id in self.target or use_same_data and file in self.files_muq:
                    self.files_mert[audio_id] = [self.files_muq[audio_id][0], self.files_muq[audio_id][1], file]
        print(f"Found {len(self.files_mert)} files in {mert_dir_path}")
        
        self.files = []
        for audio_id in self.files_mert:
            self.files.append({
                "audio_id": audio_id,
                "samoye": self.files_mert[audio_id][0],
                "muq": self.files_mert[audio_id][1],
                "mert": self.files_mert[audio_id][2],
            })
        print(f"Found {len(self.files)} files")
    
    def walk_dir(self, dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                yield os.path.join(root, file)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data_bags_smy = pickle.load(open(file_path["samoye"], "rb"))
        audio_id = file_path["audio_id"]
        if audio_id in self.target:
            score = self.target[audio_id] - 1 #id要减一
        else:
            score = 4
            data_bags_smy["res_user"] = data_bags_smy["res_ori"]

        data_bags_muq = torch.load(file_path["muq"], map_location=torch.device('cpu'), weights_only=True)
        data_bags_mert = torch.load(file_path["mert"], map_location=torch.device('cpu'), weights_only=True)

        return {
            "samoye_f0": {
                "ori":data_bags_smy["res_ori"]["f0"],
                "user":data_bags_smy["res_user"]["f0"]
            },
            "samoye_hubert": {
                "ori":data_bags_smy["res_ori"]["hubert"],
                "user":data_bags_smy["res_user"]["hubert"]
            },
            "samoye_whisper": {
                "ori":data_bags_smy["res_ori"]["whisper"],
                "user":data_bags_smy["res_user"]["whisper"]
            },
            "mert": {
                "ori":data_bags_mert["res_ori"],
                "user":data_bags_mert["res_user"]
            },
            "muq": {
                "ori":data_bags_muq["res_ori"],
                "user":data_bags_muq["res_user"]
            },
            "score": score
        }
    
    @staticmethod
    def collect_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        收集函数，将batch中的数据按seq_size方向拼接，不足部分补0，并生成mask
        
        Args:
            batch: 包含多个样本的列表，每个样本是一个字典
            
        Returns:
            处理后的batch字典，包含拼接后的数据和对应的mask
        """
        result = {}
        
        # 获取所有特征名称
        feature_names = list(batch[0].keys())
        
        for feature_name in feature_names:
            feature_data = batch[0][feature_name]
            
            # 检查是否是嵌套字典结构（包含'ori'和'user'）
            if isinstance(feature_data, dict) and 'ori' in feature_data and 'user' in feature_data:
                result[feature_name] = {}
                
                # 处理ori和user数据
                for key in ['ori', 'user']:
                    # 收集所有样本的该特征数据
                    all_samples = [sample[feature_name][key] for sample in batch]
                    
                    # 获取该特征的维度信息
                    sample_shape = all_samples[0].shape
                    
                    # 确定拼接的维度（seq_size所在的维度）
                    if len(sample_shape) == 1:
                        # 一维数据，如f0特征 [seq_size*4]
                        seq_dim = 0
                        feature_dim = 1
                        target_shape = (-1, 1)  # 转换为二维便于处理
                    elif len(sample_shape) == 2:
                        # 二维数据，如whisper、hubert、mert特征 [seq_size*N, feature_dim]
                        seq_dim = 0
                        feature_dim = sample_shape[1]
                        target_shape = (-1, feature_dim)
                    elif len(sample_shape) == 3:
                        # 三维数据，如muq特征 [1, seq_size, feature_dim]
                        seq_dim = 1
                        feature_dim = sample_shape[2]
                        target_shape = (1, -1, feature_dim)
                    else:
                        raise ValueError(f"不支持的张量维度: {sample_shape}")
                    
                    # 找到最大序列长度
                    max_seq_len = max(sample.shape[seq_dim] for sample in all_samples)
                    
                    # 填充和拼接
                    padded_samples = []
                    masks = []
                    
                    for sample in all_samples:
                        current_seq_len = sample.shape[seq_dim]
                        
                        # 根据维度进行填充
                        if len(sample_shape) == 1:
                            # 一维数据填充
                            padding_size = max_seq_len - current_seq_len
                            if padding_size > 0:
                                padded_sample = torch.nn.functional.pad(sample.unsqueeze(1), 
                                                    (0, 0, 0, padding_size), 
                                                    mode='constant', value=0).squeeze(1)
                            else:
                                padded_sample = sample
                            padded_samples.append(padded_sample)
                            
                            # 生成mask [max_seq_len]
                            mask = torch.zeros(max_seq_len, dtype=torch.bool)
                            mask[:current_seq_len] = True
                            masks.append(mask)
                            
                        elif len(sample_shape) == 2:
                            # 二维数据填充
                            padding_size = max_seq_len - current_seq_len
                            if padding_size > 0:
                                padded_sample = torch.nn.functional.pad(sample, 
                                                    (0, 0, 0, padding_size), 
                                                    mode='constant', value=0)
                            else:
                                padded_sample = sample
                            padded_samples.append(padded_sample)
                            
                            # 生成mask [max_seq_len]
                            mask = torch.zeros(max_seq_len, dtype=torch.bool)
                            mask[:current_seq_len] = True
                            masks.append(mask)
                            
                        elif len(sample_shape) == 3:
                            # 三维数据填充
                            padding_size = max_seq_len - current_seq_len
                            if padding_size > 0:
                                # 在序列维度上填充
                                padded_sample = torch.nn.functional.pad(sample, 
                                                    (0, 0, 0, padding_size, 0, 0), 
                                                    mode='constant', value=0)
                            else:
                                padded_sample = sample
                            padded_samples.append(padded_sample)
                            
                            # 生成mask [1, max_seq_len]
                            mask = torch.zeros((1, max_seq_len), dtype=torch.bool)
                            mask[0, :current_seq_len] = True
                            masks.append(mask)
                    
                    # 拼接所有样本
                    stacked_data = torch.stack(padded_samples, dim=0)
                    stacked_mask = torch.stack(masks, dim=0)
                    
                    result[feature_name][key] = stacked_data
                    result[feature_name][f'{key}_mask'] = stacked_mask
                    
            else:
                # 如果不是嵌套字典结构，直接处理
                all_samples = [sample[feature_name] for sample in batch]
                try:
                    result_tmp = torch.tensor(all_samples)
                except:
                    result_tmp = [all_samples]

                result[feature_name] = result_tmp
        
        return result

class AudioDataset_scoring_hybrid_feature(Dataset):
    def __init__(self, target_file):
        
        self.target = dict()
        """
        数据集格式
        [
            [
                //一个数组为一组，每一组按顺序排，第一个最接近1，最后一个最接近0
                {"audio_file":"音频1","muq":"音频1muq"},
                {"audio_file":"音频2","muq":"音频2muq"},
                ...
            ],
            [],
            ...
        ]
        """
        with open(target_file, "r") as f:
            self.target = json.load(f)
        
        self.samples = []
        
        for group_idx, group in enumerate(self.target):
            n = len(group)
            if n < 2:  # 如果组内样本少于2个，跳过
                continue
                
            # 为组内每个样本计算分数：从1到0线性递减
            for sample_idx, sample in enumerate(group):
                # 计算分数：第一个样本为1，最后一个样本为0，中间线性递减
                score = 1.0 - (sample_idx / (n - 1)) if n > 1 else 0.5
                
                basename = sample["muq"].split("/")[-1].split(".")[0]
                hybrid_path = "data/sort/data/audio_processed_hybrid/"+basename+".pt"
                if os.path.exists(os.path.join(ROOT_DIR, hybrid_path)):
                    self.samples.append({
                        "group_idx": group_idx,
                        "sample_idx": sample_idx,
                        "score": score,
                        "muq_path": sample["muq"],
                        "hybrid_path": hybrid_path
                    })
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载音频的muq特征
        feature = torch.load(os.path.join(ROOT_DIR, sample["hybrid_path"]), weights_only=True, map_location="cpu")
        feature["score"] = sample["score"]

        return feature

    @staticmethod
    def collate_fn(batch):
        """
        简单手动实现的版本，避免复杂逻辑
        """
        # 获取每个特征的最大序列长度
        max_seq_pitch = max([item["samoye_pitch"].shape[0] for item in batch])
        max_seq_whisper = max([item["samoye_whisper"].shape[0] for item in batch])
        max_seq_hubert = max([item["samoye_hubert"].shape[0] for item in batch])
        max_seq_mert = max([item["mert"].shape[0] for item in batch])
        max_seq_muq = max([item["muq"].shape[1] for item in batch])
        
        batch_size = len(batch)
        
        # 初始化输出tensors
        pitch_batch = torch.zeros(batch_size, max_seq_pitch)
        whisper_batch = torch.zeros(batch_size, max_seq_whisper, 1280)
        hubert_batch = torch.zeros(batch_size, max_seq_hubert, 256)
        mert_batch = torch.zeros(batch_size, max_seq_mert, 1024)
        muq_batch = torch.zeros(batch_size, 1, max_seq_muq, 1024)
        if "score" in batch[0]:
            score_batch = torch.zeros(batch_size)
        
        # 初始化mask tensors
        pitch_mask = torch.zeros(batch_size, max_seq_pitch)
        whisper_mask = torch.zeros(batch_size, max_seq_whisper)
        hubert_mask = torch.zeros(batch_size, max_seq_hubert)
        mert_mask = torch.zeros(batch_size, max_seq_mert)
        muq_mask = torch.zeros(batch_size, max_seq_muq)
        
        for i, item in enumerate(batch):
            # 转换为tensor
            pitch_data = torch.tensor(audioscore.audio_cut.f0_to_coarse(item["samoye_pitch"])) \
                if not isinstance(item["samoye_pitch"], torch.Tensor) else \
                    audioscore.audio_cut.f0_to_coarse(item["samoye_pitch"])
            whisper_data = torch.tensor(item["samoye_whisper"]) if not isinstance(item["samoye_whisper"], torch.Tensor) else item["samoye_whisper"]
            hubert_data = torch.tensor(item["samoye_hubert"]) if not isinstance(item["samoye_hubert"], torch.Tensor) else item["samoye_hubert"]
            mert_data = torch.tensor(item["mert"]) if not isinstance(item["mert"], torch.Tensor) else item["mert"]
            muq_data = torch.tensor(item["muq"]) if not isinstance(item["muq"], torch.Tensor) else item["muq"]
            if "score" in batch[0]:
                score_data = torch.tensor(item["score"]) if not isinstance(item["score"], torch.Tensor) else item["score"]
            
            # 填充数据
            pitch_len = pitch_data.shape[0]
            pitch_batch[i, :pitch_len] = pitch_data
            pitch_mask[i, :pitch_len] = 1.0
            
            whisper_len = whisper_data.shape[0]
            whisper_batch[i, :whisper_len] = whisper_data
            whisper_mask[i, :whisper_len] = 1.0
            
            hubert_len = hubert_data.shape[0]
            hubert_batch[i, :hubert_len] = hubert_data
            hubert_mask[i, :hubert_len] = 1.0
            
            mert_len = mert_data.shape[0]
            mert_batch[i, :mert_len] = mert_data
            mert_mask[i, :mert_len] = 1.0
            
            muq_len = muq_data.shape[1]
            muq_batch[i, :, :muq_len] = muq_data
            muq_mask[i, :muq_len] = 1.0

            if "score" in batch[0]:
                score_batch[i] = score_data
        res = {
            "samoye_pitch": pitch_batch,
            "samoye_whisper": whisper_batch,
            "samoye_hubert": hubert_batch,
            "mert": mert_batch,
            "muq": muq_batch,
            "mask": {
                "samoye_pitch": pitch_mask,
                "samoye_whisper": whisper_mask,
                "samoye_hubert": hubert_mask,
                "mert": mert_mask,
                "muq": muq_mask
            }
        }
        if "score" in batch[0]:
            res["score"] = score_batch
        return res

class AudioDataset_pairwise_hybrid_feature(Dataset):
    def __init__(self, target_file):
        
        self.target = dict()
        """
        数据集格式
        [
            [
                //一个数组为一组，每一组按顺序排
                {"audio_file":"音频1","muq":"音频1muq"},
            ],
            [],
            ...
        ]
        """
        with open(target_file, "r") as f:
            self.target = json.load(f)
        
        self.pairs = []
        self.hybrid_feature_files = {}
        
        for group_idx, group in enumerate(self.target):
            n = len(group)
            # 生成所有有序对（i,j）且i≠j
            ordered_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            # random.shuffle(ordered_pairs)  # 随机打乱顺序
            # print(ordered_pairs)
            
            # 生成标签：1表示(i,j)顺序，0表示(j,i)顺序
            for i, j in ordered_pairs:
                label = 1 if i < j else 0  # 按索引顺序定义前后关系

                hybrid_feature_i = self.get_hybrid_path(group[i]["muq"])
                hybrid_feature_j = self.get_hybrid_path(group[j]["muq"])

                if group[i]["score"]==group[j]["score"]:
                    continue

                if os.path.exists(hybrid_feature_i) and os.path.exists(hybrid_feature_j):
                    self.pairs.append({
                        "group_idx": group_idx,
                        "i": i,
                        "j": j,
                        "hybrid_feature_i": hybrid_feature_i,
                        "hybrid_feature_j": hybrid_feature_j,
                        "label": label
                    })
    
    @staticmethod
    def get_hybrid_path(muq_path):
        basename = muq_path.split("/")[-1].split(".")[0]
        hybrid_path = "data/sort/data/audio_processed_hybrid/"+basename+".pt"
        return hybrid_path

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 加载两个音频的muq特征
        feature_i = torch.load(os.path.join(ROOT_DIR, pair["hybrid_feature_i"]), weights_only=True, map_location="cpu")
        feature_j = torch.load(os.path.join(ROOT_DIR, pair["hybrid_feature_j"]), weights_only=True, map_location="cpu")

        # print(muq_i.shape, muq_j.shape)
        
        return feature_i, feature_j, pair["label"]
    
    @staticmethod
    def collate_fn(batch):
        feature_i = [item[0] for item in batch]
        feature_j = [item[1] for item in batch]
        
        return {
            "feature_a": AudioDataset_scoring_hybrid_feature.collate_fn(feature_i),
            "feature_b": AudioDataset_scoring_hybrid_feature.collate_fn(feature_j),
            "labels": torch.tensor([item[2] for item in batch])
        }
    
class AudioDataset_pairwise_audio(Dataset):
    def __init__(self, target_file):
        
        self.target = dict()
        """
        数据集格式
        [
            [
                //一个数组为一组，每一组按顺序排
                {"audio_file":"音频1","muq":"音频1muq"},
            ],
            [],
            ...
        ]
        """
        with open(target_file, "r") as f:
            self.target = json.load(f)
        
        self.pairs = []
        self.hybrid_feature_files = {}
        
        # print("loading")
        self.file_exist_cache = {}
        
        for group_idx, group in enumerate(self.target):
            n = len(group)
            # 生成所有有序对（i,j）且i≠j
            ordered_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            # random.shuffle(ordered_pairs)  # 随机打乱顺序
            # print(ordered_pairs)
            # print(len(ordered_pairs))
            # 生成标签：1表示(i,j)顺序，0表示(j,i)顺序
            for i, j in ordered_pairs:
                # print(i,j)
                label = 1 if i < j else 0  # 按索引顺序定义前后关系

                audio_i = group[i]["audio_file"]
                audio_j = group[j]["audio_file"]

                if self.file_exist(audio_i) and self.file_exist(audio_j):
                    self.pairs.append({
                        "group_idx": group_idx,
                        "i": i,
                        "j": j,
                        "audio_i": audio_i,
                        "audio_j": audio_j,
                        "label": label
                    })
        print("load pairs:", len(self.pairs))
    
    def file_exist(self, path):
        if path in self.file_exist_cache:
            return self.file_exist_cache[path]
        res = os.path.exists(path)
        self.file_exist_cache[path] = res
        return res

    @staticmethod
    def _normalize_loudness(audio: torch.Tensor, sr: int) -> torch.Tensor:
        meter = pyln.Meter(sr)
        audio_npd = rearrange(audio, "c n -> n c").numpy()
        loudness = meter.integrated_loudness(audio_npd)
        audio_norm = pyln.normalize.loudness(audio_npd, loudness, -16)
        return rearrange(torch.from_numpy(audio_norm), "n c -> c n").float()

    @staticmethod
    def load_m4a_as_tensor(file_path, target_sr=24000):
        """
        加载M4A音频文件并转换为PyTorch张量
        
        参数:
            file_path: M4A文件路径
            target_sr: 目标采样率，默认24000Hz
        
        返回:
            audio_tensor: PyTorch张量，形状为(1, samples) - 单声道
            target_sr: 实际采样率
        """
        # 使用librosa加载音频文件
        # sr=None表示保持原始采样率，但我们会重采样到目标采样率
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with audioread.audio_open(file_path) as input_file:
                audio, original_sr = librosa.load(input_file, sr=target_sr, mono=True)
            audio = audioscore.audio_cut.random_cut_audio(audio, original_sr, 60)[0]
        # 将numpy数组转换为PyTorch张量
        audio_tensor = torch.from_numpy(audio).float()
        
        # 确保是单声道，添加通道维度
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # 形状变为(1, samples)
        
        # audio_tensor = AudioDataset_pairwise_audio._normalize_loudness(audio_tensor, original_sr)

        return audio_tensor, target_sr

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 加载两个音频的muq特征
        feature_i = AudioDataset_pairwise_audio.load_m4a_as_tensor(os.path.join(ROOT_DIR, pair["audio_i"]))[0]
        feature_j = AudioDataset_pairwise_audio.load_m4a_as_tensor(os.path.join(ROOT_DIR, pair["audio_j"]))[0]

        # print(muq_i.shape, muq_j.shape)
        
        return feature_i, feature_j, pair["label"]
    
    @staticmethod
    def collate_fn(batch):
        feature_i = [item[0] for item in batch]
        feature_j = [item[1] for item in batch]
        
        return {
            "feature_a": feature_i,
            "feature_b": feature_j,
            "labels": torch.tensor([item[2] for item in batch])
        }