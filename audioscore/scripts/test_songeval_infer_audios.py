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

def process_audio_scores(input_dir, output_csv="audio_scores.csv"):
    """
    处理目录中的m4a文件并进行音频打分
    
    Args:
        input_dir: 包含m4a文件的目录路径
        output_csv: 输出CSV文件路径
    """
    # 初始化打分器
    scorer = AudioScorer()
    
    # 创建结果DataFrame
    score_columns = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']
    columns = ['audio_file'] + score_columns
    results_df = pd.DataFrame(columns=columns)
    
    # 遍历目录查找m4a文件
    m4a_files = []
    for root, dirs, filenames in os.walk(input_dir):
        for file in filenames:
            if file.lower().endswith(".m4a") or file.lower().endswith(".mp3"):
                file_path = os.path.join(root, file)
                m4a_files.append(file_path)
    
    print(f"Found {len(m4a_files)} m4a files in directory: {input_dir}")
    
    if len(m4a_files) == 0:
        print("No m4a files found in the specified directory.")
        return results_df
    
    # 处理每个m4a文件
    for file_path in m4a_files:
        audio_file = os.path.basename(file_path)
        print(f"Processing {audio_file}")
        
        try:
            # 使用librosa读取m4a文件
            # 直接使用AudioScorer的score_audio方法，它会内部调用librosa.load
            scores = scorer.score_audio(file_path)
            
            # 准备结果行数据
            row_data = {'audio_file': audio_file}
            row_data.update(scores)
            
            # 添加到DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
            print(f"Successfully scored {audio_file}: {scores}")
                
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            traceback.print_exc()
    
    # 保存结果到CSV
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_csv}")
    
    # 打印统计信息
    print("\nScore Statistics:")
    for col in score_columns:
        if col in results_df.columns:
            print(f"{col}: mean={results_df[col].mean():.4f}, std={results_df[col].std():.4f}")
    
    return results_df

if __name__ == '__main__':
    # 使用示例
    input_directory = "data/sort/songeval/mp3"  # 修改为您的m4a文件目录
    # input_directory = "data/sort/data/audio"  # 修改为您的m4a文件目录
    output_csv_file = "data/audio_scores_songeval_sedata_results.csv"
    
    process_audio_scores(input_directory, output_csv_file)