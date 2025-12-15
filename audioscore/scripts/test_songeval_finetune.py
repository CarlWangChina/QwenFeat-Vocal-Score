import csv
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.sampler
import tqdm
import yaml
import json
import librosa
from muq import MuQ, MuQMuLan
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import audioscore.model


class AudioScorer:
    def __init__(self, checkpoint_path="src/SongEval/ckpt/model.safetensors", use_cpu=False):
        self.device = torch.device('cuda') if (torch.cuda.is_available() and (not use_cpu)) else torch.device('cpu')
        
        model = audioscore.model.SongEvalGenerator()
        model.load_state_dict(torch.load("ckpts/SongEvalGenerator/best_model_step_5000/weights.pt"))
        model.cuda()
        model.eval()

        self.model = model
        
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
        with torch.no_grad():
            output = self.muq(audio, output_hidden_states=True)
            res = self.model(output.last_hidden_state)
        return res.data.cpu()[0].item()

if __name__ == "__main__":
    scores = {}
    with open("data/sort/data/index/val.json") as f:
        val_data = json.load(f)
    for group in val_data:
        for song in group:
            try:
                if not song["audio_file"] in scores:
                    filename = os.path.basename(song["audio_file"])
                    print(filename)
                    score = AudioScorer().score_audio(song["audio_file"])
                    print(score)
                    scores[filename] = score
                    # break
            except Exception as e:
                print(e)
        # break
    with open("data/audio_scores_finetune_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "score"])
        for filename, score in scores.items():
            writer.writerow([filename, score])