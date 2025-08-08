import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import librosa

from hubert import hubert_model


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


def load_model(path, device):
    model = hubert_model.hubert_soft(path)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model


def pred_vec(model, wavPath, vecPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    vec_a = []
    idx_s = 0
    while (idx_s + 20 * 16000 < audln):
        feats = audio[idx_s:idx_s + 20 * 16000]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            vec_a.extend(vec)
        idx_s = idx_s + 20 * 16000
    if (idx_s < audln):
        feats = audio[idx_s:audln]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            # print(vec.shape)   # [length, dim=256] hop=320
            vec_a.extend(vec)
    np.save(vecPath, vec_a, allow_pickle=False)

class HubertInference:
    def __init__(self, model_path, device):
        self.model = load_model(model_path, device)
        self.device = device

    def inference_file(self, wavPath:str, vecPath:str):
        pred_vec(self.model, wavPath, vecPath, self.device)
    
    def inference(self, wav_data: np.array, sr: int = 16000):
        if sr != 16000:
            wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=16000)
        audln = wav_data.shape[0]
        vec_list = []  # 存储Tensor片段的列表
        idx_s = 0
        
        while idx_s + 20 * 16000 < audln:
            # 使用wav_data而不是self.audio
            segment = wav_data[idx_s:idx_s + 20 * 16000]
            
            # 转换为Tensor并添加维度 [1, 1, T]
            feats = torch.from_numpy(segment).unsqueeze(0).unsqueeze(0).to(self.device)
            
            if self.device != "cpu":
                feats = feats.half()
                
            with torch.no_grad():
                vec = self.model.units(feats).squeeze(0)  # [1, T, D] -> [T, D]
                vec_list.append(vec.detach().cpu().float())
                
            idx_s += 20 * 16000
            
        if idx_s < audln:
            segment = wav_data[idx_s:audln]
            
            # 转换为Tensor并添加维度 [1, 1, T]
            feats = torch.from_numpy(segment).unsqueeze(0).unsqueeze(0).to(self.device)
            
            if self.device != "cpu":
                feats = feats.half()
                
            with torch.no_grad():
                vec = self.model.units(feats).squeeze(0)  # [1, T, D] -> [T, D]
                vec_list.append(vec.detach().cpu().float())
                
        # 拼接所有Tensor片段
        return torch.cat(vec_list, dim=0) if vec_list else torch.empty(0)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-v", "--vec", help="vec", dest="vec", required=True)
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)

    wavPath = args.wav
    vecPath = args.vec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert = load_model(os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"hubert_pretrain"), "hubert-soft-0d54a1f4.pt"), device)
    pred_vec(hubert, wavPath, vecPath, device)
