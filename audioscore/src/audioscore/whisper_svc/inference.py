import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import librosa

from whisper_svc.model import Whisper, ModelDimensions
from whisper_svc.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path, device) -> Whisper:
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    # print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    # torch.save({
    #     'dims': checkpoint["dims"],
    #     'model_state_dict': model.state_dict(),
    # }, "large-v2.pt")
    return model


def pred_ppg(whisper: Whisper, wavPath, ppgPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while (idx_s + 15 * 16000 < audln):
        short = audio[idx_s:idx_s + 15 * 16000]
        idx_s = idx_s + 15 * 16000
        ppgln = 15 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    np.save(ppgPath, ppg_a, allow_pickle=False)

class WhisperInference:
    def __init__(self, model_path, device):
        self.whisper = load_model(model_path, device)
        self.device = device
        self.model_path = model_path

    def inference_file(self, wavPath: str, ppgPath: str):
        pred_ppg(self.whisper, wavPath, ppgPath, self.device)
    
    def inference(self, audio: np.ndarray, sr: int = 16000):
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audln = audio.shape[0]
        ppg_list = []  # 改为存储Tensor的列表
        idx_s = 0
        
        while idx_s + 15 * 16000 < audln:
            short = audio[idx_s:idx_s + 15 * 16000]
            idx_s += 15 * 16000
            ppgln = 15 * 16000 // 320
            
            mel = log_mel_spectrogram(short).to(self.device)
            if self.device != "cpu":
                mel = mel.half()
                
            with torch.no_grad():
                mel = mel + torch.randn_like(mel) * 0.1
                # 直接保留Tensor，不再转为NumPy
                ppg = self.whisper.encoder(mel.unsqueeze(0)).squeeze(0)
                ppg = ppg[:ppgln].detach().cpu().float()  # [length, dim=1024]
                ppg_list.append(ppg)
                
        if idx_s < audln:
            short = audio[idx_s:audln]
            ppgln = (audln - idx_s) // 320
            
            mel = log_mel_spectrogram(short).to(self.device)
            if self.device != "cpu":
                mel = mel.half()
                
            with torch.no_grad():
                mel = mel + torch.randn_like(mel) * 0.1
                ppg = self.whisper.encoder(mel.unsqueeze(0)).squeeze(0)
                ppg = ppg[:ppgln].detach().cpu().float()  # [length, dim=1024]
                ppg_list.append(ppg)
                
        # 拼接所有Tensor片段
        return torch.cat(ppg_list, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-p", "--ppg", help="ppg", dest="ppg", required=True)
    args = parser.parse_args()
    print(args.wav)
    print(args.ppg)

    wavPath = args.wav
    ppgPath = args.ppg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper = load_model(os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"whisper_pretrain"), "large-v2.pt"), device)
    pred_ppg(whisper, wavPath, ppgPath, device)
