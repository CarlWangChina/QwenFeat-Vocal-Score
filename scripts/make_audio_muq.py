import os
import sys
import traceback
import torch
import pickle
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty
import pyworld as pw

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import audioscore.dataset
import audioscore.feature
import audioscore.model
import torch
import librosa
import audioread
import torchaudio
import soundfile
import pyloudnorm as pyln
import numpy as np
# import wespeaker
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from muq import MuQ

class SimpleMuqEncoder:
    def __init__(self):
        """
        Args:
            feature_extractor: HuggingFace feature extractor
            model: HuggingFace MERT model
            feature_rate (int): 特征帧率 (Hz)
            feature_dim (int): 特征维度
            output_layer (int): 使用的 MERT 隐层层数
            target_loudness (float): 目标响度 (LUFS)
        """
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq = self.muq.half().to("cuda").eval()

    def _normalize_loudness(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        meter = pyln.Meter(sr)
        audio_npd = rearrange(audio, "c n -> n c").numpy()
        loudness = meter.integrated_loudness(audio_npd)
        audio_norm = pyln.normalize.loudness(audio_npd, loudness, -16)
        return rearrange(torch.from_numpy(audio_norm), "n c -> c n").float()

    @torch.inference_mode()
    def __call__(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Args:
            audio (torch.Tensor): 音频张量 (num_channels, num_samples)
            sr (int): 采样率
        Returns:
            torch.Tensor: 特征张量 (num_frames, feature_dim)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        # 归一化响度
        audio = self._normalize_loudness(audio, sr)

        # 重采样到 Muq 的采样率
        target_sr = 24000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        # 转 mono
        if audio.dim() == 2:
            audio = torch.mean(audio, dim=0)

        wavs = audio.unsqueeze(0).half().to("cuda") 
        with torch.no_grad():
            audio_embeds = self.muq(wavs, output_hidden_states=True)
        print(audio_embeds["hidden_states"][6].shape)
        return audio_embeds["hidden_states"][6].detach().cpu().float()

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
    with audioread.audio_open(file_path) as input_file:
        audio, original_sr = librosa.load(input_file, sr=target_sr, mono=True)
    
    # 将numpy数组转换为PyTorch张量
    audio_tensor = torch.from_numpy(audio).float()
    
    # 确保是单声道，添加通道维度
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # 形状变为(1, samples)
    
    return audio_tensor, target_sr

@torch.inference_mode()
def process_file(file_path, processor, output_queue):
    """单个文件处理函数"""
    try:
        path_out = os.path.join("data/sort/data/muq", os.path.basename(file_path)+".pt")
        
        if os.path.exists(path_out):
            output_queue.put((file_path, "exists"))
            return

        audio, sr = load_m4a_as_tensor(file_path)
        print(audio.shape, sr)
        audio = audio.view(1,-1)
        muq = processor["muq"](audio, 24000)
        # wespeaker_vec = processor["wespeaker"].extract_embedding_from_pcm(audio,24000).detach().cpu()
        # samoye_vec = processor["samoye_encoder"].process_audio(audio)
        spk_emb = processor["spk"](audio, 24000).detach().cpu()
        
        torch.save({
            "muq":muq,
            # "wespeaker_ori": wespeaker_vec,
            # "samoye_ori": samoye_vec,
            "spk_ori": spk_emb,
        }, path_out)
            
        output_queue.put((file_path, "success"))
        
    except Exception as e:
        output_queue.put((file_path, f"error: {str(e)}"))
        traceback.print_exc()

def worker(gpu_index, input_queue, output_queue):
    """工作进程函数"""
    torch.cuda.set_device(gpu_index)

    print("load model")

    encoder = {
        "muq":SimpleMuqEncoder(),
        "spk":audioscore.model.SpkEncoderHelper(),
        # "wespeaker":wespeaker.load_model('chinese'),
        # "samoye_encoder":audioscore.feature.FeatExtractor("cuda")
    }
    print("load model success")

    while True:
        try:
            file_path = input_queue.get_nowait()
            process_file(file_path, encoder, output_queue)
        except Empty:
            break

def main():
    # 创建输出目录
    os.makedirs("data/sort/data/muq", exist_ok=True)
    
    # 获取所有待处理文件
    input_files = []
    dir_path = "data/sort/data/audio"
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".m4a"):
                input_files.append(os.path.join(root, file))

    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No available GPU found")

    # 创建任务队列
    input_queue = Queue()
    output_queue = Queue()
    
    # 将文件分配到队列
    for file_path in input_files:
        input_queue.put(file_path)

    # 创建进程池
    processes = []
    for gpu_idx in range(num_gpus):
        p = Process(target=worker, args=(gpu_idx, input_queue, output_queue))
        processes.append(p)
        p.start()

    # 监控进度
    total_files = len(input_files)
    processed = 0
    success_count = 0
    
    while processed < total_files:
        try:
            file_path, status = output_queue.get(timeout=1)
            if status == "success":
                success_count += 1
                print(f"Processed: {file_path} [{success_count}/{total_files}]")
            elif status == "exists":
                print(f"Skipped existing: {file_path}")
                success_count += 1
            else:
                print(f"Failed: {file_path} - {status}")
            processed += 1
        except Empty:
            continue

    # 等待所有进程结束
    for p in processes:
        p.join()

if __name__ == "__main__":
    # MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    main()