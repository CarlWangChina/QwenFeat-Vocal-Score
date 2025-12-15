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
import torchaudio
import pyloudnorm as pyln
import numpy as np
import wespeaker
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
        self.muq = self.muq.to("cuda").eval()

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

        wavs = audio.unsqueeze(0).to("cuda") 
        with torch.no_grad():
            audio_embeds = self.muq(wavs, output_hidden_states=True)
        print(audio_embeds.last_hidden_state.shape)
        return audio_embeds.last_hidden_state.detach().cpu()

def process_file(file_path, processor, output_queue):
    """单个文件处理函数"""
    try:
        path_out = os.path.join("data/processed_muq", os.path.basename(file_path))
        
        if os.path.exists(path_out):
            output_queue.put((file_path, "exists"))
            return

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        audio_ori = []
        audio_user = []
            
        for i in range(len(data)):
            # 处理原唱音频
            audio_ori.append(data[i]["原唱音频"])
            audio_user.append(data[i]["用户音频"])
            # print(data[i]["原唱音频"].shape)

        audio_ori = np.concatenate(audio_ori, axis=0)
        audio_user = np.concatenate(audio_user, axis=0)
        res_ori = processor["muq"](torch.tensor(audio_ori, dtype=torch.float64), 16000)
        res_user = processor["muq"](torch.tensor(audio_user, dtype=torch.float64), 16000)
        
        torch.save({
            "res_ori": res_ori,
            "res_user": res_user,
            "audio_ori": audio_ori,
            "audio_user": audio_user,
            "wespeaker_ori": processor["wespeaker"].extract_embedding_from_pcm(torch.tensor(audio_ori).view(1,-1),16000).detach().cpu(),
            "wespeaker_user": processor["wespeaker"].extract_embedding_from_pcm(torch.tensor(audio_user).view(1,-1),16000).detach().cpu(),
            "samoye_ori": processor["samoye_encoder"].process_audio(audio_ori),
            "samoye_user": processor["samoye_encoder"].process_audio(audio_user),
            "spk_ori": processor["spk"](torch.tensor(audio_ori).view(1,-1), 16000).detach().cpu(),
            "spk_user": processor["spk"](torch.tensor(audio_user).view(1,-1), 16000).detach().cpu(),
        }, path_out)
            
        output_queue.put((file_path, "success"))
        
    except Exception as e:
        output_queue.put((file_path, f"error: {str(e)}"))
        traceback.print_exc()

def worker(gpu_index, input_queue, output_queue):
    """工作进程函数"""
    torch.cuda.set_device(gpu_index)

    encoder = {
        "muq":SimpleMuqEncoder(),
        "spk":audioscore.model.SpkEncoderHelper(),
        "wespeaker":wespeaker.load_model('chinese'),
        "samoye_encoder":audioscore.feature.FeatExtractor("cuda")
    }

    while True:
        try:
            file_path = input_queue.get_nowait()
            process_file(file_path, encoder, output_queue)
        except Empty:
            break

def main():
    # 创建输出目录
    os.makedirs("data/processed_muq", exist_ok=True)
    
    # 获取所有待处理文件
    input_files = []
    dir_path = "data/ori"
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".pkl"):
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
    main()