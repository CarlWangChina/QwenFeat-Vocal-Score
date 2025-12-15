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
import torch
import torchaudio
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class SimpleMertEncoder:
    def __init__(self,
                 feature_extractor,
                 model,
                 feature_rate: int,
                 feature_dim: int,
                 output_layer: int,
                 target_loudness: float = -16.0):
        """
        Args:
            feature_extractor: HuggingFace feature extractor
            model: HuggingFace MERT model
            feature_rate (int): 特征帧率 (Hz)
            feature_dim (int): 特征维度
            output_layer (int): 使用的 MERT 隐层层数
            target_loudness (float): 目标响度 (LUFS)
        """
        self.feature_extractor = feature_extractor
        self.model = model
        self.feature_rate = feature_rate
        self.feature_dim = feature_dim
        self.output_layer = output_layer
        self.target_loudness = target_loudness
        self.device = model.device

    def _normalize_loudness(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        meter = pyln.Meter(sr)
        audio_npd = rearrange(audio, "c n -> n c").numpy()
        loudness = meter.integrated_loudness(audio_npd)
        audio_norm = pyln.normalize.loudness(audio_npd, loudness, self.target_loudness)
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

        # 重采样到 MERT 的采样率
        target_sr = self.feature_extractor.sampling_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        # 转 mono
        if audio.dim() == 2:
            audio = torch.mean(audio, dim=0)

        # 提取特征
        inputs = self.feature_extractor(audio,
                                        sampling_rate=target_sr,
                                        return_tensors="pt",
                                        return_attention_mask=True,
                                        padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        hidden_states = self.model(**inputs, output_hidden_states=True)["hidden_states"]
        features = hidden_states[self.output_layer].squeeze(0).to("cpu")

        assert features.dim() == 2 and features.shape[-1] == self.feature_dim
        return features

def process_file(file_path, processor, output_queue):
    """单个文件处理函数"""
    try:
        path_out = os.path.join("data/processed_mert", os.path.basename(file_path))
        
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
        res_ori = processor(torch.tensor(audio_ori, dtype=torch.float64), 16000)
        res_user = processor(torch.tensor(audio_user, dtype=torch.float64), 16000)

        torch.save({
            "res_ori": res_ori,
            "res_user": res_user
        }, path_out)
            
        output_queue.put((file_path, "success"))
        
    except Exception as e:
        output_queue.put((file_path, f"error: {str(e)}"))
        traceback.print_exc()

def worker(gpu_index, input_queue, output_queue):
    """工作进程函数"""
    torch.cuda.set_device(gpu_index)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M")
    model = HubertModel.from_pretrained("m-a-p/MERT-v1-330M").to("cuda")

    encoder = SimpleMertEncoder(feature_extractor, model,
                                feature_rate=75,    # 具体看模型
                                feature_dim=1024,    # 具体看模型
                                output_layer=24)

    while True:
        try:
            file_path = input_queue.get_nowait()
            process_file(file_path, encoder, output_queue)
        except Empty:
            break

def main():
    # 创建输出目录
    os.makedirs("data/processed_mert", exist_ok=True)
    
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