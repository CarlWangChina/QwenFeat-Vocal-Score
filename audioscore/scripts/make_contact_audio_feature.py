import os
import sys
import torch
import pickle
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty
import pyworld as pw

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import audioscore.dataset
import audioscore.feature

def extract_features(x, fs):
    """与原始实现保持一致的特征提取函数"""
    x = x.astype(np.float64)
    fs = int(fs)
    f0, _ = pw.dio(x, fs, frame_period=5.0)
    f0 = pw.stonemask(x, f0, _, fs)
    sp = pw.cheaptrick(x, f0, _, fs)
    ap = pw.d4c(x, f0, _, fs)
    return sp, ap

def process_file(file_path, processor, output_queue):
    """单个文件处理函数"""
    try:
        path_out = os.path.join("data/processed_contact", os.path.basename(file_path))
        
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

        whisper, hubert, f0 = processor.process_audio(audio_ori)
        cheaptrick, d4c = extract_features(audio_ori, 16000)
        res_ori = {
            "whisper": whisper,
            "hubert": hubert,
            "f0": f0,
            "d4c": d4c.astype(np.float32)
        }
        whisper, hubert, f0 = processor.process_audio(audio_user)
        cheaptrick, d4c = extract_features(audio_user, 16000)
        res_user = {
            "whisper": whisper,
            "hubert": hubert,
            "f0": f0,
            "d4c": d4c.astype(np.float32)
        }

        with open(path_out, "wb") as f:
            pickle.dump({
                "res_ori": res_ori,
                "res_user": res_user
            }, f)
            
        output_queue.put((file_path, "success"))
        
    except Exception as e:
        output_queue.put((file_path, f"error: {str(e)}"))

def worker(gpu_index, input_queue, output_queue):
    """工作进程函数"""
    torch.cuda.set_device(gpu_index)
    processor = audioscore.feature.FeatExtractor("cuda")
    while True:
        try:
            file_path = input_queue.get_nowait()
            process_file(file_path, processor, output_queue)
        except Empty:
            break

def main():
    # 创建输出目录
    os.makedirs("data/processed_contact", exist_ok=True)
    
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