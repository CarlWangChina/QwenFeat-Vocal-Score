import os
import sys
import torch
import pickle
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty
import pyworld as pw
from torch import distributed as dist
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import audioscore.model
import audioscore.trainer_pairwise

import argparse
import librosa
import audioread

def test(model_path, save_name, dataset_path, inverse):
    model = audioscore.model.SongEvalGenerator_audio_lora(use_grl=False)
    if model_path:
        model.load_model(model_path)
    model = model.half().cuda()
    target_sr = 24000

    with open(dataset_path) as fp:
        data_index = json.loads(fp.read())
        data = []
        for table in data_index:
            for row in table:
                with torch.no_grad():
                    with audioread.audio_open(row["audio_file"]) as input_file:
                        audio, original_sr = librosa.load(input_file, sr=target_sr, mono=True)
                        # print(audio.shape)
                        tag_score = model(
                            wavs = torch.tensor(audio).view(1,-1).half().cuda()
                        )[0].detach().cpu().item()
                        if inverse:
                            tag_score = 5 - tag_score
                        # print(tag_score)
                data.append({
                    "模型输出分数": tag_score,
                    "原分数": row["score"],
                    "audio_file": row["audio_file"],
                    "source_file": row["source_file"],
                    "song_name": row["song_name"],
                    "singer": row["singer"],
                })
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 转换为DataFrame并处理
    df = pd.DataFrame(data)
    df.to_csv(f"results/{save_name}_未排序.csv", index=False)  # 保存原始数据
    
    # 按tag_score降序排序
    sorted_df = df.sort_values(by='模型输出分数', ascending=False)
    
    # 保存排序后的结果
    sorted_df.to_csv(f"results/{save_name}_按模型输出分数排序.csv", index=False)
    print(f"保存成功，共{len(data)}条记录，最高分：{sorted_df.iloc[0]['模型输出分数']}")

if __name__=="__main__":
    model_path_1 = "ckpts/SongEvalGenerator/step_2_al_audio/best_model_step_132000"
    model_path_2 = "/data/nfs/audioscore/ckpts/SongEvalGenerator/step_1_mul_fix/best_model_step_19000"

    dataset_path = "/data/nfs/audioscore/data/sort/data_mul/index/val.json"
    test(None, "多人混合打标_songeval原模型", dataset_path, False)
    test(model_path_1, "多人混合打标_阿乐数据集lora训练muq", dataset_path, True)
    test(model_path_2, "多人混合打标_muq未参与训练", dataset_path, False)
    dataset_path = "/data/nfs/audioscore/data/sort/data/index/val.json"
    test(None, "阿乐_songeval原模型", dataset_path, False)
    test(model_path_1, "阿乐_阿乐数据集lora训练muq", dataset_path, True)
    test(model_path_2, "阿乐_muq未参与训练", dataset_path, False)