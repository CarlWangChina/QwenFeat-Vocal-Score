import os
import sys
import torch
import pickle
import json
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty
import pandas as pd
import pyworld as pw
import subprocess
import random

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

xls_path = os.path.join(ROOT_DIR, "data", "sort", "data_mul", "xls", "index.xlsx")

df = pd.read_excel(xls_path)

score_index = {}

for index, row in df.iterrows():

    url = row["录音链接"]
    local_path = f"{ROOT_DIR}/data/sort/data/audio/"+url.split("/")[-1]
    # print(url, local_path)
    if not os.path.exists(local_path):
        print(f"下载文件: {url}")
        subprocess.run(["wget", "-O", local_path, url], check=True)
    # print(row)
    row_info = {
        "score": str(row["总分"]),
        "url": url,
        "audio_file": local_path,
        "source_file": xls_path,
        "muq": f"{ROOT_DIR}/data/sort/data/muq/"+url.split("/")[-1]+".pt",
        "sheet_name": "Sheet1",
        "song_name": str(row["歌曲名称"]),
        "singer": ""
    }
    if row["总分"] not in score_index:
        score_index[row["总分"]] = []
    score_index[row["总分"]].append(row_info)

train_set = []
val_set = []
full_set = []

for score, row_info in score_index.items():
    if len(row_info) > 2:
        train_set.extend(row_info[2:])
        val_set.extend(row_info[:2])
        # train_set.extend(row_info[:int(len(row_info) * 0.9)])
        # val_set.extend(row_info[int(len(row_info) * 0.9):])
    else:
        train_set.extend(row_info)

    full_set.extend(row_info[:])


with open(os.path.join(ROOT_DIR, "data", "sort", "data_mul", "index", "train.json"), "w") as f:
    json.dump([train_set], f, ensure_ascii=False, indent=4)

with open(os.path.join(ROOT_DIR, "data", "sort", "data_mul", "index", "val.json"), "w") as f:
    json.dump([val_set], f, ensure_ascii=False, indent=4)

with open(os.path.join(ROOT_DIR, "data", "sort", "data_mul", "index", "full.json"), "w") as f:
    json.dump([full_set], f, ensure_ascii=False, indent=4)

# 与之前的数据混合
with open(os.path.join(ROOT_DIR, "data", "sort", "data", "index", "train.json"), "r") as f:
    tmp = json.load(f)
    tmp.append(train_set)
    train_set = tmp

with open(os.path.join(ROOT_DIR, "data", "sort", "data", "index", "val.json"), "r") as f:
    tmp = json.load(f)
    tmp.append(val_set)
    val_set = tmp

with open(os.path.join(ROOT_DIR, "data", "sort", "data", "index", "index.json"), "r") as f:
    tmp = json.load(f)
    tmp.append(full_set)
    full_set = tmp
    full_set_num = len([item for sublist in full_set for item in sublist])
    print("full_set_num:", full_set_num)

with open(os.path.join(ROOT_DIR, "data", "sort", "data_mul", "index", "train_mix.json"), "w") as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

with open(os.path.join(ROOT_DIR, "data", "sort", "data_mul", "index", "val_mix.json"), "w") as f:
    json.dump(val_set, f, ensure_ascii=False, indent=4)

with open(os.path.join(ROOT_DIR, "data", "sort", "data_mul", "index", "full_mix.json"), "w") as f:
    json.dump(full_set, f, ensure_ascii=False, indent=4)