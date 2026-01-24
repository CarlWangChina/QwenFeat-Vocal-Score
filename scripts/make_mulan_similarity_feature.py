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
import json
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from muq import MuQ, MuQMuLan

device = 'cuda'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()

def audio_conv(audio):
    audio = torchaudio.transforms.Resample(16000, 24000)(torch.tensor(audio, dtype=torch.float32)).view(1, -1).cuda()
    # print(audio.shape)
    return audio

def make_mulan_similarity_feature(score_axis):
    target_file = f"data/train_ds_4_al/denoise/{score_axis}/train_score.json"
    dir_path = "data/processed"
    targets = dict()
    with open(target_file, "r") as f:
        target = json.load(f)
        for item in target:
            audio_id = item["audio"].split("/")[-1].split(".")[0]
            targets[audio_id] = item

    # walk dir_path
    files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".pkl"):
                with open(os.path.join(root, file), "rb") as f:
                    # data = pickle.load(f)
                    # data.extend(data)
                    audio_id = file.split("/")[-1].split("对比数据")[0]
                    if audio_id in target:
                        files.append(os.path.join(root, file))

    for file in files:
        with open(os.path.join(dir_path, file), "rb") as f:
            data = pickle.load(f)
            audio_id = file.split("/")[-1].split("对比数据")[0]
            if audio_id in targets:
                target_item = targets[audio_id]
                text = target_item['text']
                text = text.split("评语如下:\n")[-1].split("\n注意你也需要参考")[0]
                print(text)
                
                emb_ori = []
                emb_user = []
                    
                audio_ori = []
                audio_user = []

                with torch.no_grad():
                    for i in range(len(data)):
                        # 处理原唱音频
                        emb_ori.append(mulan(wavs=audio_conv(data[i]["原唱音频"])).detach().cpu())
                        emb_user.append(mulan(wavs=audio_conv(data[i]["用户音频"])).detach().cpu())
                        audio_ori.append(data[i]["原唱音频"])
                        audio_user.append(data[i]["用户音频"])


                    audio_ori = np.concatenate(audio_ori, axis=0)
                    audio_user = np.concatenate(audio_user, axis=0)
                    res_ori = mulan(wavs=audio_conv(audio_ori)).detach().cpu()
                    res_user = mulan(wavs=audio_conv(audio_user)).detach().cpu()

                    res_text = mulan(texts=text).detach().cpu()
                    
                res = {"ori_seg": emb_ori, "user_seg": emb_user, "ori": res_ori, "user": res_user, "text_emb": res_text, "text": text}
                # print(res)

                torch.save(res, f"data/processed_mulan/{audio_id}.pkl")

    # with open(f"data/train_ds_4_al/denoise/{score_axis}/train_score.json") as fp:
    #     train_score = json.load(fp)
    #     for item in train_score:
    #         text = item['text']
    #         print(text)

if __name__ == "__main__":
    make_mulan_similarity_feature(0)