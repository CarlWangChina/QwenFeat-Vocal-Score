import base64
import json
import uuid
import requests
import time
import pandas as pd
import librosa
import pyloudnorm as pyln
import csv

import os
import sys

from transformers import Qwen2AudioProcessor
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

from qwenaudio.processor import ProcessorGroup
import qwenaudio.prompts

def get_best_model(path):
    index = 1
    model_pathes = []
    for name in os.listdir(path):
        if name.startswith("best_model_epoch_"):
            model_path_tmp = f"{path}/{name}/lora_weights"
            model_pathes.append((model_path_tmp, int(name.split("_")[-1])))
    model_pathes.sort(key=lambda x: x[1])
    # while True:
    #     model_path_tmp = f"{path}/best_model_epoch_{index}/lora_weights"
    #     if os.path.exists(model_path_tmp):
    #         index += 1
    #         model_path = model_path_tmp
    #     else:
    #         break
    return model_pathes[-1][0]

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def load_model(model_path):
    model = qwenaudio.model.QwenAudioTowerScoreModel()
    model.load_ckpt(model_path)
    model.to("cuda")
    model.eval()
    return model

if __name__ == "__main__":
    models = []
    models.append(load_model("ckpts/train_ds_4_score_al/denoise/0/score/best_model_epoch/8"))
    models.append(load_model("ckpts/train_ds_4_score_al/denoise/1/score/best_model_epoch/3"))
    models.append(load_model("ckpts/train_ds_4_score_al/denoise/2/score/best_model_epoch/2"))
    models.append(load_model("ckpts/train_ds_4_score_al/denoise/3/score/best_model_epoch/5"))

    processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    
    df = pd.read_excel("/home/w-4090/projects/qwenaudio/data/score_ds_4.xlsx", sheet_name=1)

    song_score_data = dict()
    df_index = pd.read_excel("/home/w-4090/projects/qwenaudio/data/aldz_tag.xlsx", sheet_name=0)
    for index, row in df_index.iterrows():
        print(index, row)
        try:
            print(row["录音文件名"])
            songid = row["录音文件名"].split(".")[0]
            axis_al_0_score = safe_int(row["阿乐专业技巧得分"])
            axis_al_1_score = safe_int(row["阿乐情感表达得分"])
            axis_al_2_score = safe_int(row["阿乐音色与音质得分"])
            axis_al_3_score = safe_int(row["阿乐气息控制得分"])
            axis_dz_0_score = safe_int(row["东钊专业技巧得分"])
            axis_dz_1_score = safe_int(row["东钊情感表达得分"])
            axis_dz_2_score = safe_int(row["东钊音色与音质得分"])
            axis_dz_3_score = safe_int(row["东钊气息控制得分"])
            song_score_data[songid] = (axis_al_0_score, axis_al_1_score, axis_al_2_score, axis_al_3_score, 
                                       axis_dz_0_score, axis_dz_1_score, axis_dz_2_score, axis_dz_3_score)
        except:
            pass

    out_csv = open("/home/w-4090/projects/qwenaudio/data/score_audio_tower.csv", "w")
    writer = csv.writer(out_csv)

    count = 0

    for index, row in df.iterrows():
        # print(index, row)
        url = row["文件名"]
        file_base_name = url.split("/")[-2].replace("_trimmed", "")
        print(file_base_name)
        if file_base_name in song_score_data:
            count += 1
            axis_al_0_score, axis_al_1_score, axis_al_2_score, axis_al_3_score, axis_dz_0_score, axis_dz_1_score, axis_dz_2_score, axis_dz_3_score = song_score_data[file_base_name]
            
            print(url)
            # https://nfsfile.meiktv.com//data/nfs/funasr_nie/separated/htdemucs_ft/459652_250730073745009698_trimmed/vocals.m4a
            file_name = "/home/w-4090/projects/qwenaudio/data/download_music/"+file_base_name+"_trimmed.m4a"
            
            sr = processor.feature_extractor.sampling_rate
            data, sr = librosa.load(file_name, sr=sr, mono=True)

            meter = pyln.Meter(sr) # create BS.1770 meter
            loudness = meter.integrated_loudness(data)
            # loudness normalize audio to -12 dB LUFS
            data = pyln.normalize.loudness(data, loudness, -12.0)

            max_samples = sr * 30
            data = data[:max_samples]

            res_line = [file_name]
            mindex = 0

            res_line.append(axis_al_0_score)
            res_line.append(axis_dz_0_score)
            score = models[0].infer(data, processor)["score"]
            res_line.append(score)
            res_line.append("")
            res_line.append(axis_al_1_score)
            res_line.append(axis_dz_1_score)
            score = models[1].infer(data, processor)["score"]
            res_line.append(score)
            res_line.append("")
            res_line.append(axis_al_2_score)
            res_line.append(axis_dz_2_score)
            score = models[2].infer(data, processor)["score"]
            res_line.append(score)
            res_line.append("")
            res_line.append(axis_al_3_score)
            res_line.append(axis_dz_3_score)
            score = models[3].infer(data, processor)["score"]
            res_line.append(score)

            
            writer.writerow(res_line)
            # if count>4:
            #     break

    out_csv.close()