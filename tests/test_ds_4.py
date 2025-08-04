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
    
if __name__ == "__main__":
    processor = ProcessorGroup()
    for i in range(4):
        for t in ["noise", "denoise"]:
            print("")
            print("load model:", t, i, qwenaudio.prompts.prompt_mapper_reverse[i])
            ckpt_path = f"/home/w-4090/projects/qwenaudio/ckpts/train_ds_4_al/{t}/{i}/"
            score_path = get_best_model(f"{ckpt_path}/score")
            text_path = get_best_model(f"{ckpt_path}/text")
            print(ckpt_path, score_path, text_path)
            processor.add(score_model_path=score_path, text_model_path=text_path)

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

    out_csv = open("/home/w-4090/projects/qwenaudio/data/score_ds_4_al_top2.csv", "w")
    writer = csv.writer(out_csv)
    writer.writerow([
        "文件名",
        "专业技巧得分(阿乐)",
        "专业技巧得分(东钊)",
        "二人差距",
        "专业技巧得分(生成,noise)",
        "阿乐差距(生成,noise)",
        "东钊差距(生成,noise)",
        "最小差距",
        "概率",
        "专业技巧分析(生成,noise)", 
        "专业技巧得分(生成,denoise)",
        "阿乐差距(生成,denoise)",
        "东钊差距(生成,denoise)",
        "最小差距",
        "概率",
        "专业技巧分析(生成,denoise)", 
        "情感表达得分(阿乐)",
        "情感表达得分(东钊)",
        "二人差距",
        "情感表达得分(生成,noise)",
        "阿乐差距(生成,noise)",
        "东钊差距(生成,noise)",
        "最小差距",
        "概率",
        "情感表达分析(生成,noise)",
        "情感表达得分(生成,denoise)",
        "阿乐差距(生成,denoise)",
        "东钊差距(生成,denoise)",
        "最小差距",
        "概率",
        "情感表达分析(生成,denoise)",
        "音色与音质得分(阿乐)",
        "音色与音质得分(东钊)",
        "二人差距",
        "音色与音质得分(生成,noise)",
        "阿乐差距(生成,noise)",
        "东钊差距(生成,noise)",
        "最小差距",
        "概率",
        "音色与音质分析(生成,noise)",
        "音色与音质得分(生成,denoise)",
        "阿乐差距(生成,denoise)",
        "东钊差距(生成,denoise)",
        "最小差距",
        "概率",
        "音色与音质分析(生成,denoise)",
        "气息控制得分(阿乐)", 
        "气息控制得分(东钊)",
        "二人差距",
        "气息控制得分(生成,noise)",
        "阿乐差距(生成,noise)",
        "东钊差距(生成,noise)",
        "最小差距",
        "概率",
        "气息控制分析(生成,noise)",
        "气息控制得分(生成,denoise)",
        "阿乐差距(生成,denoise)",
        "东钊差距(生成,denoise)",
        "最小差距",
        "概率",
        "气息控制分析(生成,denoise)"
    ])

    res = dict()
    res_count = dict()
    jsonl_lines = []
    res_count["0_二人差距小于2"] = 0
    res_count["0_noise_阿乐差距小于2"] = 0
    res_count["0_noise_东钊差距小于2"] = 0
    res_count["0_noise_最小差距等于0"] = 0
    res_count["0_noise_最小差距小于等于1"] = 0
    res_count["0_denoise_阿乐差距小于2"] = 0
    res_count["0_denoise_东钊差距小于2"] = 0
    res_count["0_denoise_最小差距等于0"] = 0
    res_count["0_denoise_最小差距小于等于1"] = 0
    res_count["1_二人差距小于2"] = 0
    res_count["1_noise_阿乐差距小于2"] = 0
    res_count["1_noise_东钊差距小于2"] = 0
    res_count["1_noise_最小差距等于0"] = 0
    res_count["1_noise_最小差距小于等于1"] = 0
    res_count["1_denoise_阿乐差距小于2"] = 0
    res_count["1_denoise_东钊差距小于2"] = 0
    res_count["1_denoise_最小差距等于0"] = 0
    res_count["1_denoise_最小差距小于等于1"] = 0
    res_count["2_二人差距小于2"] = 0
    res_count["2_noise_阿乐差距小于2"] = 0
    res_count["2_noise_东钊差距小于2"] = 0
    res_count["2_noise_最小差距等于0"] = 0
    res_count["2_noise_最小差距小于等于1"] = 0
    res_count["2_denoise_阿乐差距小于2"] = 0
    res_count["2_denoise_东钊差距小于2"] = 0
    res_count["2_denoise_最小差距等于0"] = 0
    res_count["2_denoise_最小差距小于等于1"] = 0
    res_count["3_二人差距小于2"] = 0
    res_count["3_noise_阿乐差距小于2"] = 0
    res_count["3_noise_东钊差距小于2"] = 0
    res_count["3_noise_最小差距等于0"] = 0
    res_count["3_noise_最小差距小于等于1"] = 0
    res_count["3_denoise_阿乐差距小于2"] = 0
    res_count["3_denoise_东钊差距小于2"] = 0
    res_count["3_denoise_最小差距等于0"] = 0
    res_count["3_denoise_最小差距小于等于1"] = 0

    count = 0

    for index, row in df.iterrows():
        # print(index, row)
        url = row["文件名"]
        file_base_name = url.split("/")[-2].replace("_trimmed", "")
        print(file_base_name)
        if file_base_name in song_score_data:
            count += 1
            axis_al_0_score, axis_al_1_score, axis_al_2_score, axis_al_3_score, axis_dz_0_score, axis_dz_1_score, axis_dz_2_score, axis_dz_3_score = song_score_data[file_base_name]
            # axis_0_score = row["专业技巧得分"]
            # axis_1_score = row["情感表达得分"]
            # axis_2_score = row["音色与音质得分"]
            # axis_3_score = row["气息控制得分"]
            print(url)
            # https://nfsfile.meiktv.com//data/nfs/funasr_nie/separated/htdemucs_ft/459652_250730073745009698_trimmed/vocals.m4a
            file_name = "/home/w-4090/projects/qwenaudio/data/download_music/"+file_base_name+"_trimmed.m4a"
            
            sr = processor.models[0].processor.feature_extractor.sampling_rate
            data, sr = librosa.load(file_name, sr=sr, mono=True)

            meter = pyln.Meter(sr) # create BS.1770 meter
            loudness = meter.integrated_loudness(data)
            # loudness normalize audio to -12 dB LUFS
            data = pyln.normalize.loudness(data, loudness, -12.0)

            max_samples = sr * 30
            data = data[:max_samples]

            res_line = [file_name]
            mindex = 0
            for method in range(4):
                if method == 0:
                    res_line.append(axis_al_0_score)
                    res_line.append(axis_dz_0_score)
                    current_score_al = axis_al_0_score
                    current_score_dz = axis_dz_0_score
                    res_line.append(abs(current_score_al - current_score_dz))
                elif method == 1:
                    res_line.append(axis_al_1_score)
                    res_line.append(axis_dz_1_score)
                    current_score_al = axis_al_1_score
                    current_score_dz = axis_dz_1_score
                    res_line.append(abs(current_score_al - current_score_dz))
                elif method == 2:
                    res_line.append(axis_al_2_score)
                    res_line.append(axis_dz_2_score)
                    current_score_al = axis_al_2_score
                    current_score_dz = axis_dz_2_score
                    res_line.append(abs(current_score_al - current_score_dz))
                elif method == 3:
                    res_line.append(axis_al_3_score)
                    res_line.append(axis_dz_3_score)
                    current_score_al = axis_al_3_score
                    current_score_dz = axis_dz_3_score
                    res_line.append(abs(current_score_al - current_score_dz))
                if abs(current_score_al - current_score_dz) < 2:
                    res_count[f"{method}_二人差距小于2"] += 1
                for t in ["noise", "denoise"]:
                    res = processor.models[mindex].generate(data, method, simple_model=True)
                    res_score = res["score"]
                    res_text = res["text"]
                    probs = res["probs"]
                    res_line.append(res_score)
                    dis_score_al = abs(current_score_al - res_score)
                    dis_score_dz = abs(current_score_dz - res_score)
                    dis_score_min = min(dis_score_al, dis_score_dz)
                    res_line.append(dis_score_al)
                    res_line.append(dis_score_dz)
                    res_line.append(dis_score_min)                    
                    if dis_score_min <= 1:
                        res_count[f"{method}_{t}_最小差距小于等于1"] += 1
                    if dis_score_min == 0:
                        res_count[f"{method}_{t}_最小差距等于0"] += 1
                    if dis_score_al < 2:
                        res_count[f"{method}_{t}_阿乐差距小于2"] += 1
                    if dis_score_dz < 2:
                        res_count[f"{method}_{t}_东钊差距小于2"] += 1
                    res_line.append(json.dumps(probs, ensure_ascii=False))
                    res_line.append(res_text.replace("\n", " "))
                    mindex += 1
            writer.writerow(res_line)
            # if count>4:
            #     break
    writer.writerow([
        "总数",
        "",
        "",
        "二人差距小于2",
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        "", 
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        "", 
        "",
        "",
        "二人差距小于2",
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        "",
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        "",
        "",
        "",
        "二人差距小于2",
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        "",
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        "",
        "", 
        "",
        "二人差距小于2",
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        "",
        "",
        "阿乐差距小于2",
        "东钊差距小于2",
        "最小差距等于0",
        "最小差距小于等于1",
        ""
    ])
    writer.writerow([
        count,
        "",
        "",
        res_count["0_二人差距小于2"]/count,"",
        res_count["0_noise_阿乐差距小于2"]/count,
        res_count["0_noise_东钊差距小于2"]/count,
        res_count["0_noise_最小差距等于0"]/count,
        res_count["0_noise_最小差距小于等于1"]/count,"","",
        res_count["0_denoise_阿乐差距小于2"]/count,
        res_count["0_denoise_东钊差距小于2"]/count,
        res_count["0_denoise_最小差距等于0"]/count,
        res_count["0_denoise_最小差距小于等于1"]/count,
        "",
        "",
        "",
        res_count["1_二人差距小于2"]/count,"",
        res_count["1_noise_阿乐差距小于2"]/count,
        res_count["1_noise_东钊差距小于2"]/count,
        res_count["1_noise_最小差距等于0"]/count,
        res_count["1_noise_最小差距小于等于1"]/count,"","",
        res_count["1_denoise_阿乐差距小于2"]/count,
        res_count["1_denoise_东钊差距小于2"]/count,
        res_count["1_denoise_最小差距等于0"]/count,
        res_count["1_denoise_最小差距小于等于1"]/count,
        "",
        "",
        "",
        res_count["2_二人差距小于2"]/count,"",
        res_count["2_noise_阿乐差距小于2"]/count,
        res_count["2_noise_东钊差距小于2"]/count,
        res_count["2_noise_最小差距等于0"]/count,
        res_count["2_noise_最小差距小于等于1"]/count,"","",
        res_count["2_denoise_阿乐差距小于2"]/count,
        res_count["2_denoise_东钊差距小于2"]/count,
        res_count["2_denoise_最小差距等于0"]/count,
        res_count["2_denoise_最小差距小于等于1"]/count,
        "",
        "",
        "",
        res_count["3_二人差距小于2"]/count,"",
        res_count["3_noise_阿乐差距小于2"]/count,
        res_count["3_noise_东钊差距小于2"]/count,
        res_count["3_noise_最小差距等于0"]/count,
        res_count["3_noise_最小差距小于等于1"]/count,"","",
        res_count["3_denoise_阿乐差距小于2"]/count,
        res_count["3_denoise_东钊差距小于2"]/count,
        res_count["3_denoise_最小差距等于0"]/count,
        res_count["3_denoise_最小差距小于等于1"]/count,
        ""
    ])

    out_csv.close()

    # for i in range(4):
    #     for t in ["noise", "denoise"]:
    #         data_path = f"/home/w-4090/projects/qwenaudio/data/train_ds_4/{t}/{i}/"


