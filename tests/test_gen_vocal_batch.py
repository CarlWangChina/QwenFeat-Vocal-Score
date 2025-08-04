import base64
import json
import uuid
import requests
import time

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.gen_final_text
import qwenaudio.config
if __name__=="__main__":
    with open("config/vocal_id.json", "r") as f:
        data = json.load(f)
        id_to_singername = dict()
        for item in data["vocal_id"]:
            audio_path = item["path"]
            singer_id = item["id"]
            audio_name = os.path.basename(audio_path).split(".")[0]
            id_to_singername[singer_id] = audio_name
    for singer_id in range(0, 27):
        print(singer_id)
        if str(singer_id) in id_to_singername:
            out_path = f"outputs/test_26/{singer_id}_{id_to_singername[str(singer_id)]}.mp3"
        else:
            out_path = f"outputs/test_26/{singer_id}_unknow.mp3"
        if not os.path.exists(out_path):
            resp = qwenaudio.gen_final_text.synthesize_speech(
                "音色饱满但层次不足，情感表达真挚却欠缺技巧修饰，气息控制尚可但尾音不稳，整体表现有潜力但基本功仍需夯实。", 
                singer_id=str(singer_id),
                speed_ratio=1.0)
            if "data" in resp:
                data = resp["data"]
                file_to_save = open(out_path, "wb")
                file_to_save.write(base64.b64decode(data))
            else:
                print(resp)