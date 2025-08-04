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
    for speed in [0.5, 0.8, 1, 1.2, 1.5]:
    # for speed in [1]:
        for pitch_ratio in [1]:
        
            out_path = f"outputs/test_arg/speed{speed}_pitch{pitch_ratio}.mp3"
            if not os.path.exists(out_path):
                resp = qwenaudio.gen_final_text.synthesize_speech(
                    "音色饱满但层次不足，情感表达真挚却欠缺技巧修饰，气息控制尚可但尾音不稳，整体表现有潜力但基本功仍需夯实。", 
                    singer_id=str(0),
                    speed_ratio=speed,
                    pitch_ratio=pitch_ratio)
                if "data" in resp:
                    data = resp["data"]
                    file_to_save = open(out_path, "wb")
                    file_to_save.write(base64.b64decode(data))
                else:
                    print(resp)