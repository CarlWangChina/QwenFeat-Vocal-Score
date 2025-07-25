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
    for singer_id in range(0, 27):
        print(singer_id)
        resp = qwenaudio.gen_final_text.synthesize_speech(
            "音色饱满但层次不足，情感表达真挚却欠缺技巧修饰，气息控制尚可但尾音不稳，整体表现有潜力但基本功仍需夯实。", singer_id=str(singer_id))
        if "data" in resp:
            data = resp["data"]
            file_to_save = open(f"outputs/test_26/{singer_id}.mp3", "wb")
            file_to_save.write(base64.b64decode(data))
        else:
            print(resp)