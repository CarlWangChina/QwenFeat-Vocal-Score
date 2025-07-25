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

model_type = 3

def train(appid, token, audio_path, spk_id, host = "openspeech.bytedance.com"):
    url = "https://" + host + "/api/v1/mega_tts/audio/upload"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer;" + token,
        "Resource-Id": "volc.megatts.voiceclone",
    }
    encoded_data, audio_format = encode_audio_file(audio_path)
    audios = [{"audio_bytes": encoded_data, "audio_format": audio_format}]
    data = {
        "appid": appid,
        "speaker_id": spk_id, 
        "audios": audios, 
        "source": 2,
        "language": 0, 
        "model_type": model_type
    }
    response = requests.post(url, json=data, headers=headers)
    print("status code = ", response.status_code)
    if response.status_code != 200:
        raise Exception("train请求错误:" + response.text)
    print("headers = ", response.headers)
    print(response.json())


def get_status(appid, token, spk_id, host = "openspeech.bytedance.com"):
    url = "https://" + host + "/api/v1/mega_tts/status"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer;" + token,
        "Resource-Id": "volc.megatts.voiceclone",
    }
    body = {"appid": appid, "speaker_id": spk_id}
    response = requests.post(url, headers=headers, json=body)
    s = response.json()
    print(s)
    return s["status"] if "status" in s else -1


def encode_audio_file(file_path):
    with open(file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        encoded_data = str(base64.b64encode(audio_data), "utf-8")
        audio_format = os.path.splitext(file_path)[1][1:]  # 获取文件扩展名作为音频格式
        return encoded_data, audio_format

def make_upload_list(dir="data/所有人明星说话", out_path="config/vocal_id.json"):
    # walk dir
    singer_id = 1
    result = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(".mp3"):
                print(file)
                result.append({
                    "id": str(singer_id),
                    "name": file,
                    "path": os.path.join(root, file),
                    "model_id": ""
                })
                singer_id += 1
    with open(out_path, "w") as f:
        json.dump({"vocal_id":result,"default":{}}, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # make_upload_list()
    with open("config/vocal_id.json", "r") as f:
        data = json.load(f)
        model_auth_conf = dict()
        for item in data["auth"]:
            user_id = item["id"]
            model_auth_conf[user_id] = item
        # print(model_auth_conf)
        for item in data["vocal_id"]:
            tts_uid = item["user"]
            tts_user = model_auth_conf[tts_uid]
            print(item)
            print(tts_user)
            # train(qwenaudio.config.tts_appid, qwenaudio.config.tts_access_token, item["path"], item["model_id"])
            res = get_status(tts_user["tts_appid"], tts_user["tts_access_token"], item["model_id"])
            if res==0 or "force" in item and item["force"] == True:
                train(tts_user["tts_appid"], tts_user["tts_access_token"], item["path"], item["model_id"])
                time.sleep(10)
    # train(qwenaudio.config.tts_appid, qwenaudio.config.tts_access_token, "/home/w-4090/projects/qwenaudio/data/zjl300s.MP3", "S_NcV7oCTw1")
    # time.sleep(60)
    # get_status(appid=qwenaudio.config.tts_appid, token=qwenaudio.config.tts_access_token, spk_id="S_NcV7oCTw1")
    # try:
    #     resp = qwenaudio.gen_final_text.synthesize_speech("音色饱满但层次不足，情感表达真挚却欠缺技巧修饰，气息控制尚可但尾音不稳，整体表现有潜力但基本功仍需夯实。")
    #     print(f"resp body: \n{resp}")
    #     if "data" in resp:
    #         data = resp["data"]
    #         file_to_save = open(f"test_submit_type{model_type}_300s.mp3", "wb")
    #         file_to_save.write(base64.b64decode(data))
    # except Exception as e:
    #     e.with_traceback()
