#coding=utf-8

'''
requires Python 3.6 or later
pip install requests
'''
import base64
import json
import uuid
import requests

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.gen_final_text
import qwenaudio.config

host = "openspeech.bytedance.com"
api_url = f"https://{host}/api/v1/tts"


def train(appid, token, audio_path, spk_id):
    url = "https://" + host + "/api/v1/mega_tts/audio/upload"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer;" + token,
        "Resource-Id": "volc.megatts.voiceclone",
    }
    encoded_data, audio_format = encode_audio_file(audio_path)
    audios = [{"audio_bytes": encoded_data, "audio_format": audio_format}]
    data = {"appid": appid, "speaker_id": spk_id, "audios": audios, "source": 2,"language": 0, "model_type": 1}
    response = requests.post(url, json=data, headers=headers)
    print("status code = ", response.status_code)
    if response.status_code != 200:
        raise Exception("train请求错误:" + response.text)
    print("headers = ", response.headers)
    print(response.json())


def get_status(appid, token, spk_id):
    url = "https://" + host + "/api/v1/mega_tts/status"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer;" + token,
        "Resource-Id": "volc.megatts.voiceclone",
    }
    body = {"appid": appid, "speaker_id": spk_id}
    response = requests.post(url, headers=headers, json=body)
    print(response.json())


def encode_audio_file(file_path):
    with open(file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        encoded_data = str(base64.b64encode(audio_data), "utf-8")
        audio_format = os.path.splitext(file_path)[1][1:]  # 获取文件扩展名作为音频格式
        return encoded_data, audio_format

if __name__ == '__main__':
    train(qwenaudio.config.tts_appid, qwenaudio.config.tts_access_token, "/home/w-4090/projects/qwenaudio/data/zjl.MP3", "S_NcV7oCTw1")
    # try:
    #     resp = qwenaudio.gen_final_text.synthesize_speech("你好")
    #     print(f"resp body: \n{resp}")
    #     if "data" in resp:
    #         data = resp["data"]
    #         file_to_save = open("test_submit.mp3", "wb")
    #         file_to_save.write(base64.b64decode(data))
    # except Exception as e:
    #     e.with_traceback()
