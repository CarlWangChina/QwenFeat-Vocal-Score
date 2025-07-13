#coding=utf-8

'''
requires Python 3.6 or later
pip install requests
'''
import base64
import json
import uuid
import requests

# 填写平台申请的appid, access_token以及cluster
appid = "2423455087"
access_token= "UBnRWB_TOdHd6dRM3k2S24S1b_fHZ1b0"
cluster = "volcano_icl"

voice_type = "S_NcV7oCTw1"
host = "openspeech.bytedance.com"
api_url = f"https://{host}/api/v1/tts"

header = {"Authorization": f"Bearer;{access_token}"}

request_json = {
    "app": {
        "appid": appid,
        "token": "access_token",
        "cluster": cluster
    },
    "user": {
        "uid": "388808087185088"
    },
    "audio": {
        "voice_type": voice_type,
        "encoding": "mp3",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
    },
    "request": {
        "reqid": str(uuid.uuid4()),
        "text": "字节跳动语音合成",
        "text_type": "plain",
        "operation": "query",
        "with_frontend": 1,
        "frontend_type": "unitTson"

    }
}

if __name__ == '__main__':
    try:
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        print(f"resp body: \n{resp.json()}")
        if "data" in resp.json():
            data = resp.json()["data"]
            file_to_save = open("test_submit.mp3", "wb")
            file_to_save.write(base64.b64decode(data))
    except Exception as e:
        e.with_traceback()
