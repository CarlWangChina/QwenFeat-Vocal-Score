import json
import time
import requests
from typing import Iterator, Tuple

group_id = "1953157493374853484"
api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLmtbfonrrnlKjmiLdfNDA5NzQ2NDI4MDYyNzQwNDgxIiwiVXNlck5hbWUiOiLmtbfonrrnlKjmiLdfNDA5NzQ2NDI4MDYyNzQwNDgxIiwiQWNjb3VudCI6IiIsIlN1YmplY3RJRCI6IjE5NTMxNTc0OTMzNzkwNDc3ODgiLCJQaG9uZSI6IjE5OTgyMzI0MzUxIiwiR3JvdXBJRCI6IjE5NTMxNTc0OTMzNzQ4NTM0ODQiLCJQYWdlTmFtZSI6IiIsIk1haWwiOiIiLCJDcmVhdGVUaW1lIjoiMjAyNS0wOC0xMSAxNzowMzo1MSIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.JYYZIa108nyKwlf1xK3O7ZdI9GLCrRkZDVul36vEX-qzjAJdv7msLuX8K22VPZKmsS__ptrtxTRvzQZvDjoBlkcjYzBU0cYkSURoyjLzoTE3p5NS6GsrQi9TdjH4Peq92pMTbcmRQf9cb-gQjfPPtjx5Jt4yeiLobz8qtVJjfnXJKUfp17nK4nPBDWKPP_1EnkP_YV79Da0qu8vehqhEDakZ-Vxca8h7uTGAeN0U5MFxUvRH8aMzkxT14b4P0HDpp5LYLPLRjEY7Cvd-S1fXqmC4neeH_ZJVkTXLCc1oYxoY4t_7a4O8K_DjtiT9Kqud1IpvoBL2TwFcvw_xX3nWZQ"
voice_id = "zjl_sdadgfdhfghfkmyukiknbv"

def voice_clone_create():
    #复刻音频上传
    url = f'https://api.minimaxi.com/v1/files/upload?GroupId={group_id}'
    headers1 = {
        'authority': 'api.minimaxi.com',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'purpose': 'voice_clone'
    }

    files = {
        'file': open('output.mp3', 'rb')
    }
    response = requests.post(url, headers=headers1, data=data, files=files)
    file_id = response.json().get("file").get("file_id")
    print(file_id)

    #示例音频上传
    url = f'https://api.minimaxi.com/v1/files/upload?GroupId={group_id}'
    headers1 = {
        'authority': 'api.minimaxi.com',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'purpose': 'prompt_audio'
    }

    files = {
        'file': open('prompt.mp3', 'rb')
    }
    response = requests.post(url, headers=headers1, data=data, files=files)
    prompt_file_id = response.json().get("file").get("file_id")
    print(prompt_file_id)


    #音频复刻
    url = f'https://api.minimaxi.com/v1/voice_clone?GroupId={group_id}'
    payload2 = json.dumps({
    "file_id": file_id,
    "voice_id": "test1234"
    })
    headers2 = {
    'Authorization': f'Bearer {api_key}',
    'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers2, data=payload2)
    print(response.text)

def get_voice_list():
    url = f'https://api.minimaxi.com/v1/get_voice'
    headers = {
        'authority': 'api.minimaxi.com',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'voice_type': 'all'
    }

    response = requests.post(url, headers=headers, data=data)

    print(response.text)


def call_tts_stream(group_id: str, api_key: str, text: str, emotion: str, voice_id: str, file_format: str) -> Iterator[bytes]:
    """调用TTS流式API并返回音频数据迭代器"""
    url = f"https://api.minimaxi.com/v1/t2a_v2?GroupId={group_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream"
    }
    
    body = json.dumps({
        "model": "speech-2.5-hd-preview",#speech-2.5-turbo-preview
        "text": text,
        "stream": True,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
            "emotion":emotion#["happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm"]
        },
        "pronunciation_dict": {
            "tone": [
            ]
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": file_format,
            "channel": 1
        }
    })
    print(body)
    
    response = requests.post(url, headers=headers, data=body, stream=True)
    response.raise_for_status()
    
    # 逐行处理事件流
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                try:
                    event_data = json.loads(decoded_line[5:].strip())
                    if "data" in event_data and "audio" in event_data["data"]:
                        audio_hex = event_data["data"]["audio"]
                        yield bytes.fromhex(audio_hex)
                except json.JSONDecodeError:
                    print(f"JSON解析错误: {decoded_line}")
                except ValueError:
                    print(f"十六进制解码错误: {decoded_line}")

def save_audio_file(audio_stream: Iterator[bytes], out_path: str) -> Tuple[str, int]:
    """将音频流保存到文件并返回文件名和文件大小"""
    timestamp = int(time.time())
    total_bytes = 0
    
    with open(out_path, 'wb') as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
    
    return total_bytes

if __name__ == '__main__':
    # 配置参数（实际使用中应从安全来源获取）
    text = "音色饱满但层次不足，情感表达真挚却欠缺技巧修饰，气息控制尚可但尾音不稳，整体表现有潜力但基本功仍需夯实。"
    file_format = 'mp3'  # 支持 mp3/pcm/flac

    for emotion in ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm"]:
        # 调用TTS API并保存结果
        try:
            print("开始合成语音...")
            audio_stream = call_tts_stream(
                group_id=group_id, 
                api_key=api_key, 
                text=text, 
                file_format=file_format,
                emotion=emotion,
                voice_id=voice_id)
            file_size = save_audio_file(audio_stream, f"output/output_{emotion}.{file_format}")
            print(f"文件大小: {file_size} 字节")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
