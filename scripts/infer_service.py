import os
import sys
import asyncio
from aiohttp import web
import librosa
import torch

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.processor
import qwenaudio.prompts

# 初始化全局处理器（在服务启动时加载）
processor = None

async def init_app():
    """应用初始化，加载模型"""
    global processor
    print("Initializing model...")
    processor = qwenaudio.processor.ScoreProcessor("/home/w-4090/projects/qwenaudio_score/ckpts/lora-64-4-pf_score/model_epoch76",
                                                   "/home/w-4090/projects/qwenaudio_score/ckpts/generator-lora-128-64-textonly/best_model_epoch_6/lora_weights")
    print("Model initialized")
    return web.Application()

async def audio_handler(request):
    """处理音频请求"""
    try:

        # 读取音频文件
        reader = await request.multipart()
        field = await reader.next()
        assert field.name == "file"
        audio_bytes = await field.read()

        # 异步处理音频
        result = await asyncio.to_thread(process_audio, audio_bytes)
        return web.json_response(result)

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

def process_audio(audio_bytes):
    """同步处理音频（在独立线程中运行）"""
    try:
        # 保存临时音频文件
        temp_path = "/tmp/temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        # 加载音频
        data, sr = librosa.load(
            temp_path,
            sr=processor.processor.feature_extractor.sampling_rate,
            mono=True
        )

        # 截取30秒
        max_samples = processor.processor.feature_extractor.sampling_rate * 30
        data = data[:max_samples]

        result = {}
        # 生成评分
        for i in range(4):
            result[qwenaudio.prompts.prompt_mapper_reverse[i]] = processor.generate_score_refix(data, i)# 生成分数
        
        # print(result)

        return result

    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")

if __name__ == "__main__":
    # 创建应用
    app = web.Application()
    app.on_startup.append(lambda _: init_app())
    
    # 添加路由
    app.router.add_post("/score", audio_handler)
    
    # 运行服务
    web.run_app(
        app,
        host="0.0.0.0",
        port=8080,
        access_log_format='%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
    )