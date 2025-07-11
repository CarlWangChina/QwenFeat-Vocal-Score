import os
import sys
import asyncio
from aiohttp import web
import librosa
import torch
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import functools
import itertools

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.processor
import qwenaudio.prompts
import qwenaudio.gen_final_text

# GPU设备配置（根据实际GPU数量调整）
GPU_DEVICES = [7]
NUM_WORKERS = len(GPU_DEVICES)

# 全局进程池
executor = None
worker_round_robin = None

class ProcessorWorker:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        # torch.cuda.set_device(self.gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu_id}"
        self.processor = None
        self._init_processor()

    def _init_processor(self):
        """在指定GPU上初始化处理器"""
        print(f"Initializing processor on GPU {self.gpu_id}...")
        self.processor = qwenaudio.processor.create_processor(
            "ckpts/generator-lora-32-16-scoreonly-f16/best_model_epoch_13/lora_weights",
            "ckpts/generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights"
        )
        print(f"Processor on GPU {self.gpu_id} initialized.")

    def process_audio(self, audio_bytes, get_final_text=False, render_final_text=False):
        """处理音频数据"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()

            data, sr = librosa.load(
                tmp_file.name,
                sr=self.processor.processor.feature_extractor.sampling_rate,
                mono=True
            )

            max_samples = self.processor.processor.feature_extractor.sampling_rate * 30
            data = data[:max_samples]

            result = {}
            result_to_gen = {}
            for i in range(4):
                val = self.processor.generate(data, i, simple_model=True)
                result_to_gen[qwenaudio.prompts.prompt_mapper_reverse[i]] = (val["text"], val["score"])
                result[qwenaudio.prompts.prompt_mapper_reverse[i]] = val
            
            if get_final_text:
                result["final_text"] = qwenaudio.gen_final_text.generate_vocal_critique(result_to_gen)
                
            return result

async def init_app(app):
    """应用初始化"""
    global executor, worker_round_robin
    
    print("Starting server with multiple GPU workers...")
    
    # 配置GPU设备
    multiprocessing.set_start_method("spawn", force=True)
    
    # 创建进程池并初始化worker
    print(f"Creating process pool with {NUM_WORKERS} workers...")
    executor = ProcessPoolExecutor(max_workers=NUM_WORKERS)
    
    # 初始化所有worker
    init_futures = []
    for i, gpu_id in enumerate(GPU_DEVICES):
        init_futures.append(
            asyncio.get_event_loop().run_in_executor(
                executor, 
                functools.partial(init_worker_process, gpu_id)
            )
        )
    
    # 等待所有worker初始化完成
    await asyncio.gather(*init_futures)
    print("All workers initialized successfully")
    
    # 创建worker轮询器
    worker_round_robin = itertools.cycle(range(NUM_WORKERS))

def init_worker_process(gpu_id):
    """初始化工作进程（在子进程中执行）"""
    # 每个进程有自己的全局worker
    global _worker
    _worker = ProcessorWorker(gpu_id)

def process_audio_in_worker(audio_bytes, get_final_text=False, render_final_text=False):
    """在工作进程中处理音频"""
    global _worker
    return _worker.process_audio(audio_bytes, get_final_text=get_final_text, render_final_text=render_final_text)

async def audio_handler(request):
    """处理音频请求"""
    global worker_round_robin
    
    try:
        # 读取音频文件
        reader = await request.multipart()
        field = await reader.next()
        assert field.name == "file"
        audio_bytes = await field.read()

        # 从查询参数获取get_final_text标志
        get_final_text = request.query.get("get_final_text", "false").lower() in ["true", "1", "yes"]
        render_final_text = request.query.get("render_final_text", "false").lower() in ["true", "1", "yes"]
        
        # 轮询选择worker
        worker_idx = next(worker_round_robin)
        
        # 异步处理音频
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            functools.partial(process_audio_in_worker, audio_bytes, get_final_text, render_final_text)
        )
        return web.json_response(result)

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

if __name__ == "__main__":
    app = web.Application()
    app.on_startup.append(init_app)
    app.router.add_post("/score", audio_handler)
    
    web.run_app(
        app,
        host="0.0.0.0",
        port=8080,
        access_log_format='%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
    )