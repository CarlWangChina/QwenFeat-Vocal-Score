import os
import sys
import asyncio
import traceback
from aiohttp import web
import librosa
import torch
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import functools
import itertools
import logging
import pyloudnorm as pyln

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.processor
import qwenaudio.prompts
import qwenaudio.gen_final_text

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# GPU设备配置（根据实际GPU数量调整）
GPU_DEVICES = [0]
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
        logging.info(f"Initializing processor on GPU {self.gpu_id}...")
        self.processor = qwenaudio.processor.ProcessorGroup()

        text_model = "ckpts/generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights"

        logger.info(f"Loading model 1 on GPU {self.gpu_id}...")
        self.processor.add("ckpts/train_ds_4_feat_score_al/denoise/0/score/model_epoch24", text_model)# 1. 专业技巧 qwen encoder+分类器
        logger.info(f"Loading model 2 on GPU {self.gpu_id}...")
        self.processor.add("ckpts/train_ds_4_feat_score_al/denoise/1/score/best_model_epoch/13", text_model)# 2. 情感表达 qwen2audio LLM 输出层; 4维度 top2
        logger.info(f"Loading model 3 on GPU {self.gpu_id}...")
        self.processor.add("ckpts/train_ds_4_feat_score_al/denoise/2/score/best_model_epoch/25", text_model)# 3. 音色与音质 samoye encoder+f0 to RNN1; 
        logger.info(f"Loading model 4 on GPU {self.gpu_id}...")
        self.processor.add("ckpts/train_ds_4_feat_score_al/denoise/3/score/best_model_epoch/5", text_model)# 4. 气息控制 samoye encoder+f0 to RNN2;
        logger.info(f"Processor on GPU {self.gpu_id} initialized.")
        self.processor.models[0].top2_mode = False
        self.processor.models[1].top2_mode = False
        self.processor.models[2].top2_mode = False
        self.processor.models[3].top2_mode = False
    def process_audio(
            self, audio_bytes,
            get_final_text=False, 
            render_final_text=False, 
            process_steps=[0,1,2,3], 
            singer_id="0", 
            speed_ratio=1.0, 
            return_single_score=True,
            return_sum_score=True,
            return_single_text=False,
            return_prompt=False,):
        """处理音频数据"""
        logging.info(f"Processing audio on GPU {self.gpu_id} process_steps={process_steps} get_final_text={get_final_text} render_final_text={render_final_text} singer_id={singer_id}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()

                data, sr = librosa.load(
                    tmp_file.name,
                    sr=self.processor.processor.feature_extractor.sampling_rate,
                    mono=True
                )
                dtype = data.dtype
                meter = pyln.Meter(sr) # create BS.1770 meter
                loudness = meter.integrated_loudness(data)
                data = pyln.normalize.loudness(data, loudness, -12.0).astype(dtype)

                max_samples = self.processor.processor.feature_extractor.sampling_rate * 30
                # data = data[:max_samples]

                result = {}
                result_to_gen = {}
                sum_score = 0
                for i in process_steps:
                    # print(self.processor.models[i], i , self.processor.models)
                    val = self.processor.models[i].generate(data, i, simple_model=True, return_single_text=return_single_text or get_final_text)
                    result_to_gen[qwenaudio.prompts.prompt_mapper_reverse[i]] = (val["text"], val["score"])
                    if return_single_score:
                        if not return_single_text and "text" in val:
                            del val["text"]
                        if not return_prompt and "prompt" in val:
                            del val["prompt"]
                        result[qwenaudio.prompts.prompt_mapper_reverse[i]] = val
                    if return_sum_score:
                        sum_score += val["score"]
                
                if return_sum_score:
                    result["sum_score"] = sum_score

                if get_final_text:
                    result["final_text"] = qwenaudio.gen_final_text.generate_vocal_critique(result_to_gen, author_mode=singer_id in qwenaudio.gen_final_text.singerid_to_model)
                    if render_final_text:
                        result["speech"] = qwenaudio.gen_final_text.synthesize_speech(result["final_text"]["summary"], singer_id=singer_id, speed_ratio=speed_ratio)
                        result["singer_id"] = singer_id
                        result["speech_speed_ratio"] = speed_ratio
                        # print("singer_id", singer_id)
                    
                return result
        except Exception as e:
            err_str = f"Error processing audio: {str(e)}"
            traceback.print_exc()
            return {"error": err_str}

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

def process_audio_in_worker(
        audio_bytes, 
        get_final_text=False, 
        render_final_text=False, 
        process_steps=[0,1,2,3], 
        singer_id="0", 
        speed_ratio=1.0, 
        return_single_score=True, 
        return_sum_score=True,
        return_single_text=False,
        return_prompt=False):
    """在工作进程中处理音频"""
    global _worker
    return _worker.process_audio(
        audio_bytes, 
        get_final_text=get_final_text, 
        render_final_text=render_final_text, 
        process_steps=process_steps, 
        singer_id=singer_id, 
        speed_ratio=speed_ratio, 
        return_single_score=return_single_score, 
        return_sum_score=return_sum_score,
        return_single_text=return_single_text,
        return_prompt=return_prompt)

async def process_audio_bytes(
        audio_bytes, 
        get_final_text, 
        render_final_text, 
        process_steps, 
        singer_id, 
        speed_ratio, 
        return_single_score, 
        return_sum_score,
        return_single_text,
        return_prompt):
    """处理音频字节的核心逻辑"""
    global worker_round_robin
    
    # 轮询选择worker
    worker_idx = next(worker_round_robin)

    # 把process_steps拆成字符
    process_steps_id = set()
    for i in process_steps:
        if i in ["0", "1", "2", "3"]:
            process_steps_id.add(int(i))
    
    # 异步处理音频
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        functools.partial(
            process_audio_in_worker, 
            audio_bytes, 
            get_final_text, 
            render_final_text, 
            list(process_steps_id), 
            singer_id, speed_ratio, 
            return_single_score, 
            return_sum_score,
            return_single_text,
            return_prompt)
    )

async def audio_handler(request):
    """处理音频请求"""
    try:
        # 读取音频文件
        reader = await request.multipart()
        field = await reader.next()
        assert field.name == "file"
        audio_bytes = await field.read()

        # 从查询参数获取标志
        get_final_text = request.query.get("get_final_text", "true").lower() in ["true", "1", "yes"]
        render_final_text = request.query.get("render_final_text", "false").lower() in ["true", "1", "yes"]
        process_steps = request.query.get("process_steps", "0123").lower()
        singer_id = request.query.get("singer_id", "0")
        speed_ratio = float(request.query.get("speed_ratio", "1.3"))
        return_single_score = request.query.get("return_single_score", "true").lower() in ["true", "1", "yes"]
        return_sum_score = request.query.get("return_sum_score", "true").lower() in ["true", "1", "yes"]
        return_single_text = request.query.get("return_single_text", "false").lower() in ["true", "1", "yes"]
        return_prompt = request.query.get("return_prompt", "false").lower() in ["true", "1", "yes"]
        
        result = await process_audio_bytes(
            audio_bytes, 
            get_final_text, 
            render_final_text, 
            process_steps, 
            singer_id, 
            speed_ratio, 
            return_single_score, 
            return_sum_score,
            return_single_text,
            return_prompt)
        return web.json_response(result)

    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

async def local_audio_handler(request):
    """处理本地路径的音频文件（仅允许127.0.0.1访问）"""
    # 检查客户端IP
    if request.remote != "127.0.0.1":
        return web.json_response({"error": "Forbidden"}, status=403)
    
    try:
        # 读取表单数据
        data = await request.post()
        file_path = data.get("path")
        
        if not file_path:
            return web.json_response({"error": "Missing 'path' parameter"}, status=400)
        
        # 读取本地文件
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        # 从查询参数获取标志
        get_final_text = request.query.get("get_final_text", "true").lower() in ["true", "1", "yes"]
        render_final_text = request.query.get("render_final_text", "false").lower() in ["true", "1", "yes"]
        process_steps = request.query.get("process_steps", "0123").lower()
        singer_id = request.query.get("singer_id", "0")
        speed_ratio = float(request.query.get("speed_ratio", "1.3"))
        return_single_score = request.query.get("return_single_score", "true").lower() in ["true", "1", "yes"]
        return_sum_score = request.query.get("return_sum_score", "true").lower() in ["true", "1", "yes"]
        return_single_text = request.query.get("return_single_text", "false").lower() in ["true", "1", "yes"]
        return_prompt = request.query.get("return_prompt", "false").lower() in ["true", "1", "yes"]
        
        result = await process_audio_bytes(
            audio_bytes, 
            get_final_text, 
            render_final_text, 
            process_steps, 
            singer_id, 
            speed_ratio, 
            return_single_score, 
            return_sum_score,
            return_single_text,
            return_prompt)
        return web.json_response(result)

    except FileNotFoundError:
        return web.json_response({"error": "File not found"}, status=404)
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

if __name__ == "__main__":
    app = web.Application()
    app.on_startup.append(init_app)
    app.router.add_post("/score", audio_handler)
    app.router.add_post("/score_local", local_audio_handler)  # 添加新的本地端点
    
    web.run_app(
        app,
        host="0.0.0.0",
        port=8080,
        access_log_format='%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
    )