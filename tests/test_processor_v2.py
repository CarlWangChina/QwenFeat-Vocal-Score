import json
import os
import sys
import torch
import librosa
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from tqdm import tqdm
import numpy as np
import torch.functional
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
import logging
import traceback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.prompts 
from qwenaudio.model import QwenAudioScoreModel
from qwenaudio.trainer_pf_score import AudioDataset
import qwenaudio.processor
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置最低日志级别

# 创建文件处理器（FileHandler）
file_handler = logging.FileHandler('outputs/test_loadmodel_pf_score2.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  # 处理器级别

# 创建格式化器（Formatter）
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 将格式化器绑定到处理器
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("start")


if __name__ == "__main__":

    # 初始化模型和处理器
    processor = qwenaudio.processor.ScoreProcessorV2(
        "/home/w-4090/projects/qwenaudio/ckpts/generator-lora-32-16-scoreonly-f16/best_model_epoch_13/lora_weights",
        "/home/w-4090/projects/qwenaudio/ckpts/generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights")
        # "/home/w-4090/projects/qwenaudio/ckpts/generator-lora-32-16-textonly-v2-int4/best_model_epoch_14/lora_weights")
    
    data_count = 0
    score_data_check_success = 0
    score_data_score_distance_sum = 0
    score_score_distance_less = 0
    results = []

    with open("/home/w-4090/projects/qwenaudio/data/train_gen/test_score_by_text.json") as fp:
        target_comments = json.load(fp)
        for item in target_comments:
            try:
                step_start = time.time()

                audio_path = item['audio']
                method = item['text'].split("请对他/她的")[1].split("进行打分")[0]
                target_score = item['text'].split("<|im_end|>\n<|im_start|>assistant\n")[1].split("分<|im_end|>")[0]
                print(audio_path, method)
                

                # 打分
                data, sr = librosa.load(audio_path, sr=processor.processor.feature_extractor.sampling_rate, mono=True)
                data = data[:processor.processor.feature_extractor.sampling_rate * 30] # 截取30秒
                print(data.shape)

                gen_text = processor.generate_text(data, method, simple_model=True)
                score = processor.generate_score(data, method, gen_text['text'])
                print(score)
                score["score_dist"] = abs(score["score"]-int(target_score))
                step_end = time.time()

                result_body = {
                    "generate":score,
                    "gen_method": method,
                    "audio_path":audio_path,
                    "target_score":int(target_score),
                    "target_comment":item['text'],
                    "step_time": step_end - step_start,
                }
                results.append(result_body)

                if score["score_dist"] == 0:
                    score_data_check_success += 1
                if score["score_dist"] < 2:
                    score_score_distance_less += 1
                score_data_score_distance_sum += score["score_dist"]
                data_count += 1

            except Exception as e:
                print(e)
                traceback.print_exc()
            # break
        
        with open("test_score_by_text.json", "w") as fp:
            json.dump({
                "results":results,
                "data_count": data_count,
                "acc":{
                    "predict_from_score_model_without_comment":{
                        "data_check_success": score_data_check_success/data_count,
                        "data_score_distance_sum": score_data_score_distance_sum/data_count,
                        "score_distance_less": score_score_distance_less/data_count
                    }
                }
            }, fp, indent=4, ensure_ascii=False)
