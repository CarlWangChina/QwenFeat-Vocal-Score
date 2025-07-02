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

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

# 注意：这里假设 qwenaudio.model 在您的路径中可用
from qwenaudio.model import QwenAudioScoreModel
from qwenaudio.trainer_ddp import AudioDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置最低日志级别

# 创建文件处理器（FileHandler）
file_handler = logging.FileHandler('test_loadmodel.log', mode='a', encoding='utf-8')
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

def collate_fn(batch):
    """批处理函数"""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_features = [item["input_features"] for item in batch]
    feature_attention_mask = [item["feature_attention_mask"] for item in batch]
    
    scores = [item["score"] for item in batch]
    
    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
        "input_features": torch.cat(input_features, dim=0),
        "feature_attention_mask": torch.cat(feature_attention_mask, dim=0),
        "scores": torch.stack(scores)
    }

def test_ckpt(model_path, dataset_path):
    logger.info(f"model_path={model_path} dataset_path={dataset_path}")
    processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", sample_rate=16000)
    model = QwenAudioScoreModel(
        output_num=5, 
        freeze_weight=False, 
        use_lora=False)  # 注意：这里使用了正确的模型引用
    model.load_ckpt(os.path.join(ROOT_PATH, model_path))
    dataset = AudioDataset(dataset_path, processor)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model.to("cuda")
    model.eval()
    data_count = 0
    data_check_success = 0
    data_score_distance_sum = 0
    target_value_sum = 0
    score_distance_less = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            input_features = batch["input_features"].to("cuda")
            feature_attention_mask = batch["feature_attention_mask"].to("cuda")
            scores = batch["scores"].to("cuda")

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )
            outputs = torch.softmax(outputs[0], dim=0)
            out_id = outputs.argmax().item()
            target_id = scores.item()
            outputs = outputs.tolist()
            target_value = outputs[int(target_id)]

            score_distance = abs(target_id-out_id)
            logger.info(f"target_id={target_id} out_id={out_id}, score_distance={score_distance} target_value={target_value} outs={outputs}")

            if score_distance == 0:
                data_check_success += 1
            if score_distance < 2:
                score_distance_less += 1
            data_score_distance_sum += score_distance
            target_value_sum += target_value
            data_count += 1
    
    return {
        "model_path": model_path,
        "data_count": data_count,
        "data_check_success": data_check_success/data_count,
        "data_score_distance_sum": data_score_distance_sum/data_count,
        "target_value_sum": target_value_sum/data_count,
        "score_distance_less": score_distance_less/data_count
    }

def test_model(model_path, dataset_path):
    logger.info(f"walking {model_path} dataset_path={dataset_path}")
    model_res = []
    for root, dirs, files in os.walk(model_path):
        for dir in dirs:
            pt_file = os.path.join(root, dir, "head.pt")
            if os.path.exists(pt_file):
                model_res.append(test_ckpt(os.path.join(root, dir), dataset_path))

if __name__ == "__main__":
    val_jsonl=os.path.join(ROOT_PATH, "data", "dataset", "score1", "val.jsonl")
    with open("test_loadmodel.json", "w") as f:
        json.dump(
            {
                "lora-16-16": test_model("ckpts/lora-16-16", val_jsonl),
                "lora-64-4": test_model("ckpts/lora-64-4", val_jsonl),
                "lora-64-4-oldprompt": test_model("ckpts/lora-64-4-oldprompt", val_jsonl),
                "lora-64-4-prompt-score1": test_model("ckpts/lora-64-4-prompt-score1", val_jsonl),
                "outputs-only": test_model("ckpts/outputs-only", val_jsonl),
            }, f)
    