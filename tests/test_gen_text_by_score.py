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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.prompts 
from qwenaudio.model import QwenAudioScoreModel
from qwenaudio.trainer_pf_score import AudioDataset


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

with open("/home/w-4090/projects/qwenaudio/data/gen.json") as fp:
    target_comments = json.load(fp)

def collate_fn(batch):
    """批处理函数"""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_features = [item["input_features"] for item in batch]
    feature_attention_mask = [item["feature_attention_mask"] for item in batch]

    text_prompt = []
    text_prompt_ori = []
    audio = []
    audio_path = []
    for item in batch:
        text_prompt_ori.append(item["text_prompt"])
        text_prompt.append(item["text_prompt"].split("<|audio_eos|>")[1].split("。")[0].strip()+"。分数在0到5之间，评语100字左右")
        audio.append(item["audio"])
        audio_path.append(item["audio_path"])
    
    scores = [item["score"] for item in batch]
    
    return {
        "audio": audio,
        "audio_path": audio_path,
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
        "input_features": torch.cat(input_features, dim=0),
        "feature_attention_mask": torch.cat(feature_attention_mask, dim=0),
        "scores": torch.stack(scores),
        "text_prompt": text_prompt,
        "text_prompt_ori": text_prompt_ori
    }
def extract_first_digit_loop(text):
    for char in text:
        if char.isdigit():
            return char
    return None

def decode_generated(text, target_score):
    print(text)
    # score = text.replace("，",",").split(",")[0].strip("分")[-1]
    score = extract_first_digit_loop(text)
    score_dist = abs(target_score-int(score))
    return {
        "text": text,
        "score": score,
        "score_dist": score_dist
    }

def test_ckpt(model_path, dataset_path):
    logger.info(f"model_path={model_path} dataset_path={dataset_path}")
    processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", sample_rate=16000)
    model = QwenAudioScoreModel(
        output_num=5, 
        freeze_weight=False, 
        use_lora=False)  # 注意：这里使用了正确的模型引用
    ori_model = model.model
    model.load_ckpt(os.path.join(ROOT_PATH, model_path))
    # 加载评语模型
    # lora_config = LoraConfig.from_pretrained("/home/w-4090/projects/qwenaudio/ckpts/generator-lora-128-64-textonly/best_model_epoch_13/lora_weights")
    # text_gen = get_peft_model(ori_model, lora_config)
    text_gen = PeftModel.from_pretrained(ori_model, "/home/w-4090/projects/qwenaudio/ckpts/generator-lora-128-64-textonly/best_model_epoch_6/lora_weights")

    dataset = AudioDataset(dataset_path, processor, max_length=30)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model.to("cuda")
    model.eval()
    data_count = 0
    genmodel_data_check_success = 0
    genmodel_data_score_distance_sum = 0
    genmodel_score_distance_less = 0
    refix_data_check_success = 0
    refix_data_score_distance_sum = 0
    refix_score_distance_less = 0
    score_data_check_success = 0
    score_data_score_distance_sum = 0
    score_score_distance_less = 0
    results = []
    with torch.no_grad():
        for batch in loader:
            try:
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                input_features = batch["input_features"].to("cuda")
                feature_attention_mask = batch["feature_attention_mask"].to("cuda")
                scores = batch["scores"].to("cuda")
                text_prompt = batch["text_prompt"][0]
                text_prompt_ori = batch["text_prompt_ori"][0]
                audio = batch["audio"]
                audio_path = batch["audio_path"][0]
                songid = audio_path.split("/")[-1].split(".")[0]
                gen_method = text_prompt.split("请对这段音频的")[1].split("进行评分并写出评语")[0]

                conversation = [
                    {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": "input.wav"},
                        {"type": "text", 
                        "text": f"这是一个歌手的演唱，请对他/她的{gen_method}进行打分，数值介于0到5之间，1为最差，5为最好。评分标准如下：\n"+ qwenaudio.prompts.prompt_refix[gen_method]+"+\n打分完成后再编写一段100字左右的评语，分数与评语之间用逗号分隔。"},
                    ]}
                ]

                conversation_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                )
                outputs = torch.softmax(outputs[0], dim=0)
                out_id = outputs.argmax().item()
                target_id = scores.item()
                target_score = target_id+1
                outputs = outputs.tolist()
                target_value = outputs[int(target_id)]

                score_distance = abs(target_id-out_id)
                logger.info(f"target_id={target_id} out_id={out_id}, score_distance={score_distance} target_value={target_value} outs={outputs}")
                
                #测试直接生成
                inputs = processor(text=conversation_text, audios=audio, return_tensors="pt", padding=True).to("cuda")
                generate_ids = text_gen.generate(**inputs, max_length=1024)
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                text_by_genmodel = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                res_genmodel = decode_generated(text_by_genmodel,target_score)
                res_genmodel["prompt"] = conversation_text

                #测试加上分数再生成
                conversation_text += str(out_id+1)+"分，"
                inputs = processor(text=conversation_text, audios=audio, return_tensors="pt", padding=True).to("cuda")
                generate_ids = text_gen.generate(**inputs, max_length=1024)
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                text_by_genmodel_with_score = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                #测试重新修复分数
                conversation_refix_text = processor.apply_chat_template([
                        {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "input.wav"},
                            {"type": "text", "text": f"这是一段音频，对它的{gen_method}评分如下：\n{text_by_genmodel_with_score}\n请根据这条内容给出一个评分数值，数值介于0到5之间，1为最差，5为最好。评分标准如下：\n"+ qwenaudio.prompts.prompt_refix[gen_method]},
                        ]}
                    ], add_generation_prompt=True, tokenize=False)
                inputs = processor(text=conversation_refix_text, audios=audio, return_tensors="pt", padding=True).to("cuda")
                generate_ids = text_gen.generate(**inputs, max_length=1024)
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                text_by_genmodel_refix = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                res_refix = decode_generated(text_by_genmodel_refix.strip("分")+"分，"+text_by_genmodel_with_score, target_score)
                res_refix["prompt"] = conversation_refix_text

                result_body = {
                    "generate":{
                        "text_by_genmodel": res_genmodel,
                        "generate_comment": res_refix,
                        "predict_from_score_model_without_comment":{
                            "score":out_id+1,
                            "prompt":text_prompt_ori,
                            "score_dist":score_distance,
                        }
                    },
                    "gen_method": gen_method,
                    "audio_path":audio_path,
                    "target_score":int(target_score),
                    "target_comment":target_comments[songid][qwenaudio.prompts.prompt_mapper[gen_method]],
                    "songid":songid
                }
                results.append(result_body)

                if score_distance == 0:
                    score_data_check_success += 1
                if score_distance < 2:
                    score_score_distance_less += 1
                score_data_score_distance_sum += score_distance
                
                if res_refix["score_dist"] == 0:
                    refix_data_check_success += 1
                if res_refix["score_dist"] < 2:
                    refix_score_distance_less += 1
                refix_data_score_distance_sum += res_refix["score_dist"]

                if res_genmodel["score_dist"] == 0:
                    genmodel_data_check_success += 1
                if res_genmodel["score_dist"] < 2:
                    genmodel_score_distance_less += 1
                genmodel_data_score_distance_sum += res_genmodel["score_dist"]


                data_count += 1
                # break
            except Exception as e:
                print(e)

    
    return {
        "results":results,
        "model_path": model_path,
        "data_count": data_count,
        "acc":{
            "predict_from_score_model_without_comment":{
                "data_check_success": score_data_check_success/data_count,
                "data_score_distance_sum": score_data_score_distance_sum/data_count,
                "score_distance_less": score_score_distance_less/data_count
            },
            "generate_comment":{
                "data_check_success": refix_data_check_success/data_count,
                "data_score_distance_sum": refix_data_score_distance_sum/data_count,
                "score_distance_less": refix_score_distance_less/data_count
            },
            "text_by_genmodel":{
                "data_check_success": genmodel_data_check_success/data_count,
                "data_score_distance_sum": genmodel_data_score_distance_sum/data_count,
                "score_distance_less": genmodel_score_distance_less/data_count
            }
        }
    }

if __name__ == "__main__":
    val_jsonl="/home/w-4090/projects/qwenaudio/data/train_pf_score/test.json"
    with open("outputs/test_loadmodel_pfsc_2.json", "w") as f:
        json.dump(
            {
                # "epoch_96": test_ckpt("/home/w-4090/projects/qwenaudio/ckpts/lora-64-4-pf_score/model_epoch96", val_jsonl),
                # "epoch_62": test_ckpt("/home/w-4090/projects/qwenaudio/ckpts/lora-64-4-pf_score/best_model_epoch/62", val_jsonl),
                # "epoch_91": test_ckpt("/home/w-4090/projects/qwenaudio/ckpts/lora-64-4-pf_score/best_model_epoch/91", val_jsonl),
                "epoch_76": test_ckpt("/home/w-4090/projects/qwenaudio/ckpts/lora-64-4-pf_score/model_epoch76", val_jsonl),
            }, f, ensure_ascii=False, indent=4)
    