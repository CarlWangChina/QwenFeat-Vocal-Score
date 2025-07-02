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

import qwenaudio.prompts 
from qwenaudio.model import QwenAudioScoreModel
from qwenaudio.trainer_pf_score import AudioDataset

def extract_first_digit_loop(text):
    for char in text:
        if char.isdigit():
            return char
    return None

def decode_generated(text):
    # print(text)
    # score = text.replace("，",",").split(",")[0].strip("分")[-1]
    score = extract_first_digit_loop(text)
    return {
        "text": text,
        "score": score,
    }

class ScoreProcessor:
    def __init__(self, score_model_path, text_model_path, processor_name="Qwen/Qwen2-Audio-7B-Instruct"):
        self.processor = Qwen2AudioProcessor.from_pretrained(processor_name, sample_rate=16000)
        self.model = QwenAudioScoreModel(
            output_num=5, 
            freeze_weight=False, 
            use_lora=False)  # 注意：这里使用了正确的模型引用
        self.ori_model = self.model.model
        self.model.load_ckpt(score_model_path)
        self.text_gen = PeftModel.from_pretrained(self.ori_model, text_model_path)
        self.model.to("cuda")
        self.model.eval()
        self.text_gen.to("cuda")
        self.text_gen.eval()
    
    def generate_text(self, audio, gen_method):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        gen_method_text = qwenaudio.prompts.prompt_mapper_reverse[gen_method]
        conversation = [
            {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "input.wav"},
                {"type": "text", 
                "text": f"这是一个歌手的演唱，请对他/她的{gen_method_text}进行打分，数值介于0到5之间，1为最差，5为最好。评分标准如下：\n"+ qwenaudio.prompts.prompt_refix[gen_method_text]+"+\n打分完成后再编写一段100字左右的评语，分数与评语之间用逗号分隔。"},
            ]}
        ]

        conversation_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=conversation_text, audios=audio, return_tensors="pt", padding=True).to("cuda")
        generate_ids = self.text_gen.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text_by_genmodel = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res_genmodel = decode_generated(text_by_genmodel)
        res_genmodel["prompt"] = conversation_text

        return res_genmodel

    def generate_score(self, audio, gen_method):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        
        conversation = [
            # {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "input.wav"},
                {"type": "text", 
                "text": qwenaudio.prompts.prompts[gen_method]},
            ]}
        ]
        
        conversation_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=conversation_text, audios=audio, return_tensors="pt", padding=True).to("cuda")

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            input_features=inputs.input_features,
            feature_attention_mask=inputs.feature_attention_mask,
        )
        out_id = outputs[0].argmax().item()

        return {"score":out_id+1, "prompt": conversation_text}
    
    def generate_text_with_score(self, audio, gen_method, score):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        gen_method_text = qwenaudio.prompts.prompt_mapper_reverse[gen_method]
        conversation = [
            {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "input.wav"},
                {"type": "text", 
                "text": f"这是一个歌手的演唱，请对他/她的{gen_method_text}进行打分，数值介于0到5之间，1为最差，5为最好。评分标准如下：\n"+ qwenaudio.prompts.prompt_refix[gen_method_text]+"+\n打分完成后再编写一段100字左右的评语，分数与评语之间用逗号分隔。"},
            ]}
        ]

        conversation_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        conversation_text += str(score)+"分，"
        inputs = self.processor(text=conversation_text, audios=audio, return_tensors="pt", padding=True).to("cuda")
        generate_ids = self.text_gen.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text_by_genmodel = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res_genmodel = decode_generated(text_by_genmodel)
        res_genmodel["prompt"] = conversation_text

        return res_genmodel

    def generate_score_with_text(self, audio, gen_method, text_by_genmodel_with_score):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        gen_method_text = qwenaudio.prompts.prompt_mapper_reverse[gen_method]
        conversation_refix_text = self.processor.apply_chat_template([
                {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "input.wav"},
                    {"type": "text", "text": f"这是一个歌手的演唱，对它的{gen_method}评分如下：\n{text_by_genmodel_with_score}\n请根据这条内容给出一个评分数值，数值介于0到5之间，1为最差，5为最好。评分标准如下：\n"+ qwenaudio.prompts.prompt_refix[gen_method_text]},
                ]}
            ], add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=conversation_refix_text, audios=audio, return_tensors="pt", padding=True).to("cuda")
        generate_ids = self.text_gen.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text_by_genmodel_refix = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res_refix = decode_generated(text_by_genmodel_refix.strip("分")+"分，"+text_by_genmodel_with_score)
        res_refix["prompt"] = conversation_refix_text
        return res_refix