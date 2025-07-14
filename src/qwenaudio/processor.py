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
import random
from transformers import StoppingCriteria

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
        "score": int(score) if score is not None else 3,
        "score_ori": score
    }

class CustomStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0])
        return "\n" in decoded_text
    
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
        inputs = self.processor(text=conversation_text, audio=audio, return_tensors="pt", padding=True).to("cuda")
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
        inputs = self.processor(text=conversation_text, audio=audio, return_tensors="pt", padding=True).to("cuda")

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
        inputs = self.processor(text=conversation_text, audio=audio, return_tensors="pt", padding=True).to("cuda")
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
        inputs = self.processor(text=conversation_refix_text, audio=audio, return_tensors="pt", padding=True).to("cuda")
        generate_ids = self.text_gen.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text_by_genmodel_refix = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res_refix = decode_generated(text_by_genmodel_refix.strip("分")+"分，"+text_by_genmodel_with_score)
        res_refix["prompt"] = conversation_refix_text
        return res_refix
    
    def generate_score_refix(self, audio, gen_method):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        score_1 = self.generate_score(audio, gen_method)
        text_1 = self.generate_text_with_score(audio, gen_method, score_1["score"])
        score = self.generate_score_with_text(audio, gen_method, text_1["text"])
        text_1["score"] = score["score"]
        text_1["text"] = str(score["score"])+"分，"+text_1["text"]
        return text_1
        
class ScoreProcessorV2:
    def __init__(self, score_model_path, text_model_path, processor_name="Qwen/Qwen2-Audio-7B-Instruct"):
        self.processor = Qwen2AudioProcessor.from_pretrained(processor_name, sample_rate=16000)
        self.ori_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            # torch_dtype=torch.bfloat16,
            # load_in_4bit=True
            # load_in_8bit=True
        )
        self.ori_model.half()
        self.text_gen = PeftModel.from_pretrained(self.ori_model, text_model_path)
        self.score_gen = PeftModel.from_pretrained(self.ori_model, score_model_path)
        self.ori_model.to("cuda")
        self.ori_model.eval()
        self.text_gen.to("cuda")
        self.text_gen.eval()
        self.score_gen.to("cuda")
        self.score_gen.eval()
        with open("data/train_gen/prompt_set.json") as fp:
            self.prompt_sample = json.load(fp)
        
        # 定义目标字符集合
        self.target_chars = ['1', '2', '3', '4', '5']

        # 获取这些字符对应的token ID
        target_token_ids = []
        for char in self.target_chars:
            # 编码单个字符
            token_id = self.processor.tokenizer.encode(char, add_special_tokens=False)
            # 确保是单个token
            if len(token_id) == 1:
                target_token_ids.append(token_id[0])
            else:
                # print(f"警告: 字符 '{char}' 被编码为多个token: {token_id}")
                print(f"Warning: Character '{char}' is encoded as multiple tokens: {token_id}")
        self.target_token_ids = target_token_ids

        print("load ScoreProcessorV2 done")

    def generate_text(self, audio, gen_method, simple_model=False):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        if simple_model:
            prompt = qwenaudio.prompts.prompt_gen_text_simple[gen_method]
        else:
            prompt = qwenaudio.prompts.prompt_gen_text[gen_method]
            prompt_sample = set()
            # 随机取50个样例
            while len(prompt_sample) < 30:
                for j in ["1分", "2分", "3分", "4分", "5分"]:
                    prompt_sample.add(random.choice(self.prompt_sample[gen_method][j]))
            prompt_sample = list(prompt_sample)
            random.shuffle(prompt_sample)
            prompt += "\n"+"\n".join(prompt_sample)
        
        conversation = [
            {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "input.wav"},
                {"type": "text", "text": prompt},
            ]}
        ]
        conversation_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=conversation_text, audio=audio, return_tensors="pt", padding=True).to("cuda")
        generate_ids = self.text_gen.generate(**inputs, max_length=4096, eos_token_id=self.processor.tokenizer.encode("\n")[0])
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text_by_genmodel = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {
            "prompt": conversation_text,
            "text": text_by_genmodel
        }

    def generate_score(self, audio, gen_method, text):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        conversation = [
            {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "input.wav"},
                {"type": "text", "text": qwenaudio.prompts.prompt_gen_score_by_text[gen_method]+"\n"+text+"\n"+qwenaudio.prompts.prompt_gen_score_by_text_end},
            ]}
        ]
        conversation_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=conversation_text, audio=audio, return_tensors="pt", padding=True).to("cuda")
        # generate_ids = self.score_gen.generate(**inputs, max_length=4096)
        # generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        # text_by_genmodel = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        with torch.no_grad():
            outputs = self.score_gen(**inputs)

            # 获取下一个token的logits
            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # 提取目标token的概率
            target_logits = next_token_logits[:, self.target_token_ids]  # [1, 6]
            target_probs = torch.nn.functional.softmax(target_logits, dim=-1)  # [1, 6]

            # 转换为概率字典
            char_probs = {}
            for i, char in enumerate(self.target_chars):
                # 获取概率值（转换为Python float）
                prob = target_probs[0, i].item()
                char_probs[char] = prob

            # 归一化概率
            total_prob = sum(char_probs.values())
            if total_prob > 0:
                for char in char_probs:
                    char_probs[char] /= total_prob

            # 现在char_probs包含每个数字字符的概率
            # print("数字概率分布:", char_probs)
            
            char_probs_max_key = max(char_probs, key=char_probs.get)
            # print("预测的分数是：", char_probs_max_key)
            score = int(char_probs_max_key)

        return {
            "prompt": conversation_text,
            "text": str(score)+"分，"+text,
            "score":score,
            "probs":char_probs
        }
    
    def generate(self, audio, gen_method, simple_model=False):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        gen_text = self.generate_text(audio, gen_method, simple_model)
        score = self.generate_score(audio, gen_method, gen_text['text'])
        return score

class ScoreProcessorV3:
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
        
        with open("data/train_gen/prompt_set.json") as fp:
            self.prompt_sample = json.load(fp)
        print("load ScoreProcessorV3 done")

    def generate_text(self, audio, gen_method, simple_model=False):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        if simple_model:
            prompt = qwenaudio.prompts.prompt_gen_text_simple[gen_method]
        else:
            prompt = qwenaudio.prompts.prompt_gen_text[gen_method]
            prompt_sample = set()
            # 随机取50个样例
            while len(prompt_sample) < 50:
                for j in ["1分", "2分", "3分", "4分", "5分"]:
                    prompt_sample.add(random.choice(self.prompt_sample[gen_method][j]))
            prompt_sample = list(prompt_sample)
            random.shuffle(prompt_sample)
            prompt += "\n"+"\n".join(prompt_sample)
        
        conversation = [
            {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "input.wav"},
                {"type": "text", "text": prompt},
            ]}
        ]
        conversation_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=conversation_text, audio=audio, return_tensors="pt", padding=True).to("cuda")
        generate_ids = self.text_gen.generate(**inputs, max_length=4096)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        text_by_genmodel = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {
            "prompt": conversation_text,
            "text": text_by_genmodel
        }

    def generate_score(self, audio, gen_method, text):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]
        conversation = [
            {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "input.wav"},
                {"type": "text", "text": qwenaudio.prompts.prompt_gen_score_by_text[gen_method]+"\n"+text+"\n"+qwenaudio.prompts.prompt_gen_score_by_text_end},
            ]}
        ]
        conversation_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=conversation_text, audio=audio, return_tensors="pt", padding=True).to("cuda")

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            input_features=inputs.input_features,
            feature_attention_mask=inputs.feature_attention_mask,
        )
        out_id = outputs[0].argmax().item()

        return {"score":out_id+1, "prompt": conversation_text}
    
    def generate(self, audio, gen_method, simple_model=False):
        if isinstance(gen_method, str):
            gen_method = qwenaudio.prompts.prompt_mapper[gen_method]

        gen_text = self.generate_text(audio, gen_method, simple_model)
        score = self.generate_score(audio, gen_method, gen_text['text'])
        score["text"] = str(score["score"])+"分，"+gen_text['text']
        return score

def create_processor(score_model_path, text_model_path):
    print("create_processor")
    print("score_model_path", score_model_path)
    print("text_model_path", text_model_path)
    if not os.path.exists(score_model_path) or not os.path.exists(text_model_path):
        raise FileNotFoundError("model not found")
    elif os.path.exists(os.path.join(score_model_path,"head.pt")):
        return ScoreProcessorV3(score_model_path, text_model_path)
    else:
        return ScoreProcessorV2(score_model_path, text_model_path)