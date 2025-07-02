import os
import sys
import json
import torch
import time
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_PATH = os.path.join(ROOT_PATH, "data", "score")
SCORE_OUTPUT_PATH = os.path.join(ROOT_PATH, "data", "dataset")
sys.path.append(os.path.join(ROOT_PATH, "src"))
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, LoftQConfig

import qwenaudio.prompts

if __name__=="__main__":
    with open("data/train_gen/test.json") as f:
        data_json = json.load(f)

    # 加载模型
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model.half()
    model.to("cuda")

    lora_config = LoraConfig.from_pretrained("/home/w-4090/projects/qwenaudio/ckpts/generator-lora-128-64/best_model_epoch_14/lora_weights")
    # 重建PEFT模型
    model = get_peft_model(model, lora_config)

    res = []
    #开始测试
    with open("test_model_gen.json", "w") as f:
        for item in data_json:
            data_text = item["text"].split("<|audio_eos|>")[1].split("<|im_end|>")[0]
            data_answer_ori = item["text"].split("<|im_start|>assistant")[1].split("<|im_end|>")[0]
            print(data_text, data_answer_ori)


            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "input.wav"},
                    {"type": "text", "text": data_text},
                ]},
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = [librosa.load(item["audio"], sr=processor.feature_extractor.sampling_rate)[0][:16000*30]]

            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to("cuda")
            # inputs.input_ids = inputs.input_ids

            # 统计耗时
            start_time = time.time()

            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_length=1024)
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(response)
            time_dur = time.time() - start_time
            print("耗时：", time_dur)

            res = {"audio":item["audio"], "time_taken":time_dur, "prompt":data_text, "ori_output":data_answer_ori, "generate_output":response}

            # break

            f.write(json.dumps(res, ensure_ascii=False, indent=4)+"\n========\n")
            f.flush()