import os
import sys
import json
import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_PATH = os.path.join(ROOT_PATH, "data", "score")
SCORE_OUTPUT_PATH = os.path.join(ROOT_PATH, "data", "dataset")
sys.path.append(os.path.join(ROOT_PATH, "src"))
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import random
import qwenaudio.prompts

random.seed(42)

qwenaudio.prompts.prompts  = [
    # 1. 专业技巧
    "请对这段音频的专业技巧进行评分并写出评语，评分和评语之间用冒号分隔。",
    # 2. 情感表达		
    "请对这段音频的情感表达进行评分并写出评语，评分和评语之间用冒号分隔。",
    # 3. 音色与音质		
    "请对这段音频的音色与音质进行评分并写出评语，评分和评语之间用冒号分隔。",
    # 4.气息控制		
    "请对这段音频的气息控制进行评分并写出评语，评分和评语之间用冒号分隔。"

]

def sync_shuffle(*arrays):
    # 生成初始索引
    indices = list(range(len(arrays[0])))
    # 打乱索引顺序
    random.shuffle(indices)
    # 按打乱后的索引重组所有数组
    return [
        [arr[i] for i in indices] 
        for arr in arrays
    ]

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    gen_json_path = "data/gen.json"
    score_json_path = "data/scores.json"
    data_gen_text = []
    data_gen_score_by_text = []
    data_gen_text_simple = []
    generated_prompt = dict()
    
    with open(gen_json_path, 'r') as f:
        gen_json = json.load(f)
    
    with open(score_json_path, 'r') as f:
        score_json = json.load(f)

    with open(os.path.join(ROOT_PATH, "data/train_gen/data_score_by_text.json")) as f:
        data_gen_score_by_text_ori = json.load(f)
        for item in data_gen_score_by_text_ori["results"]:
            row = item["generate"]["text"].split("，")
            row = "，".join(row[1:])
            method = item["gen_method"]
            songid = item["audio_path"].split("/")[-1].split(".")[0]
            # print(row, method, songid)
            if songid not in generated_prompt:
                generated_prompt[songid] = dict()
            if len(row)<50:
                generated_prompt[songid][method] = row

    prompt_set = [{"1分":set(), "2分":set(), "3分":set(), "4分":set(), "5分":set()} for _ in range(4)]
    for song_id, song_data in score_json.items():
        for audio_id, audio_body in song_data.items():
            if audio_id in gen_json:
                gen_text_list = gen_json[audio_id]
                for i,gen_text in enumerate(gen_text_list):
                    # print(i, gen_text.split("：")[1])
                    prompt_set[i][gen_text.split("：")[0]].add(gen_text.split("：")[1])
    
    prompt_set = [{"1分":list(set(prompt_set[i]["1分"])), "2分":list(set(prompt_set[i]["2分"])), "3分":list(set(prompt_set[i]["3分"])), "4分":list(set(prompt_set[i]["4分"])), "5分":list(set(prompt_set[i]["5分"]))} for i in range(4)]
    
    # print(len(prompt_set[0]))
    
    for song_id, song_data in score_json.items():
        for audio_id, audio_body in song_data.items():
            audio_path = audio_body["audio_path"]

            if audio_id in gen_json:

                gen_text_list = gen_json[audio_id]

                i = 0

                for prompt, gen_text in zip(qwenaudio.prompts.prompt_gen_text, gen_text_list):
                    # print(i)
                    score_text = gen_text.split("：")[0].strip(",")
                    gen_text = gen_text.split("：")[1].strip(",")

                    prompt_sample = set()
                    prompt_sample.add(gen_text)
                    # 随机取50个样例
                    while len(prompt_sample) < 50:
                        for j in ["1分", "2分", "3分", "4分", "5分"]:
                            prompt_sample.add(random.choice(prompt_set[i][j]))
                    prompt_sample = list(prompt_sample)
                    random.shuffle(prompt_sample)

                    conversation = [
                        {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "input.wav"},
                            {"type": "text", "text": prompt+"\n"+"\n".join(prompt_sample)},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": gen_text}
                        ]}
                    ]
                    conversation_text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                    data_gen_text.append({
                        "audio": audio_path,
                        "text": conversation_text
                    })
                    
                    fix_gen_text = generated_prompt[audio_id][qwenaudio.prompts.prompt_mapper_reverse[i]] if qwenaudio.prompts.prompt_mapper_reverse[i] in generated_prompt[audio_id] else gen_text
                    conversation = [
                        {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "input.wav"},
                            {"type": "text", "text": qwenaudio.prompts.prompt_gen_score_by_text[i]+"\n"+fix_gen_text+"\n"+qwenaudio.prompts.prompt_gen_score_by_text_end},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": score_text}
                        ]}
                    ]
                    conversation_text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                    data_gen_score_by_text.append({
                        "audio": audio_path,
                        "text": conversation_text
                    })

                    conversation = [
                        {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "input.wav"},
                            {"type": "text", "text": qwenaudio.prompts.prompt_gen_text_simple[i]},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": gen_text}
                        ]}
                    ]
                    conversation_text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                    data_gen_text_simple.append({
                        "audio": audio_path,
                        "text": conversation_text
                    })

                    i += 1
    
    # random.shuffle(data_gen_text)
    # random.shuffle(data_gen_score_by_text)
    data_gen_text, data_gen_score_by_text, data_gen_text_simple = sync_shuffle(data_gen_text, data_gen_score_by_text, data_gen_text_simple)

    data_train = data_gen_text[:int(len(data_gen_text)*0.8)]
    data_test = data_gen_text[int(len(data_gen_text)*0.8):]
    with open(os.path.join(ROOT_PATH, "data/train_gen/train.json"), 'w') as f:
        json.dump(data_train, f, ensure_ascii=False, indent=4)
    with open(os.path.join(ROOT_PATH, "data/train_gen/test.json"), 'w') as f:
        json.dump(data_test, f, ensure_ascii=False, indent=4)
    
    data_train = data_gen_score_by_text[:int(len(data_gen_score_by_text)*0.8)]
    data_test = data_gen_score_by_text[int(len(data_gen_score_by_text)*0.8):]
    with open(os.path.join(ROOT_PATH, "data/train_gen/train_score_by_text.json"), 'w') as f:
        json.dump(data_train, f, ensure_ascii=False, indent=4)
    with open(os.path.join(ROOT_PATH, "data/train_gen/test_score_by_text.json"), 'w') as f:
        json.dump(data_test, f, ensure_ascii=False, indent=4)
    with open(os.path.join(ROOT_PATH, "data/train_gen/full_score_by_text.json"), 'w') as f:
        json.dump(data_gen_score_by_text, f, ensure_ascii=False, indent=4)

    data_train = data_gen_text_simple[:int(len(data_gen_text_simple)*0.8)]
    data_test = data_gen_text_simple[int(len(data_gen_text_simple)*0.8):]
    with open(os.path.join(ROOT_PATH, "data/train_gen/train_gen_text_simple.json"), 'w') as f:
        json.dump(data_train, f, ensure_ascii=False, indent=4)
    with open(os.path.join(ROOT_PATH, "data/train_gen/test_gen_text_simple.json"), 'w') as f:
        json.dump(data_test, f, ensure_ascii=False, indent=4)

    with open(os.path.join(ROOT_PATH, "data/train_gen/prompt_set.json"), 'w') as f:
        json.dump(prompt_set, f, ensure_ascii=False, indent=4)