import os
import sys
import json
import random
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_PATH = os.path.join(ROOT_PATH, "data", "score")
SCORE_OUTPUT_PATH = os.path.join(ROOT_PATH, "data", "dataset")
sys.path.append(os.path.join(ROOT_PATH, "src"))
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import qwenaudio.prompts

random.seed(42)


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    gen_json_path = "/home/w-4090/projects/qwenaudio/data/gen_ldz.json"
    score_json_path = "/home/w-4090/projects/qwenaudio/data/scores.json"
    data_text = [[] for _ in range(4)]
    data_text_noise = [[] for _ in range(4)]
    data_score = [[] for _ in range(4)]
    data_score_noise = [[] for _ in range(4)]
    
    with open(gen_json_path, 'r') as f:
        gen_json = json.load(f)
    
    with open(score_json_path, 'r') as f:
        score_json = json.load(f)
    
    for song_id, song_data in score_json.items():
        for audio_id, audio_body in song_data.items():
            audio_path = audio_body["audio_path"]

            if audio_id in gen_json:

                gen_text_list = gen_json[audio_id]

                axis = 0
                for prompt, gen_text in zip(qwenaudio.prompts.prompt_gen_score_by_text, gen_text_list):

                    conversation = [
                        {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "input.wav"},
                            {"type": "text", "text": qwenaudio.prompts.prompt_gen_text_simple[axis]},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": gen_text[1]}
                        ]}
                    ]

                    conversation_text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                    
                    print(audio_id, audio_path, gen_text)
                    data_text[axis].append({
                        "audio": audio_path,
                        "text": conversation_text,
                        "score": gen_text[0]
                    })
                    data_text_noise[axis].append({
                        "audio": audio_path.replace("cutted_score_audio", "cutted_score_audio_separated"),
                        "text": conversation_text,
                        "score": gen_text[0]
                    })

                    conversation = [
                        {'role': 'system', 'content': 'You are a critic in the field of vocal performance.'},
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "input.wav"},
                            {"type": "text", "text": qwenaudio.prompts.prompt_gen_score_by_text[axis]+"\n"+gen_text[1]+"\n"+qwenaudio.prompts.prompt_gen_score_by_text_end},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": str(gen_text[0])+"分"}
                        ]}
                    ]
                    conversation_text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                    data_score[axis].append({
                        "audio": audio_path,
                        "text": conversation_text,
                        "score": gen_text[0]
                    })
                    data_score_noise[axis].append({
                        "audio": audio_path.replace("cutted_score_audio", "cutted_score_audio_separated"),
                        "text": conversation_text,
                        "score": gen_text[0]
                    })

                    axis += 1

    base_dir = "/home/w-4090/projects/qwenaudio/data/train_ds_4_ldz"

    os.makedirs(f"{base_dir}", exist_ok=True)

    for i, d_n in enumerate(zip(data_text, data_text_noise, data_score, data_score_noise)):
        d = d_n[0] 
        data_train = d[:int(len(d)*0.8)]
        data_test = d[int(len(d)*0.8):]
        os.makedirs(f"{base_dir}/noise/{i}", exist_ok=True)
        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/noise/{i}/train_text.json"), 'w') as f:
            json.dump(data_train, f, ensure_ascii=False, indent=4)

        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/noise/{i}/test_text.json"), 'w') as f:
            json.dump(data_test, f, ensure_ascii=False, indent=4)
        
        d = d_n[1]
        data_train = d[:int(len(d)*0.8)]
        data_test = d[int(len(d)*0.8):]
        os.makedirs(f"{base_dir}/denoise/{i}", exist_ok=True)
        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/denoise/{i}/train_text.json"), 'w') as f:
            json.dump(data_train, f, ensure_ascii=False, indent=4)

        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/denoise/{i}/test_text.json"), 'w') as f:
            json.dump(data_test, f, ensure_ascii=False, indent=4)
        
        d = d_n[2]
        data_train = d[:int(len(d)*0.8)]
        data_test = d[int(len(d)*0.8):]
        os.makedirs(f"{base_dir}/noise/{i}", exist_ok=True)
        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/noise/{i}/train_score.json"), 'w') as f:
            json.dump(data_train, f, ensure_ascii=False, indent=4)

        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/noise/{i}/test_score.json"), 'w') as f:
            json.dump(data_test, f, ensure_ascii=False, indent=4)

        d = d_n[3]
        data_train = d[:int(len(d)*0.8)]
        data_test = d[int(len(d)*0.8):]
        os.makedirs(f"{base_dir}/denoise/{i}", exist_ok=True)
        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/denoise/{i}/train_score.json"), 'w') as f:
            json.dump(data_train, f, ensure_ascii=False, indent=4)

        with open(os.path.join(SCORE_OUTPUT_PATH, f"{base_dir}/denoise/{i}/test_score.json"), 'w') as f:
            json.dump(data_test, f, ensure_ascii=False, indent=4)

