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

    gen_json_path = "/home/w-4090/projects/qwenaudio/data/gen2.json"
    score_json_path = "/home/w-4090/projects/qwenaudio/data/scores.json"
    data = []
    
    with open(gen_json_path, 'r') as f:
        gen_json = json.load(f)
    
    with open(score_json_path, 'r') as f:
        score_json = json.load(f)
    
    for song_id, song_data in score_json.items():
        for audio_id, audio_body in song_data.items():
            audio_path = audio_body["audio_path"]

            if audio_id in gen_json:

                gen_text_list = gen_json[audio_id]

                for prompt, gen_text in zip(qwenaudio.prompts.prompts, gen_text_list):

                    conversation = [
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "input.wav"},
                            {"type": "text", "text": prompt},
                        ]}
                    ]

                    conversation_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    
                    # print(audio_id, audio_path, prompt, gen_text)
                    data.append({
                        "audio": audio_path,
                        "text": conversation_text,
                        "score": gen_text[0]
                    })
    
    random.shuffle(data)

    data_train = data[:int(len(data)*0.8)]
    data_test = data[int(len(data)*0.8):]

    os.makedirs("/home/w-4090/projects/qwenaudio/data/train_pf_score", exist_ok=True)

    with open(os.path.join(SCORE_OUTPUT_PATH, "/home/w-4090/projects/qwenaudio/data/train_pf_score/train.json"), 'w') as f:
        json.dump(data_train, f, ensure_ascii=False, indent=4)

    with open(os.path.join(SCORE_OUTPUT_PATH, "/home/w-4090/projects/qwenaudio/data/train_pf_score/test.json"), 'w') as f:
        json.dump(data_test, f, ensure_ascii=False, indent=4)
