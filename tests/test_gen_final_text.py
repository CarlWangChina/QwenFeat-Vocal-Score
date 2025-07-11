import json
import os
import sys
import librosa

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.processor
import qwenaudio.gen_final_text
import qwenaudio.config

if __name__ == "__main__":
    input_path = "/home/w-4090/cutted_score_audio_separated/446892/344523004.wav"
    processor = qwenaudio.processor.create_processor(
            "ckpts/generator-lora-32-16-scoreonly-f16/best_model_epoch_13/lora_weights",
            "ckpts/generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights"
        )
    
    data, sr = librosa.load(
            str(input_path),
            sr=processor.processor.feature_extractor.sampling_rate,
            mono=True
        )

    # 截取前30秒
    max_samples = processor.processor.feature_extractor.sampling_rate * 30
    data = data[:max_samples]

    result = {}
    for i in range(4):
        val = processor.generate(data, i, simple_model=True)
        result[qwenaudio.prompts.prompt_mapper_reverse[i]] = (val["text"], val["score"])
    
    print(qwenaudio.gen_final_text.generate_vocal_critique(result, qwenaudio.config.api_key))
    