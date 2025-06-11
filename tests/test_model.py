import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
# print(sys.path)
import torch
import qwenaudio.model

if __name__ == "__main__":
    proc = qwenaudio.model.QwenAudioScore()
    res = proc.preprocess_audio_file(["data/translate_to_chinese.wav"])
    print(res.input_ids.shape)
    print(res.attention_mask.shape)
    print(res.input_features.shape)
    print(res.feature_attention_mask.shape)

    res = proc.process_audio_file(["data/translate_to_chinese.wav"])
    print(res.shape)
    res = proc.process_audio([torch.zeros(16000).numpy()])
    print(res.shape)