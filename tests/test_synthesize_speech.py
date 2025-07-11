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
    qwenaudio.gen_final_text.synthesize_speech("语音", input_path)