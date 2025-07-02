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
import qwenaudio.trainer_generator

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    dataset = qwenaudio.trainer_generator.AudioDataset(
        "/home/w-4090/projects/qwenaudio/data/gen.json",
        "/home/w-4090/projects/qwenaudio/data/scores.json",
        processor
    )
    print(dataset.__getitem__(0))