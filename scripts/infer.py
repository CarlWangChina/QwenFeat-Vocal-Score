import os
import sys
import argparse
import json
import librosa
import torch
from pathlib import Path

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

import qwenaudio.processor
import qwenaudio.prompts

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Scoring CLI Tool")
    parser.add_argument("input_file", type=Path, help="Input audio file path")
    parser.add_argument("output_file", type=Path, help="Output JSON file path")
    parser.add_argument("--score_path", type=str, help="Path to model checkpoint", default="/home/w-4090/projects/qwenaudio_score/ckpts/lora-64-4-pf_score/model_epoch76")
    parser.add_argument("--generator_path", type=str, help="Path to generator weights", default="/home/w-4090/projects/qwenaudio_score/ckpts/generator-lora-128-64-textonly/best_model_epoch_6/lora_weights")
    return parser.parse_args()

def initialize_processor(score_path, generator_path):
    """初始化音频处理器"""
    print(f"Initializing model from: {score_path}")
    processor = qwenaudio.processor.ScoreProcessor(
        score_path,
        generator_path
    )
    print("Model initialized successfully")
    return processor

def process_audio(input_path, processor):
    """处理音频文件"""
    try:
        # 加载音频文件
        data, sr = librosa.load(
            str(input_path),
            sr=processor.processor.feature_extractor.sampling_rate,
            mono=True
        )

        # 截取前30秒
        max_samples = processor.processor.feature_extractor.sampling_rate * 30
        data = data[:max_samples]

        # 生成评分
        result = {}
        for i in range(4):
            result[qwenaudio.prompts.prompt_mapper_reverse[i]] = processor.generate_score_refix(data, i)
        
        return result

    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")

def main():
    args = parse_args()
    
    # 确保输出目录存在
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # 初始化处理器
    processor = initialize_processor(args.score_path, args.generator_path)

    try:
        # 处理音频
        results = process_audio(args.input_file, processor)
        
        # 写入结果文件
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Results saved to: {args.output_file}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()