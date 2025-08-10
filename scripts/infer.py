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
    return parser.parse_args()

def initialize_processor():
    """初始化音频处理器"""
    print(f"Initializing model")
    processor = qwenaudio.processor.ProcessorGroup()

    text_model = "ckpts/generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights"

    processor.add("ckpts/train_ds_4_score_al/denoise/0/score/best_model_epoch/8", text_model)# 1. 专业技巧 qwen encoder+分类器
    processor.add("ckpts/train_ds_4_al/denoise/1/score/best_model_epoch_39/lora_weights", "ckpts/train_ds_4_al/denoise/1/text/best_model_epoch_39/lora_weights")# 2. 情感表达 qwen2audio LLM 输出层; 4维度 top2
    processor.add("ckpts/train_ds_4_feat_score_al/denoise/2/score/best_model_epoch/25", text_model)# 3. 音色与音质 samoye encoder+f0 to RNN1; 
    processor.add("ckpts/train_ds_4_feat_score_al/denoise/3/score/best_model_epoch/5", text_model)# 4. 气息控制 samoye encoder+f0 to RNN2;
    processor.models[0].top2_mode = False
    processor.models[1].top2_mode = True
    processor.models[2].top2_mode = False
    processor.models[3].top2_mode = False
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
            result[qwenaudio.prompts.prompt_mapper_reverse[i]] = processor.models[i].generate(data, i, simple_model=True)
        
        return result

    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")

def main():
    args = parse_args()
    
    # 确保输出目录存在
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # 初始化处理器
    processor = initialize_processor()

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