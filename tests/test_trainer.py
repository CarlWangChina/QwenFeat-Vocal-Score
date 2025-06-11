import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))
# print(sys.path)
import torch
import qwenaudio.model
import qwenaudio.trainer
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioProcessor


if __name__ == "__main__":
    # 初始化模型和处理器
    processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", sample_rate=16000)
    model = qwenaudio.model.QwenAudioScoreModel(output_num=5)

    print(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # 仅检查全连接层
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    print(f"模块 {name} 的参数 {param_name} 未冻结")

    
    # 创建训练器
    trainer = qwenaudio.trainer.QwenAudioTrainer(
        model=model,
        processor=processor,
        train_jsonl=os.path.join(ROOT_PATH, "data", "dataset", "score1", "train.jsonl"),
        val_jsonl=os.path.join(ROOT_PATH, "data", "dataset", "score1", "val.jsonl"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 配置训练参数
    trainer.batch_size = 1
    trainer.lr = 3e-6
    trainer.epochs = 20
    trainer.save_dir = "./audio_model_checkpoints"
    
    # 开始训练
    trainer.train()