import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))
# print(sys.path)
import torch
import qwenaudio.model
from qwenaudio.trainer_pf_score import QwenAudioTrainer,QwenAudioScoreModel
from torch import distributed as dist
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioProcessor
import torch.distributed as dist
import argparse


if __name__ == "__main__":
    # 如果是分布式训练，初始化进程组
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print("rank:", rank, "world_size:", world_size)
    device = f"cuda:{rank}"

    # 初始化模型和处理器
    processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", sample_rate=16000)
    model = QwenAudioScoreModel(
        output_num=5, 
        freeze_weight=False, 
        use_lora=True,
        lora_r=16, 
        lora_alpha=16)  # 注意：这里使用了正确的模型引用
    if dist.get_rank() == 0:
        print(model)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):  # 仅检查全连接层
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        print(f"模块 {name} 的参数 {param_name} 未冻结")

    # 创建训练器
    trainer = QwenAudioTrainer(
        model=model,
        processor=processor,
        train_json=os.path.join(ROOT_PATH, "data", "train_pf_score", "train.json"),
        val_json=os.path.join(ROOT_PATH, "data", "train_pf_score", "test.json"),
        device=device,
        local_rank=rank,
        world_size=world_size
    )
    
    # 配置训练参数
    trainer.batch_size = 1
    trainer.lr = 3e-6
    trainer.epochs = 100
    trainer.save_dir = "./ckpts/score_lora-16-16-pf_score"
    
    # 开始训练
    trainer.train()