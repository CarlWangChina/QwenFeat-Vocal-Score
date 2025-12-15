import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))
# print(sys.path)
import torch
import qwenaudio.model
from qwenaudio.trainer_audio_feat import AudioTrainer
from torch import distributed as dist
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

    parser = argparse.ArgumentParser(description="Qwen2-Audio分布式训练脚本")
    parser.add_argument("--train_set", type=str, required=True, help="训练数据集路径")
    parser.add_argument("--val_set", type=str, required=True, help="验证数据集路径")
    parser.add_argument("--save_dir", type=str, required=True, help="模型保存目录")
    args = parser.parse_args()

    # 初始化模型和处理器
    model = qwenaudio.model.AudioFeatClassifier()
    if dist.get_rank() == 0:
        print(model)

    # 创建训练器
    trainer = AudioTrainer(
        model=model,
        train_json=args.train_set,
        val_json=args.val_set,
        device=device,
        local_rank=rank,
        world_size=world_size
    )
    
    # 配置训练参数
    trainer.batch_size = 1
    trainer.lr = 3e-5
    trainer.epochs = 90
    trainer.save_dir = args.save_dir
    
    # 开始训练
    trainer.train()