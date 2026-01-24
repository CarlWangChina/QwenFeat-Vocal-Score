import os
import sys
import torch
import pickle
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty
import pyworld as pw
from torch import distributed as dist

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import audioscore.model
import audioscore.trainer_pairwise_hybrid

import argparse

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '23456'  # 选择空闲端口

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # 如果是分布式训练，初始化进程组
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print("rank:", rank, "world_size:", world_size)
    device = f"cuda"

    # 初始化模型和处理器
    model = audioscore.model.AudioFeatHybrid()
    # model = audioscore.model.MuqFeatCmpSortNetwork()
    if dist.get_rank() == 0:
        print(model)

    # 创建训练器
    trainer = audioscore.trainer_pairwise_hybrid.Trainer(
        model=model,
        train_json="data/sort/data/index/train.json",
        val_json="data/sort/data/index/val.json",
        device=device,
        local_rank=rank,
        world_size=world_size,
        data_type="use_tensor",
        save_dir="ckpts/AudioFeatHybrid_pairwise/",
        # data_type="use_audio_feat",
        # data_type="use_full_audio",
    )
    
    # 配置训练参数
    trainer.batch_size = 8
    trainer.lr = 5e-4
    trainer.epochs = 100
    trainer.contact_batch = True
    trainer.use_step_lr = False
    
    # 开始训练
    trainer.train()