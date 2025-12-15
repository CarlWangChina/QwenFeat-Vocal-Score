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
import audioscore.trainer_pairwise_audio

import argparse
import datetime

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '23456'  # 选择空闲端口

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # 如果是分布式训练，初始化进程组
    dist.init_process_group(backend='gloo', timeout=datetime.timedelta(seconds=7200))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print("rank:", rank, "world_size:", world_size)
    device = f"cuda"

    # 初始化模型和处理器
    model = audioscore.model.SongEvalGenerator_audio_lora(use_grl=True) # 改为use_grl=True后启用对抗训练
    # model.load_model("/data/nfs/audioscore/ckpts/SongEvalGenerator/step_1_mul/best_model_step_98000")
    model.load_model("/data/nfs/audioscore/ckpts/SongEvalGenerator/step_1_mul_audio_grl/best_model_step_74000")
    # model = audioscore.model.MuqFeatSortNetwork()
    # model = audioscore.model.MuqFeatCmpSortNetwork()

    # 对抗训练的模型
    spk_encoder = audioscore.model.SpkEncoderHelper().cuda()

    # if dist.get_rank() == 0:
    #     print(model)

    # 创建训练器
    trainer = audioscore.trainer_pairwise_audio.Trainer(
        model=model,
        spk_encoder=spk_encoder, # 对抗训练的模型（可选）
        spk_encoder=None,
        train_json="data/sort/data/index/train.json",
        val_json="data/sort/data/index/val.json",
        device=device,
        local_rank=rank,
        world_size=world_size,
        data_type="use_tensor",
        save_dir="ckpts/SongEvalGenerator/step_2_al_audio/",
        # data_type="use_audio_feat",
        # data_type="use_full_audio",
    )
    
    # 配置训练参数
    trainer.batch_size = 1
    trainer.lr = 1e-5
    trainer.epochs = 100
    trainer.gamma = 0.9
    trainer.grl_gm = 0.5
    trainer.contact_batch = True
    trainer.use_step_lr = False
    
    # 开始训练
    trainer.train()