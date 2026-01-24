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
import audioscore.trainer_siamese

import argparse

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '23456'  # 选择空闲端口

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_axis', type=int, required=True)
    args = parser.parse_args()
    score_axis = args.score_axis
    # 如果是分布式训练，初始化进程组
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print("rank:", rank, "world_size:", world_size)
    device = f"cuda"

    # 初始化模型和处理器
    # model = audioscore.model.MuqFeatCmpNetwork()
    # model = audioscore.model.AudioFeatCmpNetwork_rl(seq_size=163840)
    model = audioscore.model.AudioFeatCmpNetwork_music(seq_size=163840)
    # model = audioscore.model.MertFeatCmpNetwork(seq_size=163840)
    # model = audioscore.model.AudioFeatCmpNetwork(seq_size=163840)
    # model = audioscore.model.AudioFeatSiameseNetwork_cross(seq_size=163840)
    if dist.get_rank() == 0:
        print(model)

    # dataset_pkg_path = "data/processed_mert/"
    dataset_pkg_path = "data/processed/"
    # dataset_pkg_path = "data/processed_contact/"
    # dataset_pkg_path = "data/processed_muq/"
    # 创建训练器
    trainer = audioscore.trainer_siamese.Trainer(
        model=model,
        train_json=(dataset_pkg_path, f"data/train_ds_4_al/denoise/{score_axis}/train_score.json"),
        val_json=(dataset_pkg_path, f"data/train_ds_4_al/denoise/{score_axis}/test_score.json"),
        device=device,
        local_rank=rank,
        world_size=world_size,
        # data_type="use_tensor",
        data_type="use_audio_feat",
        # data_type="use_full_audio",
    )
    
    # 配置训练参数
    trainer.batch_size = 1
    trainer.lr = 3e-5
    trainer.epochs = 300
    trainer.save_dir = f"ckpts/AudioFeatCmpNetwork_music/{score_axis}"
    trainer.contact_batch = True
    trainer.use_pitch_feature = True
    
    # �***REMOVED***始训练
    trainer.train()