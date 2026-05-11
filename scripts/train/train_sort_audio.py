import os
import sys
import torch
import pickle
import numpy as np
import json
import random
import re
from tqdm import tqdm
from multiprocessing import Process, Queue
from queue import Empty
import pyworld as pw
from torch import distributed as dist

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import audioscore.model
import audioscore.trainer_pairwise_audio
from audioscore.dataset import AudioDataset_pairwise_audio, ROOT_DIR as DATASET_ROOT_DIR

import argparse
import datetime

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '23456'  # 选择空闲端口

def resolve_audio_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(DATASET_ROOT_DIR, path)


def parse_step_from_checkpoint_path(path):
    if not path:
        return 0
    dirname = os.path.basename(os.path.normpath(path))
    match = re.search(r"best_model_step_(\d+)", dirname)
    if match:
        return int(match.group(1))
    return 0


def precache_audio(
    json_path,
    cache_dir,
    cache_base_dir,
    cache_kind,
    target_sr,
    segment_duration,
    max_duration,
    seed,
):
    if not json_path or not cache_dir:
        return
    if not os.path.exists(json_path):
        print(f"Pre-cache skipped, missing json: {json_path}")
        return

    random.seed(seed)
    np.random.seed(seed)

    with open(json_path, "r") as f:
        data = json.load(f)

    audio_files = []
    for group in data:
        for item in group:
            audio_files.append(item["audio_file"])

    seen = set()
    unique_files = []
    for path in audio_files:
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)

    for rel_path in tqdm(unique_files, desc=f"Pre-caching {os.path.basename(json_path)}"):
        audio_path = resolve_audio_path(rel_path)
        cache_path = AudioDataset_pairwise_audio.build_cache_path(
            audio_path, cache_dir, cache_base_dir
        )
        if cache_path and os.path.exists(cache_path):
            continue

        if cache_kind == "full":
            audio_tensor = AudioDataset_pairwise_audio.load_m4a_full_as_tensor(
                audio_path,
                target_sr=target_sr,
                max_duration=max_duration,
            )[0]
        else:
            audio_tensor = AudioDataset_pairwise_audio.load_m4a_as_tensor(
                audio_path,
                target_sr=target_sr,
                segment_duration=segment_duration,
            )[0]

        AudioDataset_pairwise_audio.save_cached_tensor(cache_path, audio_tensor, target_sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_model",
        type=str,
        default="",
        help="Checkpoint directory to resume from, e.g. /home/cqy/Audioscore/ckpts/.../best_model_step_2000",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="dataset/grouped_4171_95_5_h2l/train_4171_95.json",
        help="Grouped JSON used for training.",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default="dataset/grouped_4171_95_5_h2l/val_4171_05.json",
        help="Grouped JSON used for validation.",
    )
    args = parser.parse_args()
    # 如果是分布式训练，初始化进程组
    dist.init_process_group(backend='gloo', timeout=datetime.timedelta(seconds=7200))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print("rank:", rank, "world_size:", world_size)
    device = f"cuda"

    train_json = args.train_json
    val_json = args.val_json
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"--train_json not found: {train_json}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"--val_json not found: {val_json}")
    cache_base_dir = "/cache/cqy_cache/audio-score"
    dataset_tag = os.path.basename(os.path.dirname(train_json))
    cache_root = os.path.join(cache_base_dir, f"feature_{dataset_tag}")
    train_cache_dir = os.path.join(cache_root, "train_full")
    val_cache_dir = os.path.join(cache_root, "val_60s")
    target_sr = 24000
    segment_duration = 60
    max_duration = None

    if rank == 0:
        print(f"Train json: {train_json}")
        print(f"Val json: {val_json}")
        print(f"Cache root: {cache_root}")

    if rank == 0:
        precache_audio(
            train_json,
            train_cache_dir,
            cache_base_dir,
            cache_kind="full",
            target_sr=target_sr,
            segment_duration=segment_duration,
            max_duration=max_duration,
            seed=1234,
        )
        precache_audio(
            val_json,
            val_cache_dir,
            cache_base_dir,
            cache_kind="segment",
            target_sr=target_sr,
            segment_duration=segment_duration,
            max_duration=None,
            seed=1234,
        )
    dist.barrier()

    # 初始化模型和处理器
    model = audioscore.model.SongEvalGenerator_audio_lora(use_grl=False) # 改为use_grl=True后启用对抗训练
    # model.load_model("/data/nfs/audioscore/ckpts/SongEvalGenerator/step_1_mul/best_model_step_98000")
    resume_model_path = args.resume_model.strip()
    initial_global_step = 0
    if resume_model_path:
        if not os.path.exists(resume_model_path):
            raise FileNotFoundError(f"--resume_model not found: {resume_model_path}")
        model.load_model(resume_model_path)
        initial_global_step = parse_step_from_checkpoint_path(resume_model_path)
        if rank == 0:
            print(f"Resume model: {resume_model_path}")
            print(f"Resume global_step: {initial_global_step}")
    else:
        base_model_path = "ckpts/SongEvalGenerator/step_2_al_audio/best_model_step_132000"
        model.load_model(base_model_path)
        if rank == 0:
            print(f"Start new training from base model: {base_model_path}")
    # model = audioscore.model.MuqFeatSortNetwork()
    # model = audioscore.model.MuqFeatCmpSortNetwork()

    # 对抗训练的模型
    # spk_encoder = audioscore.model.SpkEncoderHelper().cuda()

    # if dist.get_rank() == 0:
    #     print(model)

    # 创建训练器
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = audioscore.trainer_pairwise_audio.Trainer(
        model=model,
        # spk_encoder=spk_encoder, # 对抗训练的模型（可选）
        spk_encoder=None,
        train_json=train_json,
        val_json=val_json,
        device=device,
        local_rank=rank,
        world_size=world_size,
        data_type="use_tensor",
        save_dir=os.path.join("ckpts/SongEvalGenerator/step_3_extended_audio", run_id),
        train_cache_dir=train_cache_dir,
        train_cache_mode="read_write",
        train_cache_base_dir=cache_base_dir,
        train_cache_kind="full",
        train_cache_target_sr=target_sr,
        train_cache_segment_duration=segment_duration,
        train_cache_max_duration=max_duration,
        val_cache_dir=val_cache_dir,
        val_cache_mode="read_write",
        val_cache_base_dir=cache_base_dir,
        val_cache_kind="segment",
        val_cache_target_sr=target_sr,
        val_cache_segment_duration=segment_duration,
        initial_global_step=initial_global_step,
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
    trainer.eval_steps = 500
    
    # 开始训练
    trainer.train()
