import os
import sys
import torch
import re
from torch import distributed as dist

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import audioscore.model
import audioscore.trainer_pairwise_audio

import argparse
import datetime

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '23456'  # 选择空闲端口

def parse_step_from_checkpoint_path(path):
    if not path:
        return 0
    dirname = os.path.basename(os.path.normpath(path))
    match = re.search(r"best_model_step_(\d+)", dirname)
    if match:
        return int(match.group(1))
    return 0


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
        default="dataset/ktv_rerecord_3760/pairwise/train_mixed_3375.json",
        help="Grouped JSON used for training.",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default="dataset/ktv_rerecord_3760/pairwise/test_ktv_lt85_385.json",
        help="Grouped pure-KTV JSON used for final evaluation.",
    )
    parser.add_argument("--cache_base_dir", default="/cache/cqy_cache/audio-score")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument(
        "--save_root",
        default="ckpts/SongEvalGenerator/step_3_ktv_rerecord_pairwise",
    )
    parser.add_argument("--segment_duration", type=int, default=60)
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=10)
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
    cache_base_dir = args.cache_base_dir
    cache_root = args.cache_dir.strip() or os.path.join(
        cache_base_dir,
        "pre_store_3760_full_24k",
    )
    train_cache_dir = cache_root
    val_cache_dir = cache_root
    target_sr = args.target_sr
    segment_duration = args.segment_duration
    max_duration = None

    if rank == 0:
        print(f"Train json: {train_json}")
        print(f"Val json: {val_json}")
        print(f"Full-audio cache: {cache_root}")
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(
            f"Full-audio cache not found: {cache_root}. "
            "Run scripts/cache_audio_features.py before training."
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
        if rank == 0:
            print("Starting from original MuQ + SongEvalGenerator base weights (fresh LoRA)")
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
        save_dir=os.path.join(args.save_root, run_id),
        train_cache_dir=train_cache_dir,
        train_cache_mode="read",
        train_cache_base_dir=cache_base_dir,
        train_cache_kind="full",
        train_cache_cut_mode="random",
        train_cache_target_sr=target_sr,
        train_cache_segment_duration=segment_duration,
        train_cache_max_duration=max_duration,
        val_cache_dir=val_cache_dir,
        val_cache_mode="read",
        val_cache_base_dir=cache_base_dir,
        val_cache_kind="full",
        val_cache_cut_mode="stable",
        val_cache_target_sr=target_sr,
        val_cache_segment_duration=segment_duration,
        initial_global_step=initial_global_step,
        # data_type="use_audio_feat",
        # data_type="use_full_audio",
    )
    
    # 配置训练参数
    trainer.batch_size = 1
    trainer.lr = 1e-5        # head + LayerNorm LR
    trainer.lora_lr = 5e-5   # LoRA adapter LR（更快适配）
    trainer.epochs = args.epochs
    trainer.lr_warmup_ratio = 0.05
    trainer.lr_warmup_max_steps = 2000
    trainer.min_lr = 1e-7
    trainer.min_save_pearson = 0.4
    trainer.gamma = 0.9
    trainer.grl_gm = 0.5
    trainer.contact_batch = True
    trainer.use_step_lr = False
    trainer.eval_steps = 500
    
    # 开始训练
    trainer.train()
