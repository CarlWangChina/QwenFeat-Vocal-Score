import argparse
import datetime
import os
import random
import re
import sys

import numpy as np
import torch
from torch import distributed as dist

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from audioscore.model_regression_audio import MuQRegressionAudioLora
from audioscore.trainer_regression_audio import Trainer


def parse_step_from_checkpoint_path(path):
    if not path:
        return 0
    dirname = os.path.basename(os.path.normpath(path))
    match = re.search(r"(?:best_model_step|step)_(\d+)", dirname)
    if match:
        return int(match.group(1))
    return 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return -1, 1, 0
    dist.init_process_group(
        backend="gloo",
        timeout=datetime.timedelta(seconds=7200),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MuQ LoRA with direct 0-100 score regression.")
    parser.add_argument(
        "--resume_model",
        type=str,
        default="",
        help="Regression checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="dataset/ktv_rerecord_3760/regression/train_mixed_3375.json",
        help="Flat regression JSON used for training.",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default="dataset/ktv_rerecord_3760/regression/test_ktv_lt85_385.json",
        help="Flat pure-KTV JSON used for final evaluation.",
    )
    parser.add_argument("--cache_base_dir", type=str, default="/cache/cqy_cache/audio-score")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--save_root", type=str, default="ckpts/SongEvalGenerator/regression_audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5, help="Fallback LR for other trainable parameters.")
    parser.add_argument("--head_lr", type=float, default=1e-5, help="Regression head LR.")
    parser.add_argument("--lora_lr", type=float, default=5e-5, help="MuQ LoRA LR.")
    parser.add_argument("--ln_lr", type=float, default=1e-5, help="MuQ LayerNorm LR.")
    parser.add_argument("--lora_r", type=int, default=128, help="MuQ LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=48, help="MuQ LoRA alpha.")
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_max_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--segment_duration", type=int, default=90)
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument(
        "--score_scale",
        type=float,
        default=100.0,
        help="Divide percent scores by this value when computing regression loss.",
    )
    parser.add_argument(
        "--huber_beta",
        type=float,
        default=0.05,
        help="SmoothL1 beta in normalized score units. 0.05 equals 5 points when score_scale=100.",
    )
    parser.add_argument("--min_save_pearson", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not os.path.exists(args.train_json):
        raise FileNotFoundError(f"--train_json not found: {args.train_json}")
    if not os.path.exists(args.val_json):
        raise FileNotFoundError(f"--val_json not found: {args.val_json}")

    rank, world_size, local_rank = init_distributed()
    set_seed(args.seed + (0 if rank < 0 else rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = "cuda"
    else:
        device = "cpu"

    cache_dir = args.cache_dir.strip() or os.path.join(args.cache_base_dir, "pre_store_3760_full_24k")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"Full-audio cache not found: {cache_dir}. "
            "Run scripts/cache_audio_features.py before training."
        )

    if rank in [-1, 0]:
        print("Regression train json:", args.train_json)
        print("Regression val json:", args.val_json)
        print("Full-audio cache:", cache_dir)
        print("Score scale: 0-100")
        print("Segment duration:", args.segment_duration)
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    model = MuQRegressionAudioLora(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    resume_model_path = args.resume_model.strip()
    initial_global_step = 0
    if resume_model_path:
        if not os.path.isdir(resume_model_path):
            raise FileNotFoundError(f"--resume_model not found: {resume_model_path}")
        model.load_model(resume_model_path)
        initial_global_step = parse_step_from_checkpoint_path(resume_model_path)
        if rank in [-1, 0]:
            print(f"Resume regression model: {resume_model_path}")
            print(f"Resume global_step: {initial_global_step}")
    elif rank in [-1, 0]:
        print("Starting regression training from original MuQ + fresh regression head.")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = os.path.basename(os.path.dirname(args.train_json))
    trainer = Trainer(
        model=model,
        train_json=args.train_json,
        val_json=args.val_json,
        device=device,
        local_rank=rank,
        world_size=world_size,
        save_dir=os.path.join(args.save_root, dataset_tag, run_id),
        train_cache_dir=cache_dir,
        train_cache_mode="read",
        train_cache_base_dir=args.cache_base_dir,
        train_cache_kind="full",
        train_cache_cut_mode="random",
        train_cache_target_sr=args.target_sr,
        train_cache_segment_duration=args.segment_duration,
        val_cache_dir=cache_dir,
        val_cache_mode="read",
        val_cache_base_dir=args.cache_base_dir,
        val_cache_kind="full",
        val_cache_cut_mode="stable",
        val_cache_target_sr=args.target_sr,
        val_cache_segment_duration=args.segment_duration,
        initial_global_step=initial_global_step,
    )

    trainer.batch_size = args.batch_size
    trainer.lr = args.lr
    trainer.head_lr = args.head_lr
    trainer.lora_lr = args.lora_lr
    trainer.ln_lr = args.ln_lr
    trainer.epochs = args.epochs
    trainer.lr_warmup_ratio = args.warmup_ratio
    trainer.lr_warmup_max_steps = args.warmup_max_steps
    trainer.min_lr = args.min_lr
    trainer.eval_steps = args.eval_steps
    trainer.log_steps = args.log_steps
    trainer.score_scale = args.score_scale
    trainer.huber_beta = args.huber_beta
    trainer.min_save_pearson = args.min_save_pearson
    trainer.seed = args.seed

    trainer.train()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
