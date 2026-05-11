import argparse
import json
import os
import random
import sys

import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from audioscore.dataset import AudioDataset_pairwise_audio, ROOT_DIR as DATASET_ROOT_DIR


def resolve_audio_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(DATASET_ROOT_DIR, path)


def main():
    parser = argparse.ArgumentParser(description="Cache audio clips for pairwise audio training/validation.")
    parser.add_argument(
        "--json",
        default="dataset/grouped_4171_95_5_h2l/val_4171_05.json",
        help="Path to dataset json.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/cache/cqy_cache/audio-score/feature_grouped_4171_95_5_h2l/val_60s",
        help="Directory to store cached tensors.",
    )
    parser.add_argument(
        "--cache-base-dir",
        default="/cache/cqy_cache/audio-score",
        help="Base directory used to preserve relative paths.",
    )
    parser.add_argument("--cache-kind", choices=["segment", "full"], default="segment", help="Cache kind.")
    parser.add_argument("--target-sr", type=int, default=24000, help="Target sample rate.")
    parser.add_argument("--segment-duration", type=int, default=60, help="Clip length in seconds.")
    parser.add_argument(
        "--max-duration",
        type=int,
        default=0,
        help="Max duration for full cache in seconds (0 = no limit).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for clipping.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = no limit).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already cached.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.json, "r") as f:
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

    if args.limit and args.limit > 0:
        unique_files = unique_files[: args.limit]

    cache_base_dir = args.cache_base_dir or os.path.dirname(args.cache_dir)

    for rel_path in tqdm(unique_files, desc="Caching audio"):
        audio_path = resolve_audio_path(rel_path)
        cache_path = AudioDataset_pairwise_audio.build_cache_path(
            audio_path, args.cache_dir, cache_base_dir
        )
        if args.skip_existing and cache_path and os.path.exists(cache_path):
            continue

        if args.cache_kind == "full":
            audio_tensor = AudioDataset_pairwise_audio.load_m4a_full_as_tensor(
                audio_path,
                target_sr=args.target_sr,
                max_duration=args.max_duration if args.max_duration > 0 else None,
            )[0]
        else:
            audio_tensor = AudioDataset_pairwise_audio.load_m4a_as_tensor(
                audio_path,
                target_sr=args.target_sr,
                segment_duration=args.segment_duration,
            )[0]

        AudioDataset_pairwise_audio.save_cached_tensor(cache_path, audio_tensor, args.target_sr)


if __name__ == "__main__":
    main()
