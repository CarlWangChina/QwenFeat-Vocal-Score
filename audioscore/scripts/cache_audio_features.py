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


def link_cache_file(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = f"{dst_path}.tmp"
    if os.path.lexists(tmp_path):
        os.remove(tmp_path)
    try:
        os.link(src_path, tmp_path)
    except OSError:
        os.symlink(src_path, tmp_path)
    os.replace(tmp_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Build a shared audio tensor cache for pairwise audio training/validation.")
    parser.add_argument(
        "--json",
        nargs="+",
        default=[
            "dataset/ktv_rerecord_3760/pairwise/train_mixed_3375.json",
            "dataset/ktv_rerecord_3760/pairwise/test_ktv_lt85_385.json",
        ],
        help="One or more dataset json files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/cache/cqy_cache/audio-score/pre_store_3760_full_24k",
        help="Directory to store cached tensors.",
    )
    parser.add_argument(
        "--cache-base-dir",
        default="/cache/cqy_cache/audio-score",
        help="Base directory used to preserve relative paths.",
    )
    parser.add_argument("--cache-kind", choices=["segment", "full"], default="full", help="Cache kind.")
    parser.add_argument(
        "--reuse-cache-dir",
        action="append",
        default=["/cache/cqy_cache/audio-score/feature_grouped_3760_95_5_h2l/train_full"],
        help="Existing cache directory to hardlink/symlink from before decoding missing audio. Can be repeated.",
    )
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
    parser.add_argument(
        "--skip-decode-missing",
        action="store_true",
        help="Only map existing reusable cache files; do not decode raw audio for missing files.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    audio_files = []
    for json_path in args.json:
        with open(json_path, "r") as f:
            data = json.load(f)
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
    existing_count = 0
    mapped_count = 0
    decoded_count = 0
    missing_count = 0

    for rel_path in tqdm(unique_files, desc="Caching audio"):
        audio_path = resolve_audio_path(rel_path)
        cache_path = AudioDataset_pairwise_audio.build_cache_path(
            audio_path, args.cache_dir, cache_base_dir
        )
        if args.skip_existing and cache_path and os.path.exists(cache_path):
            existing_count += 1
            continue

        mapped = False
        if args.cache_kind == "full":
            for reuse_cache_dir in args.reuse_cache_dir:
                reuse_cache_path = AudioDataset_pairwise_audio.build_cache_path(
                    audio_path, reuse_cache_dir, cache_base_dir
                )
                if (
                    reuse_cache_path
                    and os.path.exists(reuse_cache_path)
                    and os.path.abspath(reuse_cache_path) != os.path.abspath(cache_path)
                ):
                    link_cache_file(reuse_cache_path, cache_path)
                    mapped_count += 1
                    mapped = True
                    break
        if mapped:
            continue

        if args.skip_decode_missing:
            missing_count += 1
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
        decoded_count += 1

    print(
        "Cache build done: "
        f"total={len(unique_files)} existing={existing_count} "
        f"mapped={mapped_count} decoded={decoded_count} missing={missing_count} "
        f"cache_dir={args.cache_dir}"
    )


if __name__ == "__main__":
    main()
