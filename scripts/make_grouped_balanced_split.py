#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build score-balanced grouped JSON splits by dealing score-sorted "
            "items round-robin into groups."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tag", default=None, help="Dataset tag used in output file names.")
    parser.add_argument("--num-groups", required=True, type=int)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def flatten_grouped_json(data: list[Any]) -> list[dict[str, Any]]:
    if not data:
        return []
    if isinstance(data[0], dict):
        return data
    return [item for group in data for item in group]


def summarize(name: str, groups: list[list[dict[str, Any]]]) -> None:
    items = [item for group in groups for item in group]
    scores = [float(item["score"]) for item in items]
    sizes = [len(group) for group in groups]
    means = [statistics.mean(float(item["score"]) for item in group) for group in groups]
    print(f"{name}: groups={len(groups)} items={len(items)}")
    print(f"  group_sizes={dict(sorted(Counter(sizes).items()))}")
    print(
        "  score="
        f"min={min(scores):.1f} max={max(scores):.1f} "
        f"mean={statistics.mean(scores):.3f}"
    )
    print(
        "  group_mean="
        f"min={min(means):.3f} max={max(means):.3f} "
        f"std={statistics.pstdev(means):.3f}"
    )


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    indexed_items = list(enumerate(flatten_grouped_json(data)))
    if not indexed_items:
        raise ValueError(f"No items found in {args.input}")
    if args.num_groups <= 0:
        raise ValueError("--num-groups must be positive")
    if args.num_groups > len(indexed_items):
        raise ValueError("--num-groups cannot exceed item count")

    missing_score = [item for _, item in indexed_items if "score" not in item]
    if missing_score:
        raise ValueError(f"{len(missing_score)} items do not contain a score field")

    # Stable tie handling keeps the input order among items with the same score.
    indexed_items.sort(key=lambda pair: (-float(pair[1]["score"]), pair[0]))

    grouped: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(args.num_groups)]
    for rank, pair in enumerate(indexed_items):
        grouped[rank % args.num_groups].append(pair)

    for group in grouped:
        group.sort(key=lambda pair: (-float(pair[1]["score"]), pair[0]))

    groups = [[item for _, item in group] for group in grouped]

    val_group_count = max(1, round(args.num_groups * args.val_ratio))
    val_group_count = min(val_group_count, args.num_groups - 1)

    group_order = list(range(args.num_groups))
    random.Random(args.seed).shuffle(group_order)
    val_indices = set(group_order[:val_group_count])

    train_groups = [groups[i] for i in group_order if i not in val_indices]
    val_groups = [groups[i] for i in group_order if i in val_indices]

    tag = args.tag or args.input.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"train_{tag}_95.json"
    val_path = args.out_dir / f"val_{tag}_05.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_groups, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val_groups, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summarize("full", groups)
    summarize("train", train_groups)
    summarize("val", val_groups)
    print(f"val_group_indices={sorted(val_indices)}")
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
