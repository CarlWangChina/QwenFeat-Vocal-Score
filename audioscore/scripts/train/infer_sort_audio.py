import argparse
import datetime
import json
import os
import random
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
    )
)

import audioscore.model
from audioscore.dataset import AudioDataset_pairwise_audio, ROOT_DIR as DATASET_ROOT_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_grouped(model, dataset, device, inverse=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    all_true = []
    all_pred = []
    predictions = []

    with torch.no_grad():
        for group_idx, group in enumerate(tqdm(dataset.target, desc="Evaluating groups")):
            items = []
            for item in group:
                audio_file = item["audio_file"]
                if dataset.file_exist(audio_file):
                    items.append(item)

            if len(items) < 2:
                continue

            preds = []
            trues = []
            for item in items:
                audio_path = os.path.join(DATASET_ROOT_DIR, item["audio_file"])
                wav = dataset.load_audio_with_cache(audio_path).to(device)
                pred = model(wavs=wav).detach().float().cpu().view(-1)[0]
                if inverse:
                    pred = 5.0 - pred
                true = float(item["score"])
                pred_value = float(pred.item())

                preds.append(pred)
                trues.append(true)
                all_true.append(true)
                all_pred.append(pred_value)
                predictions.append(
                    {
                        "group_idx": group_idx,
                        "audio_file": item["audio_file"],
                        "true_score": true,
                        "pred_score": pred_value,
                    }
                )

            scores_tensor = torch.stack(preds).view(-1)
            n = scores_tensor.numel()
            if n < 2:
                continue

            diff = scores_tensor[:, None] - scores_tensor[None, :]
            true_scores = torch.tensor(trues, dtype=diff.dtype)
            true_diff = true_scores[:, None] - true_scores[None, :]

            base_mask = ~torch.eye(n, dtype=torch.bool)
            valid_mask = base_mask & (true_diff != 0)
            if not valid_mask.any():
                continue

            label = (true_diff > 0).to(diff.dtype)
            total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                diff[valid_mask], label[valid_mask], reduction="sum"
            ).item()

            preds_matrix = diff > 0
            correct = (preds_matrix == label.bool())[valid_mask].sum().item()
            total_correct += correct
            total_pairs += int(valid_mask.sum().item())

    avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
    accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0

    pearson = 0.0
    spearman = 0.0
    if len(all_true) > 1 and np.std(all_true) > 0 and np.std(all_pred) > 0:
        pearson = float(pearsonr(all_true, all_pred)[0])
        spearman = float(spearmanr(all_true, all_pred)[0])

    metrics = {
        "pairwise_avg_loss": float(avg_loss),
        "pairwise_accuracy": float(accuracy),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "num_groups": int(len(dataset.target)),
        "num_items": int(len(predictions)),
        "num_pairs": int(total_pairs),
    }
    return metrics, predictions


def build_default_output_path(model_path: str, val_json: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    val_name = os.path.splitext(os.path.basename(val_json))[0]
    return os.path.join(model_path, f"infer_{val_name}_{timestamp}.json")


def main():
    parser = argparse.ArgumentParser(description="Run pure inference on grouped audio val set.")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint directory path")
    parser.add_argument("--val_json", type=str, required=True, help="Validation grouped json path")
    parser.add_argument("--output_json", type=str, default="", help="Where to save metrics + predictions")
    parser.add_argument("--device", type=str, default="", help="e.g. cuda:0 / cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Apply the legacy online mapping `5 - raw_score` before evaluation.",
    )

    parser.add_argument("--cache_base_dir", type=str, default="/cache/cqy_cache/audio-score")
    parser.add_argument("--val_cache_dir", type=str, default="")
    parser.add_argument(
        "--val_cache_mode",
        type=str,
        default="read",
        choices=["none", "read", "write", "read_write"],
    )
    parser.add_argument(
        "--val_cache_kind",
        type=str,
        default="full",
        choices=["segment", "full"],
    )
    parser.add_argument(
        "--val_cache_cut_mode",
        type=str,
        default="stable",
        choices=["random", "center", "stable"],
    )
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--segment_duration", type=int, default=60)
    parser.add_argument("--max_duration", type=float, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")
    if not os.path.isfile(args.val_json):
        raise FileNotFoundError(f"val_json not found: {args.val_json}")

    set_seed(args.seed)

    device = args.device.strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")

    val_cache_dir = args.val_cache_dir.strip()
    if not val_cache_dir:
        val_cache_dir = os.path.join(args.cache_base_dir, "pre_store_3760_full_24k")

    model = audioscore.model.SongEvalGenerator_audio_lora(use_grl=False)
    model.load_model(args.model_path)
    model = model.to(device)

    val_dataset = AudioDataset_pairwise_audio(
        args.val_json,
        cache_dir=val_cache_dir,
        cache_mode=args.val_cache_mode,
        cache_base_dir=args.cache_base_dir,
        cache_target_sr=args.target_sr,
        cache_segment_duration=args.segment_duration,
        cache_kind=args.val_cache_kind,
        cache_max_duration=args.max_duration,
        cache_cut_mode=args.val_cache_cut_mode,
    )

    metrics, predictions = evaluate_grouped(
        model=model,
        dataset=val_dataset,
        device=device,
        inverse=args.inverse,
    )

    output_json = args.output_json.strip() or build_default_output_path(args.model_path, args.val_json)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    payload = {
        "model_path": args.model_path,
        "val_json": args.val_json,
        "device": device,
        "seed": args.seed,
        "inverse": args.inverse,
        "val_cache_dir": val_cache_dir,
        "val_cache_mode": args.val_cache_mode,
        "val_cache_kind": args.val_cache_kind,
        "val_cache_cut_mode": args.val_cache_cut_mode,
        "metrics": metrics,
        "predictions": predictions,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Inference finished.")
    print(f"pairwise_avg_loss={metrics['pairwise_avg_loss']:.6f}")
    print(f"pairwise_accuracy={metrics['pairwise_accuracy']:.6f}")
    print(f"pearson={metrics['pearson']:.6f}")
    print(f"spearman={metrics['spearman']:.6f}")
    print(f"num_items={metrics['num_items']}")
    print(f"num_pairs={metrics['num_pairs']}")
    print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
