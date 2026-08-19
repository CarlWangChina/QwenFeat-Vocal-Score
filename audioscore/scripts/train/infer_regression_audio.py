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

from audioscore.dataset_regression_audio import AudioDataset_regression_audio
from audioscore.model_regression_audio import MuQRegressionAudioLora


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    true_scores = []
    pred_scores = []
    predictions = []

    for index in tqdm(range(len(dataset)), desc="Regression inference"):
        waveform, true_score, audio_file = dataset[index]
        prediction = model(wavs=waveform.to(device)).detach().float().cpu().view(-1)[0]
        pred_score = float(prediction.item())
        true_score = float(true_score)
        true_scores.append(true_score)
        pred_scores.append(pred_score)
        predictions.append(
            {
                "audio_file": audio_file,
                "true_score": true_score,
                "pred_score": pred_score,
            }
        )

    true_array = np.asarray(true_scores, dtype=np.float64)
    pred_array = np.asarray(pred_scores, dtype=np.float64)
    errors = pred_array - true_array
    metrics = {
        "pearson": float(pearsonr(true_array, pred_array)[0]),
        "spearman": float(spearmanr(true_array, pred_array)[0]),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "num_items": int(len(predictions)),
    }
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MuQ direct-regression checkpoint.")
    parser.add_argument("--model_path", required=True, help="Regression checkpoint directory.")
    parser.add_argument("--val_json", required=True, help="Flat or grouped validation JSON.")
    parser.add_argument("--output_json", default="", help="Metrics and predictions output path.")
    parser.add_argument("--device", default="", help="For example cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache_base_dir", default="/cache/cqy_cache/audio-score")
    parser.add_argument("--val_cache_dir", default="")
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--segment_duration", type=int, default=90)
    parser.add_argument(
        "--val_cache_cut_mode",
        choices=["random", "center", "stable"],
        default="stable",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")
    if not os.path.isfile(args.val_json):
        raise FileNotFoundError(f"val_json not found: {args.val_json}")

    set_seed(args.seed)
    device = args.device.strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_dir = args.val_cache_dir.strip() or os.path.join(
        args.cache_base_dir,
        "pre_store_3760_full_24k",
    )

    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        model_config = json.load(config_file)
    model = MuQRegressionAudioLora(
        lora_r=int(model_config.get("lora_r", 128)),
        lora_alpha=int(model_config.get("lora_alpha", 48)),
        hidden_state_index=int(model_config.get("hidden_state_index", 6)),
    )
    model.load_model(args.model_path)
    model = model.to(device)

    dataset = AudioDataset_regression_audio(
        args.val_json,
        cache_dir=cache_dir,
        cache_mode="read",
        cache_base_dir=args.cache_base_dir,
        cache_target_sr=args.target_sr,
        cache_segment_duration=args.segment_duration,
        cache_kind="full",
        cache_cut_mode=args.val_cache_cut_mode,
    )
    metrics, predictions = evaluate(model, dataset, device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = args.output_json.strip() or os.path.join(
        args.model_path,
        f"infer_regression_{timestamp}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    payload = {
        "model_path": args.model_path,
        "val_json": args.val_json,
        "device": device,
        "seed": args.seed,
        "val_cache_dir": cache_dir,
        "val_cache_cut_mode": args.val_cache_cut_mode,
        "segment_duration": args.segment_duration,
        "metrics": metrics,
        "predictions": predictions,
    }
    with open(output_json, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)

    print("Inference finished.")
    for name, value in metrics.items():
        print(f"{name}={value}")
    print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
