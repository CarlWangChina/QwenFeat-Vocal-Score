import os
import sys
import json
import random
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_PATH = os.path.join(ROOT_PATH, "data", "score")
SCORE_OUTPUT_PATH = os.path.join(ROOT_PATH, "data", "dataset")
sys.path.append(os.path.join(ROOT_PATH, "src"))
from collections import Counter

def find_mode(numbers):
    if not numbers:
        return None
    count = Counter(numbers)
    max_freq = max(count.values())
    candidates = [num for num, freq in count.items() if freq == max_freq]
    return min(candidates)

if __name__ == "__main__":
    score1_train_out = open(os.path.join(SCORE_OUTPUT_PATH, "score1", "train.jsonl"), "w")
    score1_val_out = open(os.path.join(SCORE_OUTPUT_PATH, "score1", "val.jsonl"), "w")
    score1_full_out = open(os.path.join(SCORE_OUTPUT_PATH, "score1", "full.jsonl"), "w")
    score2_train_out = open(os.path.join(SCORE_OUTPUT_PATH, "score2", "train.jsonl"), "w")
    score2_val_out = open(os.path.join(SCORE_OUTPUT_PATH, "score2", "val.jsonl"), "w")
    score2_full_out = open(os.path.join(SCORE_OUTPUT_PATH, "score2", "full.jsonl"), "w")

    dataset_score1 = []
    dataset_score2 = []

    with open(os.path.join(ROOT_PATH, "data", "scores.json")) as f:
        scores = json.load(f)
        for songid,song in scores.items():
            for segid,seg in song.items():
                print(seg["score"])
                # 求众数

                scores1 = [item["score1"] for _, item in seg["score"].items()]
                scores2 = [item["score2"] for _, item in seg["score"].items()]
                
                score1 = find_mode(scores1)
                score2 = find_mode(scores2)

                print(score1, score2)

                for file_path in seg["seg_files"]:
                    data_score1 = {"path": file_path, "score": score1}
                    data_score2 = {"path": file_path, "score": score2}
                    score1_full_out.write(json.dumps(data_score1) + "\n")
                    score2_full_out.write(json.dumps(data_score2) + "\n")
                    dataset_score1.append(data_score1)
                    dataset_score2.append(data_score2)
    
    # shuffle
    random.shuffle(dataset_score1)
    random.shuffle(dataset_score2)

    # split
    train_size = int(len(dataset_score1) * 0.9)
    train_data_score1 = dataset_score1[:train_size]
    val_data_score1 = dataset_score1[train_size:]
    train_data_score2 = dataset_score2[:train_size]
    val_data_score2 = dataset_score2[train_size:]

    # write
    for data in train_data_score1:
        score1_train_out.write(json.dumps(data) + "\n")
    for data in val_data_score1:
        score1_val_out.write(json.dumps(data) + "\n")
    for data in train_data_score2:
        score2_train_out.write(json.dumps(data) + "\n")
    for data in val_data_score2:
        score2_val_out.write(json.dumps(data) + "\n")
    score1_train_out.close()
    score1_val_out.close()
    score1_full_out.close()
    score2_train_out.close()
    score2_val_out.close()
    score2_full_out.close()
    print("done")
