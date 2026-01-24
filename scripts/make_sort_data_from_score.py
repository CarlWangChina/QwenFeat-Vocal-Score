import csv
import json

json_dir = "/home/xiaoyu/audioscore/audioscore/data/train_ds_4_al/denoise"

def read_json_data():
    file_scores = {}
    for axis in range(4):
        for mode in ["test", "train"]:
            with open(f"{json_dir}/{axis}/{mode}_score.json", "r") as f:
                data = json.load(f)
                for file_score in data:
                    if not file_score["audio"] in file_scores:
                        file_scores[file_score["audio"]] = {}
                    file_scores[file_score["audio"]][axis] = file_score["score"]
    
    print("load data:", len(file_scores))
    for path, file_score in file_scores.items():
        if len(file_score) == 4:
            score = sum(file_score.values())
            print(path, score)


if __name__ == "__main__":
    read_json_data()