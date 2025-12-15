import pandas
import itertools
import json
import numpy as np
from scipy.stats import kendalltau
import os
import random


# ===============================
# 生成所有维度的加法与减法组合
# ===============================
def generate_axis_expressions(axis):
    """生成所有可能的加减组合表达式"""
    combinations = []
    for r in range(1, len(axis) + 1):
        for combo in itertools.combinations(axis, r):
            for signs in itertools.product(["+", "-"], repeat=len(combo)):
                expr = ''.join(
                    (s if i > 0 or s == "-" else "") + combo[i]
                    for i, s in enumerate(signs)
                )
                combinations.append(expr)
    return combinations

def test_songeval_sort():
    csv_paths = ["data/audio_scores_songeval_results.csv", "data/audio_scores_results.csv"]  # 这里添加所有新的CSV路径
    json_path = "data/sort/data/index/val.json"

    # ===============================
    # 读取数据
    # ===============================
    df = pandas.read_csv(csv_paths[0])
    axis = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]

    axis_expressions = generate_axis_expressions(axis)

    # ===============================
    # 计算每个音频文件在各组合下的分数
    # ===============================
    generated_results = {}
    for index, row in df.iterrows():
        scores = {}
        for expr in axis_expressions:
            total = 0
            current_sign = 1
            for part in expr.replace("-", "+-").split("+"):
                if not part:
                    continue
                if part.startswith("-"):
                    total -= row[part[1:]]
                else:
                    total += row[part]
            scores[expr] = total
        generated_results[row["audio_file"]] = scores

    # ===============================
    # 读取人工标注数据
    # ===============================
    with open(json_path, "r") as f:
        json_data = json.load(f)

    metrics = {expr: {"correct_pairs": 0, "kendall_tau": 0, "ndcg": 0} for expr in axis_expressions}
    total_pairs = 0
    group_ndcg_scores = {expr: [] for expr in axis_expressions}

    # ===============================
    # 逐组评估人工排序
    # ===============================
    for group in json_data:
        audio_files = [(item["audio_file"].split("/")[-1], idx) for idx, item in enumerate(group)]
        file_count = len(audio_files)
        total_pairs += file_count * (file_count - 1) // 2
        manual_ranks = {file: rank for file, rank in audio_files}

        for (file1, idx1), (file2, idx2) in itertools.combinations(audio_files, 2):
            for expr in axis_expressions:
                score1 = generated_results[file1][expr]
                score2 = generated_results[file2][expr]
                if (score1 > score2 and idx1 < idx2) or (score1 < score2 and idx1 > idx2):
                    metrics[expr]["correct_pairs"] += 1

        def calculate_dcg(relevance_scores):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

        for expr in axis_expressions:
            file_scores = [(file, generated_results[file][expr]) for file, _ in audio_files]
            ideal_relevance = [file_count - rank for _, rank in audio_files]
            file_scores_sorted = sorted(file_scores, key=lambda x: x[1], reverse=True)
            predicted_relevance = [file_count - manual_ranks[file] for file, _ in file_scores_sorted]
            dcg = calculate_dcg(predicted_relevance)
            idcg = calculate_dcg(sorted(ideal_relevance, reverse=True))
            ndcg = dcg / idcg if idcg > 0 else 0
            group_ndcg_scores[expr].append(ndcg)

    # ===============================
    # 全局 Kendall Tau
    # ===============================
    for expr in axis_expressions:
        all_scores = []
        all_manual_ranks = []
        for group in json_data:
            audio_files = [(item["audio_file"].split("/")[-1], idx) for idx, item in enumerate(group)]
            for file, manual_rank in audio_files:
                all_scores.append(generated_results[file][expr])
                all_manual_ranks.append(manual_rank)
        tau, _ = kendalltau(all_scores, [-rank for rank in all_manual_ranks])
        metrics[expr]["kendall_tau"] = tau

    # ===============================
    # 计算平均 NDCG
    # ===============================
    for expr in axis_expressions:
        metrics[expr]["ndcg"] = np.mean(group_ndcg_scores[expr])

    # ===============================
    # 加载新模型分数并计算相关指标
    # ===============================
    new_model_results = {}
    for csv_path in csv_paths[1:]:  # 从第二个文件开始加载
        new_df = pandas.read_csv(csv_path)
        for index, row in new_df.iterrows():
            filename = row["filename"]
            score = row["score"]
            new_model_results[filename] = score

    # 计算新模型的 Kendall Tau 和 NDCG
    new_model_metrics = {"kendall_tau": 0, "ndcg": 0, "correct_pairs": 0}
    new_model_all_scores = []
    new_model_all_manual_ranks = []
    total_pairs_new_model = 0

    for group in json_data:
        audio_files = [(item["audio_file"].split("/")[-1], idx) for idx, item in enumerate(group)]
        file_count = len(audio_files)
        total_pairs_new_model += file_count * (file_count - 1) // 2
        manual_ranks = {file: rank for file, rank in audio_files}

        for (file1, idx1), (file2, idx2) in itertools.combinations(audio_files, 2):
            score1 = new_model_results.get(file1, 0)
            score2 = new_model_results.get(file2, 0)
            if (score1 > score2 and idx1 < idx2) or (score1 < score2 and idx1 > idx2):
                new_model_metrics["correct_pairs"] += 1

        ideal_relevance = [file_count - rank for _, rank in audio_files]
        predicted_relevance = [new_model_results.get(file, 0) for file, _ in audio_files]

        def calculate_dcg(relevance_scores):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

        dcg = calculate_dcg(predicted_relevance)
        idcg = calculate_dcg(sorted(ideal_relevance, reverse=True))
        ndcg = dcg / idcg if idcg > 0 else 0
        new_model_metrics["ndcg"] += ndcg

        for file, manual_rank in audio_files:
            new_model_all_scores.append(new_model_results.get(file, 0))
            new_model_all_manual_ranks.append(manual_rank)


    # ===============================
    # 随机模型参考（NDCG有效性验证）
    # ===============================
    random_metrics = {"correct_pairs": 0, "kendall_tau": 0, "ndcg": 0}
    random_group_ndcg = []
    random_all_scores = []
    random_all_manual_ranks = []
    total_pairs_random = 0

    for group in json_data:
        audio_files = [(item["audio_file"].split("/")[-1], idx) for idx, item in enumerate(group)]
        file_count = len(audio_files)
        total_pairs_random += file_count * (file_count - 1) // 2
        manual_ranks = {file: rank for file, rank in audio_files}
        shuffled_files = audio_files.copy()
        random.shuffle(shuffled_files)

        for (file1, idx1), (file2, idx2) in itertools.combinations(audio_files, 2):
            pred_idx1 = [f for f, _ in shuffled_files].index(file1)
            pred_idx2 = [f for f, _ in shuffled_files].index(file2)
            if (pred_idx1 < pred_idx2 and idx1 < idx2) or (pred_idx1 > pred_idx2 and idx1 > idx2):
                random_metrics["correct_pairs"] += 1

        ideal_relevance = [file_count - rank for _, rank in audio_files]
        predicted_relevance = [file_count - manual_ranks[file] for file, _ in shuffled_files]

        def calculate_dcg(relevance_scores):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

        dcg = calculate_dcg(predicted_relevance)
        idcg = calculate_dcg(sorted(ideal_relevance, reverse=True))
        ndcg = dcg / idcg if idcg > 0 else 0
        random_group_ndcg.append(ndcg)

        for file, manual_rank in audio_files:
            random_all_scores.append(random.random())
            random_all_manual_ranks.append(manual_rank)

    tau, _ = kendalltau(random_all_scores, [-r for r in random_all_manual_ranks])
    random_metrics["kendall_tau"] = tau
    random_metrics["ndcg"] = np.mean(random_group_ndcg)
    random_correct_ratio = random_metrics["correct_pairs"] / total_pairs_random


    # 计算Kendall Tau
    tau, _ = kendalltau(new_model_all_scores, [-r for r in new_model_all_manual_ranks])
    new_model_metrics["kendall_tau"] = tau
    new_model_metrics["ndcg"] /= len(json_data)


    # ===============================
    # 输出报告
    # ===============================
    best_combo_tau = max(axis_expressions, key=lambda x: metrics[x]["kendall_tau"])
    best_combo_ndcg = max(axis_expressions, key=lambda x: metrics[x]["ndcg"])

    print("\n随机模型参考结果（验证NDCG有效性）")
    print(f"Kendall Tau: {random_metrics['kendall_tau']:.4f}")
    print(f"成对一致率: {random_correct_ratio:.4f}")
    print(f"NDCG:        {random_metrics['ndcg']:.4f}")
    print("="*60)

    # ===============================
    # 输出报告
    # ===============================
    print("增强型排序模型评估报告（含新模型）")
    print("="*60)

    # 输出新模型的评估指标
    print(f"新模型的Kendall Tau: {new_model_metrics['kendall_tau']:.4f}")
    print(f"新模型的成对一致率: {new_model_metrics['correct_pairs']/total_pairs_new_model:.4f}")
    print(f"新模型的NDCG: {new_model_metrics['ndcg']:.4f}")

    # 基于Kendall Tau 和 NDCG 排名前5的组合
    best_combo_tau = max(axis_expressions, key=lambda x: metrics[x]["kendall_tau"])
    best_combo_ndcg = max(axis_expressions, key=lambda x: metrics[x]["ndcg"])

    print(f"\n基于Kendall Tau的最佳组合: {best_combo_tau}")
    print(f"Kendall Tau: {metrics[best_combo_tau]['kendall_tau']:.4f}, 成对一致率: {metrics[best_combo_tau]['correct_pairs']/total_pairs:.4f}, NDCG: {metrics[best_combo_tau]['ndcg']:.4f}")

    print(f"\n基于NDCG的最佳组合: {best_combo_ndcg}")
    print(f"Kendall Tau: {metrics[best_combo_ndcg]['kendall_tau']:.4f}, 成对一致率: {metrics[best_combo_ndcg]['correct_pairs']/total_pairs:.4f}, NDCG: {metrics[best_combo_ndcg]['ndcg']:.4f}")

def test_songeval_sedata():
    pass

if __name__ == "__main__":
    test_songeval_sedata()
    