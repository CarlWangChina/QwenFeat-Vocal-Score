import pandas as pd
import numpy as np
import os
import json
import glob
from scipy.stats import pearsonr

def debug_audio_ids(csv_file, axis_dirs):
    """调试函数：查看CSV和JSON中的音频ID格式"""
    print("调试音频ID匹配...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 将音频ID转换为字符串
    df['audio_id_str'] = df['audio_id'].astype(str)
    print(f"CSV中的音频ID样本 (前10个): {df['audio_id_str'].head(10).tolist()}")
    print(f"CSV总行数: {len(df)}")
    
    # 检查每个axis的音频ID
    for axis in range(4):
        axis_dir = axis_dirs[axis]
        json_files = glob.glob(os.path.join(axis_dir, "**", "train_score.json"), recursive=True)
        
        axis_audio_ids = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    audio_id = item["audio"].split("/")[-1].split(".")[0]
                    axis_audio_ids.append(audio_id)
        
        print(f"Axis {axis} 音频ID样本 (前10个): {axis_audio_ids[:10]}")
        print(f"Axis {axis} 总音频数: {len(axis_audio_ids)}")
        
        # 检查交集
        common_ids = set(df['audio_id_str']).intersection(set(axis_audio_ids))
        print(f"与CSV共有的音频ID数量: {len(common_ids)}")
        if common_ids:
            print(f"共有的音频ID样本: {list(common_ids)[:5]}")
    
    return df

def load_axis_scores(axis_dirs):
    """加载4个axis的评分数据"""
    axis_scores = {}
    
    for axis in range(4):
        axis_dir = axis_dirs[axis]
        scores = {}
        
        # 查找该axis目录下的所有train_score.json文件
        json_files = glob.glob(os.path.join(axis_dir, "**", "train_score.json"), recursive=True)
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # 从路径中提取文件名（不含扩展名）
                    audio_path = item["audio"]
                    audio_id = audio_path.split("/")[-1].split(".")[0]
                    
                    # 存储评分
                    scores[audio_id] = item["score"]
        
        axis_scores[f"axis_{axis}"] = scores
    
    return axis_scores

def calculate_correlations(csv_file, axis_scores):
    """计算CSV文件中各维度与axis评分的相关系数"""
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 将音频ID转换为字符串
    df['audio_id_str'] = df['audio_id'].astype(str)
    
    # 提取用户完整音频评分列
    user_columns = [col for col in df.columns if col.startswith('user_full_')]
    dimensions = [col.replace('user_full_', '') for col in user_columns]
    
    print(f"找到的评分维度: {dimensions}")
    
    # 为每个音频添加axis评分
    matched_count = 0
    for axis_name, scores_dict in axis_scores.items():
        df[axis_name] = df['audio_id_str'].apply(lambda x: scores_dict.get(x, None))
        matched = df[axis_name].notna().sum()
        matched_count += matched
        print(f"{axis_name} 匹配的音频数量: {matched}")
    
    # 删除没有axis评分的行
    df_clean = df.dropna(subset=[f'axis_{i}' for i in range(4)])
    
    if len(df_clean) == 0:
        print("错误：没有找到匹配的音频ID")
        print("请检查音频ID的格式是否一致")
        return None
    
    print(f"有效数据量: {len(df_clean)} 个音频")
    
    # 计算相关系数
    correlation_results = {}
    
    for dimension in dimensions:
        user_col = f'user_full_{dimension}'
        dim_correlations = {}
        
        for axis in range(4):
            axis_col = f'axis_{axis}'
            
            # 计算皮尔逊相关系数
            corr, p_value = pearsonr(df_clean[user_col], df_clean[axis_col])
            dim_correlations[f'axis_{axis}'] = {
                'correlation': corr,
                'p_value': p_value,
                'strength': abs(corr),
                'direction': '正相关' if corr > 0 else '负相关'
            }
        
        correlation_results[dimension] = dim_correlations
    
    return correlation_results, df_clean

def analyze_correlations(correlation_results):
    """分析相关性结果并得出结论"""
    print("\n" + "="*80)
    print("相关性分析结果")
    print("="*80)
    
    # 为每个维度找到最强的相关性
    strongest_correlations = {}
    
    for dimension, axes in correlation_results.items():
        print(f"\n{dimension}维度:")
        print("-" * 40)
        
        strongest_axis = None
        strongest_corr = 0
        
        for axis_name, corr_info in axes.items():
            corr = corr_info['correlation']
            p_val = corr_info['p_value']
            direction = corr_info['direction']
            
            # 标注显著性
            significance = ""
            if p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
            
            print(f"  {axis_name}: 相关系数 = {corr:.4f}{significance}, p值 = {p_val:.4f}, {direction}")
            
            # 更新最强相关性
            if abs(corr) > abs(strongest_corr):
                strongest_corr = corr
                strongest_axis = axis_name
        
        strongest_correlations[dimension] = {
            'strongest_axis': strongest_axis,
            'correlation': strongest_corr,
            'direction': '正相关' if strongest_corr > 0 else '负相关',
            'strength': abs(strongest_corr)
        }
    
    # 输出结论
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    
    for dimension, info in strongest_correlations.items():
        print(f"{dimension}维度:")
        print(f"  与{info['strongest_axis']}的相关性最强")
        print(f"  相关系数: {info['correlation']:.4f} ({info['direction']})")
        print(f"  相关性强度: {get_strength_description(info['strength'])}")
        print()
    
    # 总体分析
    print("总体分析:")
    axis_counts = {}
    for dimension, info in strongest_correlations.items():
        axis = info['strongest_axis']
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
    
    for axis, count in axis_counts.items():
        print(f"  {axis}与{count}个维度相关性最强")
    
    return strongest_correlations

def get_strength_description(strength):
    """根据相关系数绝对值描述相关性强度"""
    if strength >= 0.8:
        return "极强相关"
    elif strength >= 0.6:
        return "强相关"
    elif strength >= 0.4:
        return "中等相关"
    elif strength >= 0.2:
        return "弱相关"
    else:
        return "极弱相关"

def main():
    # 配置路径
    axis_dirs = [
        "data/train_ds_4_al/denoise/0",  # axis 0目录
        "data/train_ds_4_al/denoise/1",  # axis 1目录
        "data/train_ds_4_al/denoise/2",  # axis 2目录
        "data/train_ds_4_al/denoise/3"   # axis 3目录
    ]
    
    # 假设CSV文件名为 audio_scores_0.csv，可以根据需要修改
    csv_file = "data/audio_scores_0.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误：CSV文件 {csv_file} 不存在")
        return
    
    for axis_dir in axis_dirs:
        if not os.path.exists(axis_dir):
            print(f"错误：axis目录 {axis_dir} 不存在")
            return
    
    # 调试音频ID匹配
    df = debug_audio_ids(csv_file, axis_dirs)
    
    # 加载axis评分
    print("\n正在加载axis评分数据...")
    axis_scores = load_axis_scores(axis_dirs)
    
    # 检查每个axis的数据量
    for axis_name, scores in axis_scores.items():
        print(f"{axis_name}: {len(scores)} 个评分")
    
    # 计算相关系数
    print("\n正在计算相关系数...")
    results = calculate_correlations(csv_file, axis_scores)
    
    if results is None:
        print("\n无法计算相关系数，请检查音频ID格式")
        return
    
    correlation_results, df_clean = results
    
    # 分析结果
    strongest_correlations = analyze_correlations(correlation_results)
    
    # 可选：保存详细结果到文件
    output_file = "data/correlation_analysis_results.csv"
    
    # 创建详细结果DataFrame
    detailed_results = []
    for dimension, axes in correlation_results.items():
        for axis_name, corr_info in axes.items():
            detailed_results.append({
                'dimension': dimension,
                'axis': axis_name,
                'correlation': corr_info['correlation'],
                'p_value': corr_info['p_value'],
                'direction': corr_info['direction']
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()