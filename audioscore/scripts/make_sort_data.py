import os
import sys
import torch
import pickle
import json
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty
import pandas as pd
import pyworld as pw
import subprocess
import random
import audioread
import librosa

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = []

# walk ROOT_DIR/data/sort/ori_xls
for root, dirs, files in os.walk(os.path.join(ROOT_DIR, "data", "sort", "ori_xls")):
    for file in files:
        if file.endswith(".xlsx"):
            path = os.path.join(root, file)
            print(f"\n处理文件: {path}")
            
            sheets = pd.ExcelFile(path).sheet_names
            # 必需的基础字段
            base_required_columns = ['歌曲名称', '歌手名称']
            # 灵活的字段映射
            flexible_columns = {
                'score': ['标记分数', '分数标记', '分数', '标记粉丝'],  # 分数相关字段
                'url': ['录音', '录音链接']  # 录音相关字段
            }
            
            # 检查每个工作表是否包含所需字段
            matched_sheets = []  
            for sheet_name in sheets:  
                try:  
                    df = pd.read_excel(path, sheet_name=sheet_name)  
                    
                    # 检查基础必需字段
                    base_columns_ok = all(col in df.columns for col in base_required_columns)
                    
                    # 检查灵活字段 - 每个类别至少有一个字段存在
                    flexible_columns_ok = True
                    actual_columns_mapping = {}
                    
                    for target_col, possible_cols in flexible_columns.items():
                        found = False
                        for col in possible_cols:
                            if col in df.columns:
                                actual_columns_mapping[target_col] = col
                                found = True
                                break
                        if not found:
                            flexible_columns_ok = False
                            break
                    
                    if base_columns_ok and flexible_columns_ok:
                        matched_sheets.append((sheet_name, actual_columns_mapping))
                        
                except Exception as e:  
                    print(f"读取工作表 {sheet_name} 时出错: {str(e)}")  
            
            # 处理匹配的工作表
            if matched_sheets:  
                print("包含所有目标字段的工作表：", [sheet[0] for sheet in matched_sheets])
                
                # 处理每个匹配的工作表
                for sheet_name, column_mapping in matched_sheets:
                    try:
                        # 读取工作表数据
                        df = pd.read_excel(path, sheet_name=sheet_name)
                        
                        # 提取需要的列：基础列 + 映射的分数和URL列
                        columns_to_extract = base_required_columns + [
                            column_mapping['score'], 
                            column_mapping['url']
                        ]
                        df_filtered = df[columns_to_extract].copy()
                        
                        # 清理数据：去除空行
                        df_cleaned = df_filtered.dropna(how='all')
                        
                        # 转换为字典列表格式，并重命名字段
                        sheet_data = []
                        for _, row in df_cleaned.iterrows():
                            file_name = row[column_mapping['url']].split('/')[-1]
                            # 下载文件
                            local_path = os.path.join(ROOT_DIR, "data", "sort", "data", "audio", file_name)
                            muq_path = os.path.join(ROOT_DIR, "data", "sort", "data", "muq", file_name+".pt")
                            # assert os.path.exists(muq_path), f"muq文件不存在: {muq_path}"
                            if not os.path.exists(local_path):
                                print(f"下载文件: {row[column_mapping['url']]}")
                                subprocess.run(["wget", "-O", local_path, row[column_mapping['url']]], check=True)
                            try:
                                target_sr=24000
                                with audioread.audio_open(local_path) as input_file:
                                    audio, original_sr = librosa.load(input_file, sr=target_sr, mono=True)
                                record = {
                                    'score': str(row[column_mapping['score']]),
                                    'url': row[column_mapping['url']],
                                    "audio_file": local_path,
                                    'source_file': file,
                                    "muq": muq_path,
                                    'sheet_name': sheet_name,
                                    # 可选：保留其他字段用于调试
                                    'song_name': row['歌曲名称'] if '歌曲名称' in row else '',
                                    'singer': row['歌手名称'] if '歌手名称' in row else ''
                                }
                                sheet_data.append(record)
                            except Exception as err:
                                print(f"读取{local_path}失败")
                                print(err)
                        
                        # 将数据添加到总数据中
                        data.append(sheet_data)
                        
                        print(f"  工作表 '{sheet_name}' 提取了 {len(sheet_data)} 条记录")
                        print(f"    使用的字段映射: {column_mapping}")
                        
                    except Exception as e:
                        print(f"处理工作表 {sheet_name} 时出错: {str(e)}")
                
            else:  
                print("未找到包含所有目标字段的工作表")
                # 显示每个工作表存在的字段
                print("各工作表字段情况:")
                for sheet_name in sheets:
                    try:
                        df = pd.read_excel(path, sheet_name=sheet_name, nrows=0)  # 只读取列名
                        existing_columns = list(df.columns)
                        
                        # 检查基础字段缺失情况
                        missing_base_columns = [col for col in base_required_columns if col not in existing_columns]
                        
                        # 检查灵活字段缺失情况
                        missing_flexible_info = {}
                        for target_col, possible_cols in flexible_columns.items():
                            found = any(col in existing_columns for col in possible_cols)
                            if not found:
                                missing_flexible_info[target_col] = possible_cols
                        
                        print(f"  '{sheet_name}':")
                        print(f"    存在的字段: {existing_columns}")
                        if missing_base_columns:
                            print(f"    缺失的基础字段: {missing_base_columns}")
                        if missing_flexible_info:
                            print(f"    缺失的灵活字段: {missing_flexible_info}")
                            
                    except Exception as e:
                        print(f"  读取工作表 {sheet_name} 的字段时出错: {str(e)}")

# 打印总数据统计
print(f"\n总共提取了 {len(data)} 条记录")

with open(os.path.join(ROOT_DIR, "data", "sort", "data", "index", "index.json"), "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

random.seed(42)
random.shuffle(data)

train_data = data[2:]
val_data = data[:2]

with open(os.path.join(ROOT_DIR, "data", "sort", "data", "index", "train.json"), "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(os.path.join(ROOT_DIR, "data", "sort", "data", "index", "val.json"), "w") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

with open(os.path.join(ROOT_DIR, "data", "sort", "data", "index", "index.json")) as f:
    json.load(f)