import os
import sys
import json
import pandas as pd
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_PATH = os.path.join(ROOT_PATH, "data", "score")
sys.path.append(os.path.join(ROOT_PATH, "src"))

from qwenaudio.audio_cut import process_audio

class ScoreSet:
    def __init__(self):
        self.scores = dict()
    def load_xls(self, file_path):
        try:  
            # 检查文件是否存在  
            if not os.path.exists(file_path):  
                raise FileNotFoundError(f"文件 {file_path} 不存在")  
            
            songid = file_path.split("/")[-1].split("_")[0]
            if songid not in self.scores:
                self.scores[songid] = dict()
            
            # 读取XLS文件  
            df = pd.read_excel(file_path)  
            
            # 显示前几行数据  
            # print("使用pandas解析结果：")  
            # print(df.head())  
            # 遍历
            for index, row in df.iterrows():
                seg_id = row.iloc[0]
                user = row.iloc[2]
                score1 = row.iloc[3]
                score2 = row.iloc[4]
                if seg_id not in self.scores[songid]:
                    audio_path = f"/home/w-4090/cutted_score_audio/{songid}/{seg_id}.wav"
                    assert os.path.exists(audio_path), f"文件 {audio_path} 不存在"
                    audio_seg_path = os.path.abspath(os.path.join(ROOT_PATH, "data", "audio_sep_seg", str(songid), str(seg_id)))
                    self.scores[songid][seg_id] = {"score":dict(), "average":0, "audio_path":audio_path, "seg_files":process_audio(audio_path, audio_seg_path)}
                    
                self.scores[songid][seg_id]["score"][user] = {"score1":score1, "score2":score2}
            
        
        except Exception as e:  
            print(f"解析XLS文件时发生错误：{e}")  
            return None  
    
    def make_average(self):
        for songid in self.scores:
            for seg_id in self.scores[songid]:
                scores = self.scores[songid][seg_id]["score"]
                score1_average = sum([score["score1"] for score in scores.values()]) / len(scores)
                score2_average = sum([score["score2"] for score in scores.values()]) / len(scores)
                self.scores[songid][seg_id]["average"] = {"score1":score1_average, "score2":score2_average}


if __name__ == "__main__":
    sc = ScoreSet()
    # walk SCORE_PATH
    print("walking score path:", SCORE_PATH)
    for root, dirs, files in os.walk(SCORE_PATH):
        for file in files:
            if file.endswith(".xlsx"):
                path = os.path.join(root, file)
                print(path)
                sc.load_xls(path)
    sc.make_average()
    with open(os.path.join(ROOT_PATH, "data", "scores.json"), "w") as f:
        json.dump(sc.scores, f, ensure_ascii=False, indent=4)