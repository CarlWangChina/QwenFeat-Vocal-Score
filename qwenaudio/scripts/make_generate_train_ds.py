import os
import sys
import json
import pandas as pd
import soundfile as sf
import numpy as np
import pyloudnorm as pyln

if __name__ == "__main__":
    
    df = pd.read_excel("/home/w-4090/projects/qwenaudio/data/专业打分结果提取汇总核验 魅ktv.xlsx",sheet_name=1)  
    """
    音频网页链接        https://v.meiktv.com/admin/service/wsl/files/r...
    评级打分者姓名/昵称                                                   阿乐
    “音色”维度打分                                                    2.0
    “音色”维度评语                                           缺乏共鸣或层次,声音单一
    “情感”维度打分                                                    2.0
    “情感”维度评语                                             缺乏情感波动或表现力
    “技巧”维度打分                                                    2.0
    “技巧”维度评语                                   肌肉僵硬,喉咙用力,声音失去弹性和流畅性
    “气息控制”维度打分                                                  3.0
    “气息控制”维度评语                                         尾音发虚,呼吸声过于明显
    备注                                                          NaN
    辅助标记                                                        NaN
    """    
    res = dict()
    res2 = dict()
    for index, row in df.iterrows():

        # print(index, row["“音色”维度打分"])
        # 从表格提取4个分数和评语
        # print(row["辅助标记"])
        # print(row["record_id"])
        if row["辅助标记"]!="*" and row["评级打分者姓名/昵称"]=="阿乐":
            score0 = int(row["“技巧”维度打分"])
            score1 = int(row["“情感”维度打分"])
            score2 = int(row["“音色”维度打分"])
            score3 = int(row["“气息控制”维度打分"])
            comment0 = row["“技巧”维度评语"]
            comment1 = row["“情感”维度评语"]
            comment2 = row["“音色”维度评语"]
            comment3 = row["“气息控制”维度评语"]
            # print(score0, score1, score2, score3, comment0, comment1, comment2, comment3)
            prompt = [
                f"{score0}分：{comment0}",
                f"{score1}分：{comment1}",
                f"{score2}分：{comment2}",
                f"{score3}分：{comment3}",
            ]
            res[row["record_id"]] = prompt
            res2[row["record_id"]] = [
                [score0, comment0],
                [score1, comment1],
                [score2, comment2],
                [score3, comment3],
            ]

    # with open("/home/w-4090/projects/qwenaudio/data/gen.json", "w") as f:
    #     json.dump(res, f, ensure_ascii=False, indent=4)

    with open("/home/w-4090/projects/qwenaudio/data/gen_al.json", "w") as f:
        json.dump(res2, f, ensure_ascii=False, indent=4)
