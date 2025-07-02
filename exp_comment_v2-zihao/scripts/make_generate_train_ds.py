import os
import sys
import json
import pandas as pd

if __name__ == "__main__":
    # 读取原始Excel
    df = pd.read_excel("/home/w-4090/projects/qwenaudio/exp_comment_v2-zihao/data/专业打分结果提取汇总核验 魅ktv.xlsx", sheet_name=1)

    res = dict()
    jsonl_lines = []
    for index, row in df.iterrows():
        if row["辅助标记"] != "*":
            record_id = str(row["record_id"])
            comment0 = str(row["“技巧”维度评语"])
            comment1 = str(row["“情感”维度评语"])
            comment2 = str(row["“音色”维度评语"])
            comment3 = str(row["“气息控制”维度评语"])
            # 拼接为一条长评语
            long_comment = f"专业技巧上:{comment0} 情感表达上:{comment1} 音色与音质上:{comment2} 气息控制上:{comment3}"
            res[record_id] = long_comment
            # 假设音频路径为 data/audio_sep_seg/{record_id}.wav
            audio_path = f"/home/w-4090/projects/qwenaudio/data/audio_sep_seg/{record_id}.wav"
            prompt = "请对这段音频的专业技巧、情感表达、气息控制、音色与音质，这4个方面进行写评语，不少于150个汉字。"
            jsonl_lines.append(json.dumps({"audio": audio_path, "text": prompt, "label": long_comment}, ensure_ascii=False))

    # 保存为新训练数据
    with open("/home/w-4090/projects/qwenaudio/exp_comment_v2-zihao/data/gen_long_comment.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    # 保存为jsonl格式
    with open("/home/w-4090/projects/qwenaudio/exp_comment_v2-zihao/data/gen_long_comment.jsonl", "w") as f:
        for line in jsonl_lines:
            f.write(line + "\n")
