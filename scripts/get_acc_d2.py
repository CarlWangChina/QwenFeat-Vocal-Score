import json
import csv
from io import StringIO

def json_to_csv(json_data):
    # 解析JSON数据
    data = json.loads(json_data)
    results = data["results"]
    
    # 创建CSV输出缓冲区
    output = StringIO()
    writer = csv.writer(output)
    
    # 写入CSV表头
    headers = [
        "audio_path", "gen_method", "generated_score", "generated_text",
        "target_score", "target_comment", "step_time", "score_dist"
    ]
    writer.writerow(headers)
    
    # 处理每条记录
    for item in results:
        row = [
            item["audio_path"],
            item["gen_method"],
            item["generate"]["score"],
            item["generate"]["text"].strip().replace("\n", " "),  # 清理换行符
            item["target_score"],
            item["target_comment"].strip().replace("\n", " "),    # 清理换行符
            item["step_time"],
            item["generate"]["score_dist"]
        ]
        writer.writerow(row)
    
    # 获取CSV内容
    output.seek(0)
    return output.read()

# 示例使用（实际使用时从文件读取JSON）
if __name__ == "__main__":
    # 假设这是从文件读取的JSON数据
    with open("/home/w-4090/projects/qwenaudio/results/test_score_by_text_无分类模型_无参考评语.json", "r", encoding="utf-8") as f:
        json_input = f.read()
    
    csv_output = json_to_csv(json_input)
    
    # 将结果写入CSV文件
    with open("output.csv", "w", encoding="utf-8", newline="") as f:
        f.write(csv_output)


