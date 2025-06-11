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

def calculate_volume(frame):
    """计算音频帧的音量（RMS）"""
    return np.sqrt(np.mean(np.square(np.abs(frame))))

def process_audio(input_path, output_dir, threshold=0.01, frame_size=1024, segment_duration=10):
    """
    处理音频文件，检测高音量片段并截取
    
    参数:
        input_path: 输入音频路径
        output_dir: 输出目录
        threshold: 音量阈值 (0-1范围)
        frame_size: 检测帧大小（采样点数）
        segment_duration: 截取片段时长（秒）
    """
    # 读取音频文件
    data, sr = sf.read(input_path, dtype='float32')
    num_channels = data.shape[1] if len(data.shape) > 1 else 1

    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -12 dB LUFS
    data = pyln.normalize.loudness(data, loudness, -12.0)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    current_pos = 0
    segment_count = 0
    total_samples = len(data)
    segment_samples = int(segment_duration * sr)
    
    print(f"处理开始: {input_path}")
    print(f"采样率: {sr}Hz, 总时长: {total_samples/sr:.1f}秒")
    print(f"使用阈值: {threshold:.3f}, 帧大小: {frame_size}采样点")

    out_files = []

    while current_pos + frame_size <= total_samples:
        # 提取当前帧
        start = current_pos
        end = start + frame_size
        frame = data[start:end]
        
        # 计算音量（多声道取平均）
        if num_channels > 1:
            channel_volumes = [calculate_volume(frame[:, c]) for c in range(num_channels)]
            volume = np.mean(channel_volumes)
        else:
            volume = calculate_volume(frame)

        # 检测阈值
        if volume > threshold:
            # 计算截取结束位置
            segment_end = min(start + segment_samples, total_samples)
            
            # 写入片段文件
            output_path = os.path.join(
                output_dir,
                f"segment_{segment_count:03d}_start{start/sr:.1f}s.wav"
            )
            sf.write(output_path, data[start:segment_end], sr)

            out_files.append(os.path.abspath(output_path))
            
            print(f"检测到高音量片段: {output_path} (时长: {segment_end/sr - start/sr:.1f}秒)")
            
            # 更新指针到片段末尾
            current_pos = segment_end
            segment_count += 1
        else:
            # 移动到下一帧
            current_pos += frame_size

    print(f"处理完成，共找到{segment_count}个有效片段")
    return out_files

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
                    audio_path = f"/home/w-4090/cutted_score_audio_separated/{songid}/{seg_id}.wav"
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