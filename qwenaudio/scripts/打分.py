import os
import time
import requests
import json
import pandas as pd
import subprocess

def get_downloaded_files(record_dir):
    """获取已下载的音频文件列表"""
    if not os.path.exists(record_dir):
        return []
    
    # 支持的音频文件扩展名
    audio_extensions = ['.m4a', '.mp3', '.wav', '.aac', '.flac']
    
    downloaded_files = []
    for file in os.listdir(record_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            downloaded_files.append(os.path.join(record_dir, file))
    
    return downloaded_files

def download_audio_files():
    """下载Excel文件中的所有音频文件"""
    # 配置参数
    xlsx_file = '每天top20唱歌最高分.xlsx'
    record_dir = '/home/w-4090/projects/score_test/records/'
    
    # 创建保存目录
    os.makedirs(record_dir, exist_ok=True)
    
    # 读取Excel文件
    df = pd.read_excel(xlsx_file)
    audio_urls = df['录音'].tolist()
    
    # 检查已下载的文件
    existing_files = get_downloaded_files(record_dir)
    print(f"已存在 {len(existing_files)} 个音频文件")
    
    # 批量下载音频文件
    downloaded_files = []
    
    # 如果已下载文件数量与URL数量匹配，直接返回已存在的文件
    if len(existing_files) >= len(audio_urls) and len(audio_urls) > 0:
        print("音频文件已全部下载，跳过下载步骤")
        return existing_files
    
    # 否则重新下载所有文件
    for i, url in enumerate(audio_urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # 提取文件名并保存
            filename = url.split('/')[-1]
            file_path = os.path.join(record_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_files.append(file_path)
            print(f"已下载 {filename} ({i+1}/{len(audio_urls)})")
            
        except requests.exceptions.RequestException as e:
            print(f"下载失败 {url}: {e}")
        except Exception as e:
            print(f"保存失败 {url}: {e}")
    
    # 返回所有已下载的文件
    return get_downloaded_files(record_dir) if not downloaded_files else downloaded_files

def run_infer_script(audio_file_path):
    """使用infer.py脚本进行音频评分"""
    try:
        # 创建临时输出文件
        output_txt = f"{audio_file_path}_score.txt"
        
        # 构建命令
        cmd = [
            'python',
            '/home/w-4090/projects/qwenaudio_score_with_noisy/scripts/infer.py',
            audio_file_path,
            output_txt
        ]
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # 读取输出文件内容
            with open(output_txt, 'r') as f:
                content = f.read()
            return content
        else:
            print(f"infer.py执行失败: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("infer.py执行超时")
        return None
    except Exception as e:
        print(f"执行infer.py时出错: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(output_txt):
            os.remove(output_txt)

def parse_score_result(score_json):
    """解析评分结果，提取8个字段"""
    try:
        data = json.loads(score_json)
        
        # 提取各个维度的得分和分析
        result = {}
        
        # 专业技巧
        result['专业技巧得分'] = data.get('专业技巧', {}).get('score', 0)
        result['专业技巧分析'] = data.get('专业技巧', {}).get('text', '')
        
        # 情感表达
        result['情感表达得分'] = data.get('情感表达', {}).get('score', 0)
        result['情感表达分析'] = data.get('情感表达', {}).get('text', '')
        
        # 音色与音质
        result['音色与音质得分'] = data.get('音色与音质', {}).get('score', 0)
        result['音色与音质分析'] = data.get('音色与音质', {}).get('text', '')
        
        # 气息控制
        result['气息控制得分'] = data.get('气息控制', {}).get('score', 0)
        result['气息控制分析'] = data.get('气息控制', {}).get('text', '')
        
        return result
    except json.JSONDecodeError as e:
        print(f"解析JSON失败: {e}")
        return None
    except Exception as e:
        print(f"解析评分结果时出错: {e}")
        return None
    
def process_all_audio_scores(downloaded_files, original_xlsx):
    """处理所有音频文件的评分并将结果合并到原始Excel中"""
    # 读取原始Excel文件
    original_df = pd.read_excel(original_xlsx)
    
    # 创建一个新的DataFrame来存储评分结果
    score_data = []
    
    # 为每个音频文件获取评分
    for i, file_path in enumerate(downloaded_files):
        filename = os.path.basename(file_path)
        print(f"处理文件 ({i+1}/{len(downloaded_files)}): {filename}")
        
        # 执行评分
        score_result = run_infer_script(file_path)
        if score_result:
            # 解析评分结果
            parsed_result = parse_score_result(score_result)
            if parsed_result:
                parsed_result['文件名'] = filename
                score_data.append(parsed_result)
                print(f"  成功获取评分")
                print(f"  专业技巧得分: {parsed_result['专业技巧得分']}")
            else:
                print(f"  解析评分结果失败")
        else:
            print(f"  获取评分失败")
        
        time.sleep(1.5)  # 避免过快请求导致服务器压力过大
    
    # 创建评分结果DataFrame
    if not score_data:
        print("没有获取到任何评分结果")
        return
    
    score_df = pd.DataFrame(score_data)

    # 在原始数据中添加文件名列用于匹配
    original_df['录音文件名'] = original_df['录音'].apply(lambda x: x.split('/')[-1] if pd.notnull(x) else '')
    
    # 将评分结果与原始数据合并
    merged_df = pd.merge(original_df, score_df, left_on='录音文件名', right_on='文件名', how='left')
    
    # 保存到新的Excel文件
    output_filename = '每天top20唱歌最高分_含评分.xlsx'
    merged_df.to_excel(output_filename, index=False)
    print(f"结果已保存到: {output_filename}")
    return merged_df

def main():
    """主函数"""
    # 1. 下载音频文件（带检测功能）
    print("开始下载音频文件...")
    downloaded_files = download_audio_files()
    
    if not downloaded_files:
        print("没有找到任何音频文件")
        return
    
    # 2. 处理所有音频文件的评分并生成结果
    print(f"\n开始处理 {len(downloaded_files)} 个音频文件的评分...")
    result_df = process_all_audio_scores(downloaded_files, '每天top20唱歌最高分.xlsx')
    

# 执行主函数
if __name__ == "__main__":
    main()