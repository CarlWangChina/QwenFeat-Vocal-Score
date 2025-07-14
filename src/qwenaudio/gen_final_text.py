import os
import dashscope
import qwenaudio.config
from http import HTTPStatus
from modelscope.pipelines import pipeline

import re
import base64
import json
import uuid
import requests

dashscope.api_key = qwenaudio.config.api_key

def split_text(text):
    # 使用正则表达式匹配数字后跟顿号的模式（中间可能有空格或tab）
    pattern = r'\d+[ \t]*、'
    segments = re.split(pattern, text)
    
    # 提取所有分隔符（数字、顿号及中间的空格/tab）
    separators = re.findall(pattern, text)
    
    # 由于split结果第一个元素是空字符串（因为文本以分隔符开头），所以从第二个元素开始组合
    # 将每个分隔符与对应的分段内容组合起来
    result = []
    for i in range(len(separators)):
        # 分隔符 + 对应的内容（segments[i+1]）
        segment = separators[i] + segments[i+1].strip('\n')
        result.append(segment)
    
    return result

def generate_vocal_critique(input_comments):
    """
    生成声乐评估报告和建议
    
    参数:
    input_scores (dict): 四个维度的分数，格式如 {"音色": 88, "技巧": 85, ...}
    input_comments (dict): 四个维度的评语，格式如 {"音色": "评语", "技巧": "评语", ...}
    api_key (str): DashScope API密钥
    
    返回:
    tuple: (成功状态, 完整报告, 总结评语)
            成功状态: True/False
            完整报告: API返回的完整文本
            总结评语: 提取的总结性评语(用于TTS)
    """
    
    # 构建评语字符串
    critique = ""
    for dim, value in input_comments.items():
        score = value[1]
        comment = value[0]
        critique += f"{dim}分数: {score}, 缺点: {comment}\n"
    
    # 构建Prompt
    prompt_template = f"""
你将扮演一个歌唱比赛的评委,声乐领域的教授,目前一个选手演唱后, 作为评委的你听完已经初步发现了其以下缺点:
“{critique}”

你需要思考后, 严格按照以下格式输出内容:
1、一段将4维度合并后的总结性概括评语,限制在50字以内的中文,不需要包含关于题目的任何信息,例如不需要把"50字以内"这句话本身写到输出中
2、针对4个维度上的优点和缺点展开详述, 你需要基于刚才发现的4个维度下的缺点,分别进行润色,在保持原来含义的同时让语言更加多样性,更符合你作为评委的身份和咖位,输出100-150字的中文.
3、针对目前她/他的歌声中存在的问题, 你有什么他在未来练习和演唱中的建议呢? 请你对症下药,输出200-300字中文的具体的建议,要求言之有物,让对方感到你说的很有道理, 对他提升很有帮助.
"""
    
    # 调用API
    # print("--- 正在调用千问API生成报告 ---")
    response = dashscope.Generation.call(model='qwen-plus', prompt=prompt_template)
    
    if response.status_code == HTTPStatus.OK:
        full_report = response.output['text']
        # print("报告生成成功！")
        
        # 提取总结评语（第一行且去除序号）
        summary_line = split_text(full_report)[0]
        summary_comment = summary_line.replace('1、', '').replace('\n', '').strip()
        
        return True, full_report, summary_comment
    else:
        # print(f"报告生成失败！状态码: {response.status_code}")
        return False, None, None
    
def synthesize_speech(summary_comment):
    """
    合成评委语音
    
    返回: (success, message) 元组
    """
    api_url = f"https://{qwenaudio.config.tts_host}/api/v1/tts"

    header = {"Authorization": f"Bearer;{qwenaudio.config.tts_access_token}"}

    request_json = {
        "app": {
            "appid": qwenaudio.config.tts_appid,
            "token": qwenaudio.config.tts_access_token,
            "cluster": qwenaudio.config.tts_cluster
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio": {
            "voice_type": qwenaudio.config.tts_voice_type,
            "encoding": "mp3",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": summary_comment,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson"

        }
    }
    try:
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        # print(f"resp body: \n{resp.json()}")
        # if "data" in resp.json():
        #     data = resp.json()["data"]
        #     file_to_save = open("test_submit.mp3", "wb")
        #     file_to_save.write(base64.b64decode(data))
        return resp.json()
    except Exception as e:
        # print(f"Error: {e}")
        return str(e)
