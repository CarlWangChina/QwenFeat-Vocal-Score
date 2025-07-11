import dashscope
from http import HTTPStatus

def generate_vocal_critique(input_comments, api_key):
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
    # 设置API密钥
    dashscope.api_key = api_key
    
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
1、一段将4维度合并后的总结性概括评语,限制在15-20字的中文.
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
        summary_line = full_report.split('\n')[0]
        summary_comment = summary_line.replace('1、', '').strip()
        
        return True, full_report, summary_comment
    else:
        # print(f"报告生成失败！状态码: {response.status_code}")
        return False, None, None
