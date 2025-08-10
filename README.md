# qwenaudio打分工具  
## 调用方法

### 启用服务  

`python scripts/infer_service.py`  
调用：  
`curl -X POST http://localhost:8080/score?选项   -F "file=@音频路径"`  

#### 参数说明  
post传入文件内容，get传入选项  

|参数名 |	类型 |	默认值 |	用途说明|
| ---- | ---- | ---- | ---- |
|get_final_text |	bool |	false |	是否生成总结性文本|
|render_final_text |	bool |	false |	是否渲染对总结性文本为音频|
|process_steps |	str |	"0123" |	指定需要执行的处理步骤（0123分别对应 专业技巧 情感表达 音色与音质 气息控制）|
|singer_id |	str |	"0" |	渲染和生成总结性文本时使用的歌手id|
|speed_ratio |	float |	1.3 |	语速调节系数|
|return_single_score |	bool |	true |	是否返回单个评分项|
|return_sum_score |	bool |	true |	是否返回评分总分|


### 命令行调用
`python scripts/infer.py 音频路径.wav 输出文件.txt`

如果在加载模型时卡住，可启用huggingface镜像
`export HF_ENDPOINT=https://hf-mirror.com`
