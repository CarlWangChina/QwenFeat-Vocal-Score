# Singing-Aesthetic-Assessment

Assessing the Popularity of Singing Timbre with a Multimodal Large Foundation Model

[qwenaudio](./qwenaudio/README.md) Qwen评语生成+打分部分. 包含输入音频给Qwen-audio我们用Lora训练后的版本, 输出针对歌声的存在问题的评语, 然后评语再作为输入作为深度思考部分, 最终进行音色打分, 然后用歌手音色的TTS念出来, 这一整套的流程.  
这一块的所有权重都保留了, 模型对应目录链接: 前往huggingface下载包含模型的完整仓库： [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  


[audioscore](./audioscore/README.md) MuQ打分、排序部分. 包含 加解耦 和 不加解耦 两个版本, 使用的同一套代码，只分了一个目录. 
1、使用MuQ作为encoder+后接打分器进行打分的代码,不加解耦,这一块的架构跟SongEval工作基本相同.  其中MuQ解冻用lora的权重、后面打分器的权重, 都保留了,权重对应目录链接:  [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  
2、加解耦 , 采用SaMoye-SVC的 spk encoder作为反向梯度训练、对说话人身份特征进行解耦, 来提升对美学理解的准确度的实验部分. 
二者项目代码是同一个、兼容的. 加解耦部分因为机房退租时没来得及拷贝下来, 导致模型权重丢失了. 但是使用SaMoye的spk encoder或者用wespeaker反向梯度来解耦合, 看看效果是否会变好的实验,实验结果大致如下: 使用SaMoye的spk encoder或者 wespeaker的encoder, 分别作为反向梯度来解耦合, 然后打分, 都特别难训练、难收敛. 但是batchsize小一点也能收敛. 最后评价美学等级的准确率, 效果比原来好一点点. 说明这个解耦部分确实是有用的. 

最后注意: 本仓库不包含模型部分, 研究者需要前往huggingface下载包含模型的完整仓库： [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  
