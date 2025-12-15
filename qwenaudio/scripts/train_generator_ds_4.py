import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))
# print(sys.path)
import torch
from qwenaudio.trainer_generator import QwenAudioTrainer
from torch import distributed as dist
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioProcessor
import torch.distributed as dist
import argparse
from peft import LoraConfig, get_peft_model, LoftQConfig


if __name__ == "__main__":
    # 如果是分布式训练，初始化进程组
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print("rank:", rank, "world_size:", world_size)
    device = f"cuda:{rank}"

    # 初始化模型和处理器
    processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", sample_rate=16000)
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        # torch_dtype=torch.bfloat16,
        # load_in_4bit=True
        # load_in_8bit=True
    )
    model.half()
    
    # 应用LoRA微调
    # loftq_config = LoftQConfig(loftq_bits=4)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        inference_mode=False,
        # loftq_config=loftq_config,
        # init_lora_weights="loftq"
    )
    model = get_peft_model(model, peft_config)
    
    # if rank == 0:
    #     model.print_trainable_parameters()
    
    # 启用梯度检查点以节省显存
    # lora时不能启用，否则无法求导
    # model.gradient_checkpointing_enable()
    
    # 将模型移至设备
    model.to(device)

    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):  # 仅检查全连接层
    #         for param_name, param in module.named_parameters():
    #             if param.requires_grad:
    #                 if name!="base_model.model.audio_tower.layers.0.self_attn.k_proj.lora_A.default":
    #                     param.requires_grad = False
    if dist.get_rank() == 0:
        print(model)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):  # 仅检查全连接层
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        print(f"模块 {name} 的参数 {param_name} 未冻结")
    
    parser = argparse.ArgumentParser(description="Qwen2-Audio分布式训练脚本")
    parser.add_argument("--train_set", type=str, required=True, help="训练数据集路径")
    parser.add_argument("--val_set", type=str, required=True, help="验证数据集路径")
    parser.add_argument("--save_dir", type=str, required=True, help="模型保存目录")
    args = parser.parse_args()


    # 创建训练器
    trainer = QwenAudioTrainer(
        model=model,
        processor=processor,
        train_set=(args.train_set),
        val_set=(args.val_set),
        device=device,
        local_rank=rank,
        world_size=world_size
    )
    
    # 配置训练参数
    trainer.batch_size = 1
    trainer.lr = 3e-5
    trainer.epochs = 40
    trainer.save_dir = args.save_dir
    
    # 开始训练
    trainer.train()
