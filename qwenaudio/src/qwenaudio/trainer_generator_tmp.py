import json
import os
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, set_seed
import numpy as np
import qwenaudio.prompts
import qwenaudio.audio_cut
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datetime import datetime
import time
import evaluate

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, gen_json_path, score_json_path, processor, max_length=30, is_distributed=False, mode='train'):
        """
        音频数据集加载器 - 支持分布式训练和测试集
        Args:
            gen_json_path: 生成文本JSON路径
            score_json_path: 评分JSON路径
            processor: Qwen2音频处理器
            max_length: 最大音频长度(秒)
            is_distributed: 是否在分布式环境中运行
            mode: 'train' 或 'test'
        """
        self.processor = processor
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.max_length = max_length * self.sampling_rate  # 转换为采样点数
        self.is_distributed = is_distributed
        self.mode = mode
        
        # 只在主进程显示加载信息
        if not self.is_distributed or (self.is_distributed and dist.get_rank() == 0):
            print(f"[{mode.capitalize()}] Loading dataset from {gen_json_path} and {score_json_path}")
            print(f"Max audio length: {max_length}s ({self.max_length} samples)")
            print("Sampling rate:", self.sampling_rate)

        self.data = []
        
        with open(gen_json_path, 'r') as f:
            self.gen_json = json.load(f)
        
        with open(score_json_path, 'r') as f:
            self.score_json = json.load(f)
        
        for song_id, song_data in self.score_json.items():
            for audio_id, audio_body in song_data.items():
                audio_path = audio_body["audio_path"]

                if audio_id in self.gen_json:
                    gen_text_list = self.gen_json[audio_id]

                    # 使用预定义的提示列表
                    for prompt_idx, (prompt, gen_text) in enumerate(zip(qwenaudio.prompts.prompts, gen_text_list)):
                        # 测试集只使用第一个提示，减少测试时间
                        if mode == 'test' and prompt_idx > 0:
                            continue
                            
                        conversation = [
                            {"role": "user", "content": [
                                {"type": "audio", "audio_url": "input.wav"},
                                {"type": "text", "text": prompt},
                            ]},
                            {"role": "assistant", "content": [
                                {"type": "text", "text": gen_text}
                            ]}
                        ]

                        conversation_text = self.processor.apply_chat_template(
                            conversation, 
                            add_generation_prompt=True, 
                            tokenize=False
                        )
                        
                        self.data.append({
                            "audio": audio_path,
                            "text": conversation_text,
                            "prompt": prompt,
                            "target": gen_text
                        })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio"]
        text_prompt = item["text"]
        
        try:
            # 加载音频
            audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
            
            # 训练集随机裁剪，测试集使用完整音频或中心裁剪
            if self.mode == 'train':
                # 随机裁剪
                if len(audio) > self.max_length:
                    start = np.random.randint(0, len(audio) - self.max_length)
                    audio = audio[start:start+self.max_length]
                elif len(audio) < self.max_length:
                    audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
            else:
                # 测试集：使用完整音频或中心裁剪
                if len(audio) > self.max_length:
                    # 中心裁剪
                    start = (len(audio) - self.max_length) // 2
                    audio = audio[start:start+self.max_length]
                elif len(audio) < self.max_length:
                    audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
            
            return {
                "raw_audio": audio,
                "text": text_prompt,
                "audio_path": audio_path,
                "target": item["target"]
            }
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            # 返回空数据，在collate_fn中过滤
            return {
                "raw_audio": np.zeros(self.max_length),
                "text": "",
                "audio_path": "",
                "target": ""
            }


def setup_distributed():
    """初始化分布式训练环境"""
    if "LOCAL_RANK" in os.environ:
        # 使用torchrun启动
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return local_rank, world_size, device
    else:
        # 单GPU训练
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, processor, eval_loader, device, local_rank, metric):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in eval_loader:
            # 过滤无效数据
            valid_indices = [i for i, path in enumerate(batch["audio_path"]) if path]
            if not valid_indices:
                continue
                
            # 只处理有效数据
            batch = {k: [v[i] for i in valid_indices] for k, v in batch.items()}
            
            # 准备输入
            inputs = processor(
                text=batch["text"],
                audios=batch["raw_audio"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                sample_rate=processor.feature_extractor.sampling_rate
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 计算损失
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * len(valid_indices)
            
            # 生成文本
            generate_ids = model.generate(
                input_ids=inputs["input_ids"],
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                feature_attention_mask=inputs["feature_attention_mask"],
                max_new_tokens=128,
                do_sample=False
            )
            
            # 解码预测文本
            predictions = processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(batch["target"])
    
    # 分布式环境下收集所有结果
    if dist.is_initialized():
        # 收集所有进程的预测和目标
        all_predictions_gathered = [None] * dist.get_world_size()
        all_targets_gathered = [None] * dist.get_world_size()
        
        dist.all_gather_object(all_predictions_gathered, all_predictions)
        dist.all_gather_object(all_targets_gathered, all_targets)
        
        # 主进程合并结果
        if local_rank == 0:
            all_predictions = [p for sublist in all_predictions_gathered for p in sublist]
            all_targets = [t for sublist in all_targets_gathered for t in sublist]
    
    # 只在主进程计算指标
    eval_metrics = {}
    if local_rank == 0 or not dist.is_initialized():
        # 计算平均损失
        eval_metrics["loss"] = total_loss / len(eval_loader.dataset)
        
        # 计算文本相似度指标
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        
        rouge_scores = rouge.compute(
            predictions=all_predictions, 
            references=all_targets
        )
        bleu_scores = bleu.compute(
            predictions=all_predictions, 
            references=[[t] for t in all_targets]
        )
        
        eval_metrics.update({
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "bleu": bleu_scores["bleu"]
        })
    
    return eval_metrics


def train_model(
    train_gen_json_path,
    train_score_json_path,
    test_gen_json_path,
    test_score_json_path,
    output_dir="results",
    use_lora=True,
    num_epochs=3,
    learning_rate=5e-5,
    batch_size=1,
    accumulation_steps=8
):
    """训练Qwen2-Audio模型（支持单机多卡和测试集评估）"""
    # 设置随机种子
    set_seed(42)
    
    # 初始化分布式训练环境
    local_rank, world_size, device = setup_distributed()
    is_distributed = world_size > 1
    
    # 创建输出目录（主进程）
    if local_rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # 只在主进程显示信息
    if local_rank == 0:
        print(f"Initializing training with {world_size} GPU(s)")
        print(f"Local rank: {local_rank}, World size: {world_size}")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    
    # 创建训练集和测试集
    train_dataset = AudioDataset(
        gen_json_path=train_gen_json_path,
        score_json_path=train_score_json_path,
        processor=processor,
        max_length=30,
        is_distributed=is_distributed,
        mode='train'
    )
    
    test_dataset = AudioDataset(
        gen_json_path=test_gen_json_path,
        score_json_path=test_score_json_path,
        processor=processor,
        max_length=30,
        is_distributed=is_distributed,
        mode='test'
    )
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=True
    ) if is_distributed else None
    
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=False
    ) if is_distributed else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lambda batch: {
            "raw_audio": [item["raw_audio"] for item in batch],
            "text": [item["text"] for item in batch],
            "audio_path": [item["audio_path"] for item in batch],
            "target": [item["target"] for item in batch]
        },
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=lambda batch: {
            "raw_audio": [item["raw_audio"] for item in batch],
            "text": [item["text"] for item in batch],
            "audio_path": [item["audio_path"] for item in batch],
            "target": [item["target"] for item in batch]
        },
        num_workers=2,
        pin_memory=True
    )
    
    # 加载模型
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=torch.bfloat16
    )
    
    # 应用LoRA微调
    if use_lora:
        from peft import LoraConfig, get_peft_model
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)
        
        if local_rank == 0:
            model.print_trainable_parameters()
    
    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    
    # 将模型移至设备
    model.to(device)
    
    # 如果是分布式训练，使用DDP包装模型
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 配置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(train_loader) // accumulation_steps,
        eta_min=learning_rate * 0.1
    )
    
    # 混合精度训练
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    
    # 训练循环
    best_val_loss = float('inf')
    training_log = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        # 训练步骤
        for step, batch in enumerate(train_loader):
            step_start = time.time()
            
            # 过滤无效数据
            valid_indices = [i for i, path in enumerate(batch["audio_path"]) if path]
            if not valid_indices:
                continue
                
            # 只处理有效数据
            batch = {k: [v[i] for i in valid_indices] for k, v in batch.items()}
            
            print(batch)

            # 准备输入
            inputs = processor(
                text=batch["text"],
                audios=batch["raw_audio"],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            print(inputs)
            
            # 混合精度训练
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            total_train_loss += loss.item() * accumulation_steps * len(valid_indices)
            
            # 梯度累积
            if (step + 1) % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                # 打印训练信息（只在主进程）
                if local_rank == 0 and step % (10 * accumulation_steps) == 0:
                    step_time = time.time() - step_start
                    samples_per_sec = accumulation_steps * batch_size * len(valid_indices) / step_time
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, "
                          f"Loss: {total_train_loss / ((step + 1) * batch_size * accumulation_steps):.4f}, "
                          f"LR: {current_lr:.2e}, "
                          f"Speed: {samples_per_sec:.1f} samples/s")
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # 评估测试集
        metric = evaluate.combine(["rouge", "bleu"])
        eval_metrics = evaluate_model(
            model.module if is_distributed else model,
            processor,
            test_loader,
            device,
            local_rank,
            metric
        )
        
        # 只在主进程记录和保存
        if local_rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s")
            print(f"Train Loss: {avg_train_loss:.4f}")
            
            if eval_metrics:
                print("Test Metrics:")
                for k, v in eval_metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            # 记录日志
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "time": epoch_time,
                **eval_metrics
            }
            training_log.append(log_entry)
            
            # 保存日志
            with open(os.path.join(output_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)
            
            # 保存检查点（每个epoch）
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存模型
            if use_lora:
                model.module.save_pretrained(checkpoint_dir)
            else:
                model.save_pretrained(checkpoint_dir)
            
            # 保存处理器
            processor.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
            
            # 保存最佳模型
            if eval_metrics and "loss" in eval_metrics:
                val_loss = eval_metrics["loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_dir = os.path.join(output_dir, "best_model")
                    os.makedirs(best_checkpoint_dir, exist_ok=True)
                    
                    if use_lora:
                        model.module.save_pretrained(best_checkpoint_dir)
                    else:
                        model.save_pretrained(best_checkpoint_dir)
                    
                    processor.save_pretrained(best_checkpoint_dir)
                    print(f"New best model saved with val_loss: {val_loss:.4f}")
    
    # 清理分布式训练环境
    if is_distributed:
        dist.destroy_process_group()
    
    # 最终评估（只在主进程）
    if local_rank == 0:
        print("\nTraining completed!")
        print("Final evaluation on test set:")
        
        # 加载最佳模型
        best_model_path = os.path.join(output_dir, "best_model")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            best_model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
        
        # 最终评估
        final_metrics = evaluate_model(
            model,
            processor,
            test_loader,
            device,
            local_rank,
            metric
        )
        
        print("Final Test Metrics:")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # 保存最终指标
        with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)
