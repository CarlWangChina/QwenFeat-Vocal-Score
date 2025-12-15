torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 scripts/train_ddp.py \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --output_dir ./checkpoints \
  --batch_size 8 \
  --epochs 10 \
  --lr 1e-4 \
  --max_length 10
