# torchrun --nproc_per_node=6 --nnodes=1 scripts/train/train_grl.py --score_axis 0
export CUDA_VISIBLE_DEVICES=2,3,4,5

torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 0 --alpha 0.15
torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 0 --alpha 0.25
torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 1 --alpha 0.15
torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 1 --alpha 0.25
torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 2 --alpha 0.15
torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 2 --alpha 0.25
torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 3 --alpha 0.15
torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl.py --score_axis 3 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 0 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 0 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 0 --gm 0.0000005 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 1 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 1 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 1 --gm 0.0000005 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 2 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 2 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 2 --gm 0.0000005 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 3 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 3 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_whisper.py --score_axis 3 --gm 0.0000005 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 0 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 0 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 0 --gm 0.0000005 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 1 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 1 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 1 --gm 0.0000005 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 2 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 2 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 2 --gm 0.0000005 --alpha 0.25
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 3 --gm 0.0000005 --alpha 0.15
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 3 --gm 0.0000005 --alpha 0.2
# torchrun --nproc_per_node=3 --nnodes=1 scripts/train/train_grl_hubert.py --score_axis 3 --gm 0.0000005 --alpha 0.25
