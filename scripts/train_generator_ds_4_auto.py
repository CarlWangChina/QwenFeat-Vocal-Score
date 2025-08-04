import os
import sys
for tag_name in ["ldz"]:
    for i in [2,3]:
        for t in ["noise", "denoise"]:
            ckpt_path = f"/home/w-4090/projects/qwenaudio/ckpts/train_ds_4_{tag_name}/{t}/{i}/"
            data_path = f"/home/w-4090/projects/qwenaudio/data/train_ds_4_{tag_name}/{t}/{i}/"
            print(data_path)
            os.makedirs(ckpt_path+"/text", exist_ok=True)
            cmd = f"torchrun --nproc_per_node=7 --nnodes=1 scripts/train_generator_ds_4.py "+\
                f"--train_set {data_path}/train_text.json "+\
                f"--val_set {data_path}/test_text.json "+\
                f"--save_dir {ckpt_path}/text/ "
            os.system(cmd)
            
            os.makedirs(ckpt_path+"/score", exist_ok=True)
            cmd = f"torchrun --nproc_per_node=7 --nnodes=1 scripts/train_generator_ds_4.py "+\
                f"--train_set {data_path}/train_score.json "+\
                f"--val_set {data_path}/test_score.json "+\
                f"--save_dir {ckpt_path}/score/ "
            os.system(cmd)