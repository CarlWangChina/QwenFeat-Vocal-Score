import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
# print(sys.path)
import torch
import audioscore.dataset

ds = audioscore.dataset.AudioDataset_hybrid_multifile(target_file="data/train_ds_4_al/denoise/0/test_score.json")
data = ds.__getitem__(0)
print(type(data))


print(data["samoye_f0"]["ori"].shape) # [seq_size*4]
print(data["samoye_f0"]["user"].shape) # [seq_size*4]
print(data["samoye_whisper"]["ori"].shape) # [seq_size*2, 1280]
print(data["samoye_whisper"]["user"].shape) # [seq_size*2, 1280]
print(data["samoye_hubert"]["ori"].shape) # [seq_size*2, 256]
print(data["samoye_hubert"]["user"].shape) # [seq_size*2, 256]
print(data["mert"]["ori"].shape) # [seq_size*3, 1024]
print(data["mert"]["user"].shape) # [seq_size*3, 1024]
print(data["muq"]["ori"].shape) # [1, seq_size, 1024]
print(data["muq"]["user"].shape) # [1, seq_size, 1024]
print(data["score"])

print(ds.collect_fn([ds.__getitem__(0),ds.__getitem__(1)]))