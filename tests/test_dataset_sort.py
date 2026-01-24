import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
# print(sys.path)
import torch
import audioscore.dataset
import audioscore.model

ds = audioscore.dataset.AudioDataset_scoring_hybrid_feature("data/sort/data/index/index.json")
data = ds.__getitem__(0)
print(type(data))
# print(data)
print(ds.__len__())

print(data["samoye_pitch"].shape) # [seq_size*4]
print(data["samoye_whisper"].shape) # [seq_size*2, 1280]
print(data["samoye_hubert"].shape) # [seq_size*2, 256]
print(data["mert"].shape) # [seq_size*3, 1024]
print(data["muq"].shape) # [1, seq_size, 1024]
print(data["score"]) # 0到1之间的浮点数，作为模型输出

test_collate = ds.collate_fn([ds.__getitem__(0),ds.__getitem__(1)])
model = audioscore.model.AudioFeatHybrid().cuda()
print(model(
    samoye_pitch=test_collate["samoye_pitch"].cuda(), 
    samoye_whisper=test_collate["samoye_whisper"].cuda(), 
    samoye_hubert=test_collate["samoye_hubert"].cuda(), 
    mert=test_collate["mert"].cuda(), 
    muq=test_collate["muq"].cuda(), 
    muq_mask = test_collate["mask"]["muq"].cuda()
    ))
