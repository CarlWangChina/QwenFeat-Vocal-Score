import torch
import os
import sys
import csv

def test_mulan_dist(dir_path="data/processed_mulan", csv_path="outputs/mulan_dist_test.csv"):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "num_seg", "dist_ori_emb", "dist_user_emb", "dist_ori_seg_dist_average", "dist_user_seg_dist_average"])
        for root, dirs, files in os.walk(dir_path):
                for file in files:
                    # print(file)
                    if file.endswith(".pkl"):
                        data = torch.load(os.path.join(root, file), map_location=torch.device('cpu'), weights_only=True)
                        # print(data.keys()) #['ori_seg', 'user_seg', 'ori', 'user', 'text_emb', 'text']
                        num_seg = len(data["ori_seg"])
                        assert num_seg == len(data["user_seg"])
                        text_emb = data["text_emb"]
                        ori_emb = data["ori"]
                        user_emb = data["user"]
                        dist_ori_emb = torch.cdist(ori_emb, text_emb)
                        dist_user_emb = torch.cdist(user_emb, text_emb)
                        
                        dist_ori_seg_dist_sum = 0
                        dist_user_seg_dist_sum = 0
                        for i in range(num_seg):
                            dist_ori_seg_dist = torch.cdist(data["ori_seg"][i], text_emb)
                            dist_user_seg_dist = torch.cdist(data["user_seg"][i], text_emb)
                            # print(data["ori_seg"][i].shape, text_emb.shape, dist_ori_seg_dist.shape)
                            dist_ori_seg_dist_sum += dist_ori_seg_dist
                            dist_user_seg_dist_sum += dist_user_seg_dist
                        dist_ori_seg_dist_average = dist_ori_seg_dist_sum / num_seg
                        dist_user_seg_dist_average = dist_user_seg_dist_sum / num_seg
                        writer.writerow([
                             file, 
                             num_seg, 
                             dist_ori_emb.view(-1).mean().item(), 
                             dist_user_emb.view(-1).mean().item(), 
                             dist_ori_seg_dist_average.view(-1).mean().item(), 
                             dist_user_seg_dist_average.view(-1).mean().item()])


if __name__ == "__main__":
    test_mulan_dist()