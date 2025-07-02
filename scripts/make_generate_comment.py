import json
import os

with open("/home/w-4090/projects/qwenaudio/outputs/test_loadmodel_pfsc.json") as fp:
    data = json.load(fp)
    print(data['epoch_76'].keys())
    count = 0
    generated_list = []
    for item in data['epoch_76']['results']:
        del item["generate"]["text_by_genmodel"]
        del item["generate"]["predict_from_score_model_without_comment"]
        if item["generate"]["generate_comment"]["score_dist"]<=1:
            # print(item["generate"]["generate_comment"])
            count+=1
            generated_list.append(item)
        # else:
            #删除item
            # data['epoch_76']['results'].remove(item)
    # print(count)
    data['epoch_76']['results'] = generated_list

    del data['epoch_76']["data_count"]
    del data['epoch_76']["acc"]

    with open("/home/w-4090/projects/qwenaudio/outputs/test_loadmodel_pfsc_comment.json", "w") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
