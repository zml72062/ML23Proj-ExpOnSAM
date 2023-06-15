import json
import numpy as np
prompt_modes = ["single_point_cog", "multi_points_rr", "multi_points_rw", "box"]

with open('../dataset/dataset_0.json', 'r') as f:
    data_info = json.load(f)

def convert_dict_to_mdice(d):
    l = []
    for key in d.keys():
        if 'slice' in key:
            if d[key]:
                l.append(d[key][0])
    if l:
        return np.mean(l)
    else:
        return None

results = []
for mode in ["single_point_cog", "multi_points_rr", "multi_points_rw", "box"]:
    prompt_results = {"prompt_mode": mode}
    mDice = 0
    ctr = 0
    for fg_label in range(1, 14):
        with open(f"../output/{mode}/{fg_label}.{data_info['labels'][str(fg_label)]}/dice_score.json",'r') as f:
            result = json.load(f) # list of dict
        organ_mean = np.array(list(map(lambda x: convert_dict_to_mdice(x), result)))
        mean_dice = np.mean(organ_mean[organ_mean != None]) # for a specific class, average over slices and
        prompt_results[f"{fg_label}.{data_info['labels'][str(fg_label)]}_dice"] = mean_dice
        if mean_dice:
            mDice += mean_dice
            ctr += 1
            print(mDice)

    prompt_results["mDice"] = mDice / ctr
    results.append(prompt_results)

with open('../output/prompt_result.json', 'w') as f:
    json.dump(results, f)