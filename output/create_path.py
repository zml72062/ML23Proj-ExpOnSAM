import os
import json

def mkdir(path):

	folder = os.path.exists(path)

	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径


with open('../dataset/dataset_0.json', 'r') as f:
    data_info = json.load(f)
for mode in ["single_point_cog", "multi_points_rr", "multi_points_rw", "box"]:
    for fg_label in range(1, 14):
        print(f"creating folders {mode}...")
        print(f"looking for {fg_label}.{data_info['labels'][str(fg_label)]}...")
        file = f"{mode}/{fg_label}.{data_info['labels'][str(fg_label)]}/"
        mkdir(file)             #调用函数