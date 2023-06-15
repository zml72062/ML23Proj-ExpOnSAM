import utils
import json
import os.path as osp
from tqdm import tqdm
import sys

def load_dataset(split: str = 'training',
                 json_path: str = 'RawData/dataset_0.json', 
                 data_dir: str = 'RawData'):
    """
    Load a split of dataset, as a tuple (list of images, list of labels).
    The `split` can be `'training'` or `'validation'`.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
        split_data = []
        split_labels = []
        print(f"Loading '{split}' split: ", file=sys.stderr)
        for split_dict in tqdm(config[split]):
            data = utils.load_data(osp.join(data_dir, split_dict['image']))
            data = utils.gray_to_rgb(data)
            split_data.append(data)

            label = utils.load_label(osp.join(data_dir, split_dict['label']))
            split_labels.append(label)
        return split_data, split_labels