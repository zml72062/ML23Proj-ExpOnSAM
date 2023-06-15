import sys
sys.path.append('..')
import utils
import prompt
import json
import os.path as osp
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional, Union
import torch
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

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
    
def prepare_input(sam: Sam,
                  data: np.ndarray, 
                  labels: np.ndarray,
                  point_prompt: List[str],
                  bounding_box_prompt: bool,
                  bounding_box_margin: int,
                  targets: Optional[Union[int, List[int]]] = None)\
         -> Tuple[List[Tuple[int, List[int]]], List[dict]]:
    """
    Prepare batched input to SAM model. Return two lists, the first is a
    list of (z_coor, targets present in that slice), the second is the 
    batched input which will be fed into SAM model.
    NOTE: The batched input will be load to the same GPU as the SAM model.
    """
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    # A list that records (z_coor, targets present in that slice)
    target_list: List[Tuple[int, List[int]]] = []
    # The batched input which will be fed into SAM model
    batched_input: List[dict] = []

    z_slices = labels.shape[-1]
    for i in range(z_slices):
        # Prepare image
        image_i = torch.from_numpy(
            transform.apply_image(data[..., i])
        ).to(sam.device).permute(2, 0, 1).contiguous()

        gen_targets, point_coords, point_labels, box = prompt.generate_batched_prompt(
            labels[..., i], 
            point_prompt, 
            bounding_box_prompt, 
            bounding_box_margin,
            targets
        )
        if point_coords is None and point_labels is None and box is None:
            continue

        target_list.append((i, gen_targets))

        input_dict = {
            'image': image_i,
            'original_size': data.shape[:2]
        }
        # Prepare point prompt
        if point_coords is not None:
            point_coords = transform.apply_coords_torch(
                torch.from_numpy(point_coords).to(sam.device), data.shape[:2]
            )
            input_dict['point_coords'] = point_coords
        if point_labels is not None:
            point_labels = torch.from_numpy(point_labels).to(sam.device)
            input_dict['point_labels'] = point_labels
        
        # Prepare box prompt
        if box is not None:
            box = transform.apply_boxes_torch(
                torch.from_numpy(box).to(sam.device), data.shape[:2]
            )
            input_dict['boxes'] = box

        batched_input.append(input_dict)
    return target_list, batched_input
