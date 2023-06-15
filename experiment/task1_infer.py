"""
In this script, we 
    1) Cut every 3D medical image into 2D slices, 
    2) Perform segmentation of each of the 13 organs on each slice,
    3) Combine the segmentation results into a 3D mask, and
    4) Calculate mDice score.
"""

import utils
import argparse
import sys
import torch
import numpy as np
import prompt
import random
import math
from tqdm import tqdm
sys.path.append('..')
import data_proc
    
def get_3d_segment(labels, range_start, used_targets, output):
    segment_result = np.zeros((14, *labels.shape), dtype=np.int8)
    for out, (z, targets) in zip(output, used_targets):
        batched_mask = out['masks'].cpu().numpy()
        segment_result[targets, :, :, z - range_start] = batched_mask[:, 0, :, :]
    return segment_result

def compute_mdice(dice_dict):
    count = 0
    sum = 0
    for val in dice_dict.values():
        if not math.isnan(val):
            count += 1
            sum += val
    return sum / count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command-line arguments for '
                                     'segmentation on BTCV using SAM')
    parser.add_argument('--target', type=int, default=0,
                        help='target to segment, if 0, segment all targets')
    parser.add_argument('--data_dir', type=str, default='RawData',
                        help='directory of the raw data')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to load model and data to')
    parser.add_argument('--json_path', type=str, default='RawData/dataset_0.json',
                        help='path of the json file that describes data')
    parser.add_argument('--point_prompt', type=str, nargs='*',
                        help='policy to add point prompt, can be a list of '
                             '"random", "center" or "bg_random"')
    parser.add_argument('--bounding_box_prompt', action='store_true',
                        help='whether to use bounding box prompt')
    parser.add_argument('--box_margin', type=int, default=0,
                        help='margin of the bounding box')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='how many 2D slice to copy to GPU at a time')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    training_data, training_labels = data_proc.load_dataset('training', args.json_path, args.data_dir)
    validation_data, validation_labels = data_proc.load_dataset('validation', args.json_path, args.data_dir)
    sam = utils.get_sam(device=args.device)

    if args.point_prompt is None:
        args.point_prompt = []

    prompt.print_prompt(args.point_prompt, args.bounding_box_prompt, args.box_margin)

    training_dices = []
    validation_dices = []

    print("Segmenting on training dataset...", file=sys.stderr)
    for i_sample, (data, label) in enumerate(zip(training_data, training_labels)):
        print(f"Sample {i_sample}:")
        split_ranges = list(range(0, data.shape[-1], args.batch_size)) + [data.shape[-1]]
        segment_results = []
        for idx in tqdm(range(len(split_ranges) - 1)):
            next_range = range(split_ranges[idx], split_ranges[idx + 1])
            used_targets, batched_input = data_proc.prepare_input(
                sam, data, label, next_range, 
                args.point_prompt, args.bounding_box_prompt, args.box_margin
            )
            if len(batched_input) == 0:
                segment_results.append(np.zeros((14, *label[..., next_range].shape), dtype=np.int8))
                torch.cuda.empty_cache()
                continue
            output = sam(batched_input, multimask_output=False)
            segment_result = get_3d_segment(label[..., next_range], 
                                            split_ranges[idx],
                                            used_targets, output)
            segment_results.append(segment_result)
            torch.cuda.empty_cache()
        segment_result = np.concatenate(segment_results, axis=-1)

        dice_dict = {}
        for target in range(1, 14):
            truth = utils.select_label(label, target)
            if truth.sum() == 0:
                dice_dict[target] = float('nan')
                continue
            pred = segment_result[target]
            dice_dict[target] = utils.dice_score(pred, truth)

        print("Dice scores: ")
        print(dice_dict)
        print(f"mDice:  {compute_mdice(dice_dict)}")
        torch.cuda.empty_cache()

    print("Segmenting on validation dataset...", file=sys.stderr)
    for i_sample, (data, label) in enumerate(zip(validation_data, validation_labels)):
        print(f"Sample {i_sample}:")
        split_ranges = list(range(0, data.shape[-1], args.batch_size)) + [data.shape[-1]]
        segment_results = []
        for idx in tqdm(range(len(split_ranges) - 1)):
            next_range = range(split_ranges[idx], split_ranges[idx + 1])
            used_targets, batched_input = data_proc.prepare_input(
                sam, data, label, next_range, 
                args.point_prompt, args.bounding_box_prompt, args.box_margin
            )
            if len(batched_input) == 0:
                segment_results.append(np.zeros((14, *label[..., next_range].shape), dtype=np.int8))
                torch.cuda.empty_cache()
                continue
            output = sam(batched_input, multimask_output=False)
            segment_result = get_3d_segment(label[..., next_range], 
                                            split_ranges[idx],
                                            used_targets, output)
            segment_results.append(segment_result)
            torch.cuda.empty_cache()
        segment_result = np.concatenate(segment_results, axis=-1)

        dice_dict = {}
        for target in range(1, 14):
            truth = utils.select_label(label, target)
            if truth.sum() == 0:
                dice_dict[target] = float('nan')
                continue
            pred = segment_result[target]
            dice_dict[target] = utils.dice_score(pred, truth)

        print("Dice scores: ")
        print(dice_dict)
        print(f"mDice:  {compute_mdice(dice_dict)}")
        torch.cuda.empty_cache()



