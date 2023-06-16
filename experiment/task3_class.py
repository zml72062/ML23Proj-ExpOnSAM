"""
In this script, we 
    1) Cut every 3D medical image into 2D slices, 
    2) Perform segmentation of each of the 13 organs on each slice,
    3) Combine the segmentation results into a 3D mask, 
    4) Compute BCEWithLogitsLoss between prediction and ground truth,
    5) Compute NLLLoss between predicted class and true class,
    6) Compute mDice score between prediction and ground truth,
       where the class is obtained by the model,
    8) Update the parameters of the decoder.
"""

import utils
import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
import prompt
import random
from tqdm import tqdm
sys.path.append('..')
import data_proc
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import (ReduceLROnPlateau, 
                                      StepLR, 
                                      CosineAnnealingLR, 
                                      CosineAnnealingWarmRestarts)
from task1_infer import compute_mdice
from typing import List, Tuple, Optional, Union
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

def dice_for_all_targets(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    targets = np.arange(1, 14)[:, None, None, None]
    pred_binary, truth_binary = (pred[None, ...] == targets), (truth[None, ...] == targets)
    enumerator = (pred_binary & truth_binary).sum(axis=(-3, -2, -1))
    denominator = pred_binary.sum(axis=(-3, -2, -1)) + truth_binary.sum(axis=(-3, -2, -1))
    return enumerator / denominator # may contain nan

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
    parser.add_argument('--grid_prompt', action='store_true',
                        help='whether to use grid prompt')
    parser.add_argument('--grid_distance', type=int, default=16,
                        help='grid distance of the grid prompt')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='how many 2D slice to copy to GPU at a time')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--inference', action='store_true',
                        help='whether to disable fine-tuning on BTCV dataset')
    
    # Training settings
    parser.add_argument('--valid_fold', type=int, default=5,
                        help='split training data into 6 folds, which fold as validation')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer to use, "Adam" or "SGD", default "Adam"')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate, default 1e-5')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    training_data, training_labels = data_proc.load_dataset('training', args.json_path, args.data_dir)
    valid_split_slice = list(range(4 * args.valid_fold, 4 * (args.valid_fold + 1)))

    validation_data = [training_data[i] for i in valid_split_slice]
    validation_labels = [training_labels[i] for i in valid_split_slice]

    training_data = [training_data[i] for i in range(24) if i not in valid_split_slice]
    training_labels = [training_labels[i] for i in range(24) if i not in valid_split_slice]

    test_data, test_labels = data_proc.load_dataset('validation', args.json_path, args.data_dir)

    sam = utils.get_sam(device=args.device, model_type='vit_h_class')
    optimizer_kwargs = {}
    optimizer = eval(args.optimizer)(sam.parameters(), lr=args.lr, **optimizer_kwargs)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.8, patience=10, min_lr=1e-8)

    if args.point_prompt is None:
        args.point_prompt = []

    prompt.print_prompt(args.point_prompt, args.bounding_box_prompt, args.box_margin)

    training_dices = []
    validation_dices = []
    best_validation_avg_mdice = 0.0
    best_test_mdices = []

    for epoch in range(args.epochs):
        print(f"============   Epoch {epoch + 1}   ============")
        
        print("Segmenting on training dataset...", file=sys.stderr)
        sam.train()
        training_dices = []
        for i_sample, (data, label) in enumerate(zip(training_data, training_labels)):
            training_labels_binary = (label[None, ...] == np.arange(14)[:, None, None, None]).astype(np.int8)
            print(f"Sample {i_sample}:")
            split_ranges = list(range(0, data.shape[-1], args.batch_size)) + [data.shape[-1]]
            segment_results = []
            accumulated_loss = 0.0

            for idx in tqdm(range(len(split_ranges) - 1)):
                segment_loss = 0.0
                classify_loss = 0.0
                optimizer.zero_grad()

                next_range = range(split_ranges[idx], split_ranges[idx + 1])
                if args.grid_prompt:
                    used_targets, batched_input = data_proc.prepare_grid_input(
                        sam, data, label, next_range, 
                        args.grid_distance
                    )
                else:
                    used_targets, batched_input = data_proc.prepare_input(
                        sam, data, label, next_range, 
                        args.point_prompt, args.bounding_box_prompt, args.box_margin
                    )
                if len(batched_input) == 0:
                    # gradient is always 0, no backward propagation
                    segment_results.append(np.zeros_like(label[..., next_range], dtype=np.int8))
                    torch.cuda.empty_cache()
                    continue
                output = sam(batched_input, multimask_output=False)
                num_targets = 0
                predicted_targets = []
                for (z, targets), out in zip(used_targets, output):
                    truth = torch.from_numpy(training_labels_binary[targets, ..., next_range]).flatten().to(args.device).to(torch.float)
                    pred = out['masks'][:, 0, :, :].flatten()
                    # cross entropy loss of ALL targets on a given slice
                    segment_loss += nn.BCEWithLogitsLoss(reduction='sum')(pred, truth)

                    pred_class = out['class_prediction'] # n_targets * n_classes
                    true_class = (torch.tensor(targets, dtype=int) - 1).to(args.device) # n_targets
                    # cross entropy loss of class predictions of ALL targets on a given slice
                    classify_loss += nn.NLLLoss(reduction='sum')(pred_class, true_class)
                    num_targets += len(targets)

                    with torch.no_grad():
                        predicted_targets.append((z, 1 + torch.argmax(pred_class, dim=1)))
                
                segment_loss /= (out['masks'].shape[-1] * out['masks'].shape[-2])
                classify_loss /= num_targets

                # TODO: May change weight, a balance b/w two kinds of losses
                slice_loss = segment_loss + classify_loss
                slice_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    accumulated_loss += slice_loss.item()
                    
                    segment_result = np.zeros_like(label[..., next_range], dtype=np.int8)
                    for out, (z, targets) in zip(output, predicted_targets):
                        out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                        segment_result[:, :, z - split_ranges[idx]] = (
                            targets[:, None, None, None] * out['masks']
                        ).max(dim=0).values.cpu().numpy()[0]
                    segment_results.append(segment_result)
                torch.cuda.empty_cache()
            
            segment_result = np.concatenate(segment_results, axis=-1)

            dice = dice_for_all_targets(segment_result, label)
            dice_dict = {i + 1: dice[i] for i in range(13)}

            print(f"Accumulated loss: {accumulated_loss}")
            print("Dice scores: ")
            print(dice_dict)
            train_dice = compute_mdice(dice_dict)
            training_dices.append(train_dice)
            print(f"mDice:  {train_dice}")
            torch.cuda.empty_cache()
        print(f"Average training mDice: {sum(training_dices) / 20}")

        sam.eval()
        with torch.no_grad():
            print("Segmenting on validation dataset...", file=sys.stderr)
            validation_dices = []
            for i_sample, (data, label) in enumerate(zip(validation_data, validation_labels)):
                validation_labels_binary = (label[None, ...] == np.arange(14)[:, None, None, None]).astype(np.int8)
                print(f"Sample {i_sample}:")
                split_ranges = list(range(0, data.shape[-1], args.batch_size)) + [data.shape[-1]]
                segment_results = []

                for idx in tqdm(range(len(split_ranges) - 1)):
                    next_range = range(split_ranges[idx], split_ranges[idx + 1])
                    if args.grid_prompt:
                        used_targets, batched_input = data_proc.prepare_grid_input(
                            sam, data, label, next_range, 
                            args.grid_distance
                        )
                    else:
                        used_targets, batched_input = data_proc.prepare_input(
                            sam, data, label, next_range, 
                            args.point_prompt, args.bounding_box_prompt, args.box_margin
                        )
                    if len(batched_input) == 0:
                        # gradient is always 0, no backward propagation
                        segment_results.append(np.zeros_like(label[..., next_range], dtype=np.int8))
                        torch.cuda.empty_cache()
                        continue
                    output = sam(batched_input, multimask_output=False)
                    predicted_targets = []
                    for (z, targets), out in zip(used_targets, output):
                        pred_class = out['class_prediction'] # n_targets * n_classes
                        predicted_targets.append((z, 1 + torch.argmax(pred_class, dim=1)))
                    segment_result = np.zeros_like(label[..., next_range], dtype=np.int8)
                    for out, (z, targets) in zip(output, predicted_targets):
                        out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                        segment_result[:, :, z - split_ranges[idx]] = (
                            targets[:, None, None, None] * out['masks']
                        ).max(dim=0).values.cpu().numpy()[0]
                    segment_results.append(segment_result)
                    torch.cuda.empty_cache()
                
                segment_result = np.concatenate(segment_results, axis=-1)

                dice = dice_for_all_targets(segment_result, label)
                dice_dict = {i + 1: dice[i] for i in range(13)}
                print("Dice scores: ")
                print(dice_dict)
                valid_dice = compute_mdice(dice_dict)
                validation_dices.append(valid_dice)
                print(f"mDice:  {valid_dice}")
                torch.cuda.empty_cache()
            avg_validation_mdice = sum(validation_dices) / 4
            print(f"Average validation mDice: {avg_validation_mdice}")
            if lr_scheduler is not None:
                if lr_scheduler.__class__ == ReduceLROnPlateau:
                    lr_scheduler.step(avg_validation_mdice)
                else:
                    lr_scheduler.step()

            print("Segmenting on test dataset...", file=sys.stderr)
            test_dices = []
            for i_sample, (data, label) in enumerate(zip(test_data, test_labels)):
                test_labels_binary = (label[None, ...] == np.arange(14)[:, None, None, None]).astype(np.int8)
                print(f"Sample {i_sample}:")
                split_ranges = list(range(0, data.shape[-1], args.batch_size)) + [data.shape[-1]]
                segment_results = []

                for idx in tqdm(range(len(split_ranges) - 1)):
                    next_range = range(split_ranges[idx], split_ranges[idx + 1])
                    if args.grid_prompt:
                        used_targets, batched_input = data_proc.prepare_grid_input(
                            sam, data, label, next_range, 
                            args.grid_distance
                        )
                    else:
                        used_targets, batched_input = data_proc.prepare_input(
                            sam, data, label, next_range, 
                            args.point_prompt, args.bounding_box_prompt, args.box_margin
                        )
                    if len(batched_input) == 0:
                        # gradient is always 0, no backward propagation
                        segment_results.append(np.zeros_like(label[..., next_range], dtype=np.int8))
                        torch.cuda.empty_cache()
                        continue
                    output = sam(batched_input, multimask_output=False)
                    predicted_targets = []
                    for (z, targets), out in zip(used_targets, output):
                        pred_class = out['class_prediction'] # n_targets * n_classes
                        predicted_targets.append((z, 1 + torch.argmax(pred_class, dim=1)))
                    segment_result = np.zeros_like(label[..., next_range], dtype=np.int8)
                    for out, (z, targets) in zip(output, predicted_targets):
                        out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                        segment_result[:, :, z - split_ranges[idx]] = (
                            targets[:, None, None, None] * out['masks']
                        ).max(dim=0).values.cpu().numpy()[0]
                    segment_results.append(segment_result)
                    torch.cuda.empty_cache()
                
                segment_result = np.concatenate(segment_results, axis=-1)

                dice = dice_for_all_targets(segment_result, label)
                dice_dict = {i + 1: dice[i] for i in range(13)}
                print("Dice scores: ")
                print(dice_dict)
                test_dice = compute_mdice(dice_dict)
                test_dices.append(test_dice)
                print(f"mDice:  {test_dice}")
                torch.cuda.empty_cache()

            if avg_validation_mdice > best_validation_avg_mdice:
                best_validation_avg_mdice = avg_validation_mdice
                best_test_mdices = test_dices
    print(f"Best test mDices: {best_test_mdices}")
    torch.save(sam.state_dict(), '../fine_tuned_sam_vit_h_with_classification.pth')

