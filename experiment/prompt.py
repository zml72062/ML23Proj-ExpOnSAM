import utils
import numpy as np
from typing import List, Tuple, Optional, Union

def generate_prompt(input_label: np.ndarray,
                    point_prompt: List[str],
                    bounding_box_prompt: bool,
                    bounding_box_margin: int) \
      -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate points and (or) boxes prompts, according to the input arguments.
    Return a tuple (points, labels, box), whose shapes are
    (num_points, 2), (num_points, ) and (4, ) respectively.
    When either of the prompts is absent, return None for it.
    """
    if input_label.sum() == 0: # all zero
        return None, None, None

    prompt_counts = {key: point_prompt.count(key)
                     for key in ['center', 'random', 'bg_random']}
    
    # Generate foreground random points and labels
    fg_points = utils.find_fg_random(input_label, prompt_counts['random'])
    fg_labels = np.ones((fg_points.shape[0], ), dtype=np.int8)
    # Generate background random points and labels
    bg_points = utils.find_bg_random(input_label, prompt_counts['bg_random'])
    bg_labels = np.zeros((bg_points.shape[0], ), dtype=np.int8)
    # Generate center points and labels
    center_point = utils.find_center(input_label)[None, :] if prompt_counts['center'] > 0 \
                   else np.ones((0, 2), dtype=np.int64)
    center_label = np.ones((center_point.shape[0], ), dtype=np.int8)
    
    points = np.concatenate([center_point, fg_points, bg_points], axis=0)
    labels = np.concatenate([center_label, fg_labels, bg_labels], axis=0)
    
    if points.shape[0] == 0:
        points = None
        labels = None

    if bounding_box_prompt:
        box = utils.find_bounding_box(input_label, bounding_box_margin)
    else:
        box = None
    
    return points, labels, box

def print_prompt(point_prompt: List[str],
                 bounding_box_prompt: bool,
                 bounding_box_margin: int) -> None:
    prompt_counts = {key: point_prompt.count(key)
                     for key in ['center', 'random', 'bg_random']}
    print(f"Using {prompt_counts['random']} random foreground points, ")
    print(f"      {prompt_counts['bg_random']} random background points, ")
    if prompt_counts['center'] > 0:
        print(f"      and center point, ")
    if bounding_box_prompt:
        print(f"Using bounding box with margin {bounding_box_margin},")

def generate_batched_prompt(input_label: np.ndarray,
                            point_prompt: List[str],
                            bounding_box_prompt: bool,
                            bounding_box_margin: int,
                            targets: Optional[Union[int, List[int]]] = None) \
      -> Tuple[List[int], Optional[np.ndarray], 
               Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate points and (or) boxes prompts, against multiple targets.
    Return a tuple (generated_targets, points, labels, box),
    the first component is a list of targets that are present in `input_label`, 
    and, thus, against which a prompt is generated;
    the last three components are similar to the return values of `generate_prompt()`, 
    but are batched versions.
    """
    if targets is None: # No target specified, segment all organs
        targets = list(range(1, 14))
    elif isinstance(targets, int):
        targets = [targets]
    
    batched_coords = []
    batched_labels = []
    batched_boxes = []
    batched_targets = []

    for target in targets:
        selected_label = utils.select_label(input_label, target)
        coords, labels, box = generate_prompt(
            selected_label,
            point_prompt,
            bounding_box_prompt,
            bounding_box_margin
        )
        if coords is None and labels is None and box is None:
            continue

        batched_targets.append(target)
        if coords is not None:
            batched_coords.append(coords)
        if labels is not None:
            batched_labels.append(labels)
        if box is not None:
            batched_boxes.append(box)
    
    if len(batched_coords) > 0:
        batched_coords = np.stack(batched_coords, axis=0)
    else:
        batched_coords = None

    if len(batched_labels) > 0:
        batched_labels = np.stack(batched_labels, axis=0)
    else:
        batched_labels = None

    if len(batched_boxes) > 0:
        batched_boxes = np.stack(batched_boxes, axis=0)
    else:
        batched_boxes = None

    return batched_targets, batched_coords, batched_labels, batched_boxes