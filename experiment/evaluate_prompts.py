import utils
import sys
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


predictor = utils.get_sam_predictor()


image_fold_path = '../dataset/valid'
label_fold_path = '../dataset/valid_label'
image_filenames = os.listdir(image_fold_path)
label_filenames = os.listdir(label_fold_path)


def experiment(mode: str, fg_label: int, slice_step: int):
    for label_filename in label_filenames:
        # get image data

        image_filename = re.sub('label','img', label_filename)
        print(f"loading {image_filename}...")
        image_path = image_fold_path + f'/{image_filename}'
        image_data = utils.load_data(image_path)
        dice_result = {"image": image_filename}

        max_slice = image_data.shape[2]

        for slice in tqdm(range(0, max_slice, slice_step)):
            dice_result[f"slice_{slice}"] = []
            image = utils.gray_to_rgb(utils.get_slice(image_data, slice))
            # get label data, class = fg_label
            label_path = label_fold_path + f'/{label_filename}'
            label_data = utils.load_label(label_path)
            label_data = utils.get_slice(label_data, slice)
            label_data = utils.select_label(label_data, fg_label)

            try:
                # get point in the center of ground truth
                if mode == "single_point_cog":
                    input_point = utils.find_center(label_data).reshape(1,2)
                    input_label = np.array([1])
                elif mode == "multi_points_rw":
                    input_point_1 = utils.find_center(label_data).reshape(1,2)
                    input_point_2 = utils.find_bg_random(label_data).reshape(1,2)
                    input_point = np.concatenate((input_point_1, input_point_2), axis=0)
                    input_label = np.array([1,0])
                elif mode == "multi_points_rr":
                    input_point_1 = utils.find_center(label_data).reshape(1,2)
                    input_point_2 = utils.find_fg_random(label_data).reshape(1,2)
                    input_point = np.concatenate((input_point_1, input_point_2), axis=0)
                    input_label = np.array([1,1])
                elif mode == "box":
                    input_box = utils.find_bounding_box(label_data)
                else:
                    raise Exception(f"mode `{mode}` is not implemented")

                # conduct segementation
                print(f'segmenting {image_filename}...')
                predictor.set_image(image)

                if mode == "single_point_cog":
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                elif 'multi_points' in mode:
                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )

                elif mode == "box":
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )


                # save the masks
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if i == 0 and slice % 80 == 0:
                        plt.figure(figsize=(10,10))
                        plt.imshow(image)
                        utils.show_mask(mask, plt.gca())
                        if 'point' in mode:
                            utils.show_points(input_point, input_label, plt.gca())
                        if mode == 'box':
                            utils.show_box(input_box, plt.gca())
                        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                        plt.axis('off')
                        plt.savefig(f"../output/{mode}/{fg_label}.{data_info['labels'][str(fg_label)]}/{image_filename}_slice={slice}_{i+1}.png")

                    # calculate dice score for each mask
                    dice_score = utils.dice_score(pred = mask, truth = label_data)
                    dice_result[f"slice_{slice}"].append(dice_score)
                    print(f"dice score of mask{i}: {dice_score}")

            except ValueError:
                print(f"empty slice: image: {image_filename}, slice: {slice}")
                dice_result[f"slice_{slice}"] = None

        dice_results.append(dice_result)

    with open(f"../output/{mode}/{fg_label}.{data_info['labels'][str(fg_label)]}/dice_score.json", 'w') as f:
        json.dump(dice_results, f)

# single_point: center of ground truth
mode = 'single_point_cog'

# open data_info_json
with open('../dataset/dataset_0.json', 'r') as f:
    data_info = json.load(f)

# slice_step
slice_step = 1

dice_results = []
for fg_label in range(1,14):
    print(f"looking for {fg_label}.{data_info['labels'][str(fg_label)]}...")
    experiment(mode=mode, fg_label=fg_label, slice_step=slice_step)