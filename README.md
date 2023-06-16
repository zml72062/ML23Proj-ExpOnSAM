# Experiments on Segment Anything

This repository is a copy of the [official code of Segment Anything Model](https://github.com/facebookresearch/segment-anything). We perform segmentation tasks on the [BTCV dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), using both the officially pretrained SAM model and our fine-tuned model.

Code for our experiments is available at `experiment/` directory. The raw data can be downloaded from the [link](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), and should be extracted to the `dataset/RawData/` directory, following the instructions in `dataset/RawData/extract.txt`.

## Run

### Task 1

To run Task 1 (inference of SAM on BTCV), execute
```
python task1_infer.py [--point_prompt <prompt1> <prompt2> ...]
                      [--bounding_box_prompt [--box_margin <margin>]]
```
Here, each of the `<prompt1>`, `<prompt2>`, ... takes one of `random`, `center`, or `bg_random`, referring to three different kinds of point prompts: a random foreground point, center point, or a random background point.

If `--bounding_box_prompt` is set, a bounding box surrounding the object to be segmented will be used as (additional) prompt. An optional argument `<margin>` can be set to leave a margin at four sides of the bounding box.

### Task 2

To run Task 2 (fine-tuning of SAM on BTCV), execute
```
python task2_tune.py [--point_prompt <prompt1> <prompt2> ...]
                     [--bounding_box_prompt [--box_margin <margin>]]
                     [--valid_fold <fold>]
```
The arguments `--point_prompt` and `--bounding_box_prompt` is the same as in Task 1. 

We use 6-fold cross validation in fine-tuning, i.e. 4 out of the 24 images in the training dataset will be used as validation set. Set `<fold>`, which should be a value within `[0, 5]`, to specify the fold treated as validation set (default `5`).

### Task 3

To run Task 3 (training SAM for segmentation with classification on BTCV), execute
```
python task3_class.py [--point_prompt <prompt1> <prompt2> ...]
                      [--bounding_box_prompt [--box_margin <margin>]]
                      [--grid_prompt [--grid_distance <distance>]]
                      [--valid_fold <fold>]
```
All arguments except `--grid_prompt` are the same as in Task 2. 

We support grid point prompt for Task 3. By setting `--grid_prompt`, and optionally setting `--grid_distance` to an integer, a square grid is generated on the 2D (axial) interface of the medical image. The grid points are then treated as prompts. The default value for `<distance>` is 16.