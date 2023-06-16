import os

if os.fork():
    os.system('python task2_tune.py --device cuda:0 --bounding_box_prompt --box_margin 0 --lr 0.00001 --valid_fold 0 > results/task2_fine_tune/lr=1e-5_valid_0.txt')
elif os.fork():
    os.system('python task2_tune.py --device cuda:0 --bounding_box_prompt --box_margin 0 --lr 0.00001 --valid_fold 1 > results/task2_fine_tune/lr=1e-5_valid_1.txt')
elif os.fork():
    os.system('python task2_tune.py --device cuda:1 --bounding_box_prompt --box_margin 0 --lr 0.00001 --valid_fold 2 > results/task2_fine_tune/lr=1e-5_valid_2.txt')
elif os.fork():
    os.system('python task2_tune.py --device cuda:1 --bounding_box_prompt --box_margin 0 --lr 0.00001 --valid_fold 3 > results/task2_fine_tune/lr=1e-5_valid_3.txt')
elif os.fork():
    os.system('python task2_tune.py --device cuda:2 --bounding_box_prompt --box_margin 0 --lr 0.00001 --valid_fold 4 > results/task2_fine_tune/lr=1e-5_valid_4.txt')
elif os.fork():
    os.system('python task2_tune.py --device cuda:2 --bounding_box_prompt --box_margin 0 --lr 0.00001 --valid_fold 5 > results/task2_fine_tune/lr=1e-5_valid_5.txt')
else:
    exit()

