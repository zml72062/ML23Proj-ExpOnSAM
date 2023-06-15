import os

if os.fork():
    os.system('python task1_infer.py --device cuda:0 --point_prompt random > results/random_point.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:0 --point_prompt center > results/center_point.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:0 --point_prompt center random > results/center_fg_point.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:1 --point_prompt center bg_random > results/center_bg_point.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:1 --point_prompt random random > results/two_random_point.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:1 --point_prompt random bg_random > results/bg_fg_random_point.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:2 --bounding_box_prompt --box_margin 0 > results/bounding_box_margin_0.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:2 --bounding_box_prompt --box_margin 50 > results/bounding_box_margin_50.txt')
elif os.fork():
    os.system('python task1_infer.py --device cuda:2 --bounding_box_prompt --box_margin 100 > results/bounding_box_margin_100.txt')


