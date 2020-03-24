import priors
import utils
import argparse
import os.path as path
import json
import glob
import numpy as np
from functools import partial

ARC_PATH = '/home/tanderson/git/ARC/'
ARC_TRAIN = 'data/training'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--task_pattern', required=True)
    parser.add_argument('-a', '--arc-path', default=ARC_PATH)
    parser.add_argument('-dp', '--data-path', default=ARC_TRAIN)

    return parser.parse_args()

def read_file(path):
    with open(path, 'r') as f:
        return json.load(f)
    return None

def images_from_object_cohesion(oc, grid_shape):
    for pv, regs in oc.items():
        for rg in regs:
            img = np.zeros(grid_shape)
            for r, c in list(rg):
                img[r, c] = pv
            print("display for pv: {}, reg: {}".format(pv, rg))
            utils.display(img)

def read_tasks(task_paths):
    return {path.basename(tp): read_file(tp) for tp in task_paths}

def predict(tasks, program):
    c = 0
    for task, task_data in tasks.items():
        c += 1
        print("task: {}, num: {}".format(task, c))
        for ds, data in task_data.items():
            for task_demo in data:
                for f in program:
                    o1 = f(task_demo['input'])
                    o2 = f(task_demo['output'])
                    print("input: {}\noutput:{}".format(o1, o2))
        print("")

if __name__ == '__main__':
    args = parse_args()
    task_path = path.join(*[args.arc_path, args.data_path])
    tasks = read_tasks(glob.glob(path.join(task_path, args.task_pattern)))
    predict(tasks, program=[partial(utils.func_reduce,
                                    [priors.object_cohesion, priors.obj_ratio_per_color])])
