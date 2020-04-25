import priors
import utils
import argparse
import os.path as path
import json
import glob
import numpy as np
from functools import partial
import priors
import pattern_recognition as pr
import inspect

ARC_PATH = '/home/tanderson/git/ARC/'
ARC_TRAIN = 'data/training'

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

def predict(tasks):
    answers = {}
    c = 0
    pattern_funcs = file_funcs(pr)
    for task, task_data in tasks.items():
        train_input = [td['input'] for td in task_data['train']]
        train_output = [td['output'] for td in task_data['train']]

        programs = [prg for prg in [(pf.__name__, pf(train_input, train_output)) for pf in pattern_funcs] if prg[1]]
        if programs:
            test_input = [td['input'] for td in task_data['test']]
            test_output = [td['output'] for td in task_data['test']]

            result = [(pf, all([np.all(utils.func_reduce(p, ti) == to) for ti, to \
                           in zip(test_input, test_output)])) for pf, p in programs]
            responses = any([tf for pf, tf in result])
            answers[task] = (len(programs), responses, result)

        c += 1
        if c % 10 == 0:
            print("On task {} of {}".format(c, len(tasks)))

    for t, a in sorted(answers.items(), key=lambda t: t[1][2][0]):
        num_progs, res, pf_tf = a
        print("pf_tf: {}, task: {}".format(pf_tf, t))

    corr = sum([r for _, r, _ in answers.values()])
    print("Total answers: {}, correct: {}, incorrect: {}"\
              .format(len(answers), corr, len(answers) - corr))

def tasks(task_path, args):
    if args.task_pattern:
        return glob.glob(path.join(task_path, args.task_pattern))
    elif args.tasks:
        return [path.join(task_path, t) for t in args.tasks]
    else:
        print("you must specify either --task-pattern or --tasks")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--task_pattern', required=False)
    parser.add_argument('-ts', '--tasks', nargs='+', required=False)
    parser.add_argument('-a', '--arc-path', default=ARC_PATH)
    parser.add_argument('-dp', '--data-path', default=ARC_TRAIN)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    task_path = path.join(*[args.arc_path, args.data_path])
    tasks = read_tasks(tasks(task_path, args))
    predict(tasks)
