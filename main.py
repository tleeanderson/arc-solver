import dev_funcs as dev_funcs
import time
import optimizer
import cache
import functools
import interpreter as intr
import dsl.base as db
import argparse
import os.path
import glob
import utils
import numpy as np
import eval_func

ARC_PATH = '/home/tanderson/git/ARC/'
ARC_TRAIN = 'data/training'

def run_optimizer(prob_id, train_ins, train_outs, task_data: dict, eval_func):
    start = time.time()
    program = optimizer.general_program(train_ins, train_outs, eval_func, prob_id)
    end = time.time()
    et = round((end-start)*1000)

    return program, et

def find_program(prob_id, eval_func, task_data):
    train_ins, train_outs = utils.in_out_images(task_data)
    program, et = run_optimizer(prob_id, train_ins, train_outs, task_data, eval_func)
    return program, et

def test_program(program, eval_func, task_data):
    test_ins, test_outs = utils.in_out_images(task_data)
    mem_scores = [(intr.eval_no_cache(i, program, functools.partial(eval_func, o), 
                                     eval_toks=(db.sync_mod_objs, db.output,)), o) \
                  for i, o in zip(test_ins, test_outs)]
    outs = [(pred, gt, np.all(pred == gt)) for pred, gt in [(np.ndarray.tolist(ms[0][db.OUTPUT]), o) \
                                                    for ms, o in mem_scores]]
    return [t for _, _, t in outs]

def solve_tasks(tasks, loud=False):
    ets, answers, i = [], {}, 0
    for task, task_data in tasks.items():
        try:
            i += 1
            ef = eval_func.percent_not_matching
            program, t = find_program(task, ef, task_data['train'])
            result = test_program(program, ef, task_data['test'])

            ets.append(t)
            if task in answers:
                print("duplicate task: {}".format(task))
            else:
                answers[task] = (result, all(result))

            if loud or i % 10 == 0:
                correct = sum([1 for _, c in answers.values() if c])
                print("i: {}, task: {}, num_correct: {}, avg_et: {}"\
                      .format(i, task, correct, np.average(ets)))

        except Exception as e:
            print("Exception: {}, task: {}".format(e, task))

def read_tasks(task_paths):
    return {os.path.basename(tp): utils.read_file(tp) for tp in task_paths}

def tasks(task_path, args):
    if args.task_pattern:
        return glob.glob(os.path.join(task_path, args.task_pattern))
    elif args.tasks:
        return [os.path.join(task_path, t) for t in args.tasks]
    else:
        print("you must specify either --task-pattern or --tasks")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--task_pattern', required=False)
    parser.add_argument('-ts', '--tasks', nargs='+', required=False)
    parser.add_argument('-a', '--arc-path', default=ARC_PATH)
    parser.add_argument('-dp', '--data-path', default=ARC_TRAIN)
    parser.add_argument('-l', '--loud', default=False, action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    task_path = os.path.join(*[args.arc_path, args.data_path])
    tasks = read_tasks(tasks(task_path, args))
    solve_tasks(tasks, args.loud)
