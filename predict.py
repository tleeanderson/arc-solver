import priors
import utils
import argparse
import os.path as path
import json

ARC_PATH = '/home/tanderson/git/ARC/'
ARC_TRAIN = 'data/training'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--input-tasks', required=True, nargs='+')
    parser.add_argument('-a', '--arc-path', default=ARC_PATH)

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

def test_object_cohesion(tasks, task_path):
    for t in tasks:
        data = read_file(path.join(task_path, t))
        tests = [data['train'][0]['input'], data['train'][0]['output']]
        for t in tests:
            oc = priors.object_cohesion(t)
            for pv, regs in sorted(oc.items(), key=lambda t: t[0]):
                print("pixel_value: {}, num_regions: {}".format(pv, len(regs)))
            print()
            #images_from_object_cohesion(oc, np.asarray(t).shape)


if __name__ == '__main__':
    args = parse_args()
    task_path = path.join(*[ARC_PATH, ARC_TRAIN])
    test_object_cohesion(args.input_tasks, task_path)
