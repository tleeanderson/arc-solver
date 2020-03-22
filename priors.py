import argparse
import os.path as path
import json
import numpy as np
from matplotlib import colors, pyplot
from itertools import product
import random
from functools import reduce

ARC_PATH = '/home/tanderson/git/ARC/'
ARC_TRAIN = 'data/training'

BLACK = 'k'
BLUE = 'b'
RED = 'r'
GREEN = 'g'
YELLOW = 'y'
GRAY = '0.75'
PINK = '#FF69B4'
ORANGE = '#FFA500'
CYAN = '#00FFFF'
MAROON = '#800000'

COLOR_RANGE = range(10)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--input-tasks', required=True, nargs='+')
    parser.add_argument('-a', '--arc-path', default=ARC_PATH)

    return parser.parse_args()

def read_file(path):
    with open(path, 'r') as f:
        return json.load(f)
    return None

def display(grid):
    np_g = np.array(grid)
    r, c = np_g.shape[0], np_g.shape[1]
    cmap = colors.ListedColormap([BLACK, BLUE, RED, GREEN, YELLOW, GRAY, 
                                  PINK, ORANGE, CYAN, MAROON])
    bounds = range(10)
    np_bs = np.array(bounds)
    norm = colors.BoundaryNorm(bounds, cmap.N - 1)
    fig, ax = pyplot.subplots()
    ax.matshow(np_g, cmap=cmap, norm=norm)
    ax.grid(linewidth=2, color='0.5')
    ax.set_xticks(np.arange(-.5, c, 1))
    ax.set_yticks(np.arange(-.5, r, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    pyplot.show()

def stoch_seed_sprinkle(grid, coverage=1):
    r, c = np.indices(grid.shape)
    ps = np.asarray([(r, c) for r, c in zip(r.reshape(-1), c.reshape(-1))])
    lps = len(ps)
    return ps[random.sample(range(lps), int(coverage * lps))]

def valid_point(r, c, rl, cl):
    return r > -1 and r < rl and c > -1 and c < cl

def eight_conn(p: tuple, grid, fc=lambda x: True):
    rl, cl = grid.shape
    return {(r, c) for r, c in (np.asarray(list(product((0, 1, -1), repeat=2))) + p) \
            if valid_point(r, c, rl, cl) and fc(grid[r][c])}

def grow(region, grid):
    result = set()
    for r, c in region:
        result = result.union(eight_conn((r, c), grid, lambda x: x == grid[r][c]))
    return result

def regions(grid, seeds):
    regions = {}
    for r, c in seeds:
        pv = grid[r][c]
        regions[pv] = regions[pv] + [{(r, c)}] if pv in regions else [{(r, c)}]
    return regions

def grow_regions(regions, grid):
    return {pv: [grow(r, grid) for r in regs] for pv, regs in regions.items()}

def merge(regions):
    lr = len(regions)
    new_regs = set()
    for i in range(lr):
        unions = []
        for j in range(i+1, lr):
            inter = regions[i].intersection(regions[j])
            if inter != set():
                unions.append(regions[i].union(regions[j]))
        nr = reduce(lambda s1, s2: s1.union(s2), unions) if unions \
             else regions[i]
        if all([nr.intersection(set(t)) == set() for t in new_regs]):
            new_regs.add(tuple(nr))

    #ordering is arbitrary, but necessary for equality
    return sorted([set(t) for t in new_regs], key=lambda s: list(s)[0][0])

def merge_regions(all_regs):
    return {pv: merge(regs) for pv, regs in all_regs.items()}

def object_cohesion(in_grid):
    grid = np.asarray(in_grid)
    seeds = stoch_seed_sprinkle(grid)
    regs = merge_regions(grow_regions(regions(grid, seeds), grid))
    prev_regs = None
    while prev_regs != regs:
        prev_regs = regs
        regs = merge_regions(grow_regions(regs, grid))
    return regs

def images_from_object_cohesion(oc, grid_shape):
    for pv, regs in oc.items():
        for rg in regs:
            img = np.zeros(grid_shape)
            for r, c in list(rg):
                img[r, c] = pv
            print("display for pv: {}, reg: {}".format(pv, rg))
            display(img)

def test_object_cohesion(tasks, task_path):
    for t in tasks:
        data = read_file(path.join(task_path, t))
        tests = [data['train'][0]['input'], data['train'][0]['output']]
        for t in tests:
            oc = object_cohesion(t)
            for pv, regs in sorted(oc.items(), key=lambda t: t[0]):
                print("pixel_value: {}, num_regions: {}".format(pv, len(regs)))
            print()
            #images_from_object_cohesion(oc, np.asarray(t).shape)

if __name__ == '__main__':
    args = parse_args()
    task_path = path.join(*[ARC_PATH, ARC_TRAIN])
    test_object_cohesion(args.input_tasks, task_path)
