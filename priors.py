import argparse
import os.path as path
import json
import numpy as np
from matplotlib import colors, pyplot
from itertools import product

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
CYAN = 'c'
MAROON = 'm'

COLOR_RANGE = range(10)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input-task', required=True)
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
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = pyplot.subplots()
    ax.matshow(np_g, cmap=cmap, norm=norm)
    ax.grid(linewidth=2, color='0.5')
    ax.set_xticks(np.arange(-.5, c, 1))
    ax.set_yticks(np.arange(-.5, r, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    pyplot.show()

def stoch_seed_sprinkle(grid, coverage=0.5):
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
    return tuple(result)

def region_growing(grid, seeds):
    regions = {((r, c),): grid[r][c] for r, c in seeds}
    new_regions = {grow(r, grid): regions[r] for r in regions}
    return new_regions

if __name__ == '__main__':
    args = parse_args()
    task_path = path.join(*[ARC_PATH, ARC_TRAIN, args.input_task])
    data = read_file(task_path)

    display(data['train'][0]['input'])
