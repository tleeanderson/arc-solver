import priors
import json
from matplotlib import colors, pyplot
import numpy as np
import utils

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

def color_output(obj_coh: dict, grid: np.array):
    nbg = priors.remove_background(obj_coh)
    new_grid = np.copy(grid)
    colors = [c if c < 10 else (c % 10) + 1 \
              for c in range(1, sum([len(objs) for _, objs in nbg.items()]) + 1)]
    i = 0
    for _, objs in nbg.items():
        for o in objs:
            rs, cs = utils.pairs_to_indicies(o)
            new_grid[rs, cs] = colors[i]
            i += 1

    return new_grid

def sample_split_image():    
    with open('../ARC/data/training/4612dd53.json', 'r') as f:
        tasks = json.load(f)
        train_in = [np.asarray(t['input']) for t in tasks['train']]
        nt = np.zeros((13, 13))
        nt[:11, :13] = train_in[1]
        img = np.concatenate((nt, train_in[2]), axis=1)
        return img, priors.object_cohesion(img)

def four_image():
    ti, _ = sample_split_image()
    r, c = ti.shape
    four_input = np.stack((ti, ti))
    four_input = four_input.reshape((r * four_input.shape[0], c))
    sparse_coh = priors.sparse_object_cohesion(priors.object_cohesion(four_input))

    color_image = np.zeros(four_input.shape)
    color_image = color_output(sparse_coh, color_image)
    
    b_ind, r_ind, g_ind, y_ind = [np.where(color_image == c) for c in (1, 2, 3, 4)]
    for col, inds in enumerate((b_ind, r_ind, g_ind, y_ind)):
        color_image[inds[0], inds[1]] = 0

    b_ind, r_ind = [(rs, cs - 4) for rs, cs in (b_ind, r_ind)]
    g_ind, r_ind = [(rs - 4, cs) for rs, cs in (g_ind, r_ind)]
    r_ind = (r_ind[0] + 1, r_ind[1])
    for col, inds in enumerate((b_ind, r_ind, g_ind, y_ind)):
        color_image[inds[0], inds[1]] = 1

    return color_image

def sample_image(task, ind=0, tr_tst='train'):
    with open("../ARC/data/training/{}".format(task), 'r') as f:
        tasks = json.load(f)
        in_data = tasks[tr_tst][ind]['input']
        out_data = tasks[tr_tst][ind]['output']
        return {'input': (np.asarray(in_data, dtype=np.uint8), priors.object_cohesion(in_data)),
                'output': (np.asarray(out_data, dtype=np.uint8), priors.object_cohesion(out_data))}

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

def test_data(width, height, num_objs, background=0):
    data = np.random.randint(0, 10, width*height).reshape((width, height)) if background == 1 \
           else np.zeros((width, height))
    colors = np.random.choice(10, num_objs, replace=False)

    for c in colors:
        sr = np.random.randint(1, height // 2)
        er = np.random.randint(height // 2 + 1, height)
        sc = np.random.randint(1, width // 2)
        ec = np.random.randint(width // 2 + 1, width)
        data[sr:er, sc:ec] = c

    return data
