from matplotlib import colors, pyplot
import numpy as np

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

def func_reduce(funcs, iv):
    if len(funcs) <= 0:
        return iv
    return func_reduce(funcs[1:], funcs[0](iv))

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
