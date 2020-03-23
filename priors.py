import numpy as np
from itertools import product
import random
from functools import reduce

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
