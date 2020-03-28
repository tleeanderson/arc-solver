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

    return {frozenset(t) for t in new_regs}

def merge_regions(all_regs):
    return {pv: merge(regs) for pv, regs in all_regs.items()}

def object_cohesion(in_grid):
    grid = np.asarray(in_grid, dtype=np.uint8)
    seeds = stoch_seed_sprinkle(grid)
    regs = merge_regions(grow_regions(regions(grid, seeds), grid))
    prev_regs = {}
    while prev_regs != regs:
        prev_regs = regs
        regs = merge_regions(grow_regions(regs, grid))

    #invariant: np.prod(grid) == sum of pixels in regs
    return regs, np.prod(grid.shape)

def obj_ratio_per_color(obj_coh):
    pv_regs, _ = obj_coh
    pr = {pv: len(regs) / sum([len(r) for r in regs]) for pv, regs in pv_regs.items()}
    return sorted(pr.items(), key=lambda t: t[1])

def edge_points(grid, obj):
    ef = min, lambda t: t[1]
    wf = max, lambda t: t[1]
    nf = min, lambda t: t[0]
    sf = max, lambda t: t[0]
    east, west, north, south = [fs[0](obj, key=fs[1]) for fs in (ef, wf, nf, sf)]

    return (north, south, east, west)

def intersect(gw, gh, p1, p1_axis, p2):
    p2_axis = (not p1_axis) * 1
    consts = (p1[p1_axis], p2[p2_axis]) if p1_axis == 0 else (p2[p2_axis], p1[p1_axis])
    h = {(consts[0], n) for n in range(gw)}
    v = {(n, consts[1]) for n in range(gh)}

    #invariant: |h.intersection(v)| == 1
    p = tuple(h.intersection(v))
    return p[0] if len(p) > 0 else set()

def corners(grid, obj):
    n, s, e, w = edge_points(grid, obj)
    width, height = grid.shape[1], grid.shape[0]
    tl = intersect(width, height, n, 0, e)
    bl = intersect(width, height, s, 0, e)
    br = intersect(width, height, s, 0, w)
    tr = intersect(width, height, n, 0, w)

    return (tl, bl, br, tr)

def rectangle_overlay(grid, obj):
    tl, _, br, _ = corners(grid, obj)
    rectangle = frozenset(reduce(lambda l1, l2: l1 + l2, \
                               [[(r, c) for c in range(tl[1], br[1] + 1)] \
                                              for r in range(tl[0], br[0] + 1)]))
    return rectangle

def rect_overlay_score(grid, obj_coh):
    oc, _ = obj_coh
    return {pv: [len(r) / len(rectangle_overlay(data, r)) for r in regs] \
            for pv, regs in oc.items()}
