import numpy as np
import itertools
import random
from functools import reduce
import scipy.stats

def stoch_seed_sprinkle(grid, coverage=1):
    r, c = np.indices(grid.shape)
    ps = np.asarray([(r, c) for r, c in zip(r.reshape(-1), c.reshape(-1))])
    lps = len(ps)
    return ps[random.sample(range(lps), int(coverage * lps))]

def valid_point(r, c, rl, cl):
    return r > -1 and r < rl and c > -1 and c < cl

def eight_conn(p: tuple, grid, fc):
    return neighborhood(p, grid, list(itertools.product((0, 1, -1), repeat=2)), fc)

def four_conn(p: tuple, grid):
    return neighborhood(p, gird, ((0, 1), (1, 0), (0, -1), (-1, 0)))

def neighborhood(p: tuple, grid, nbrhood, fc=lambda x: True):
    rl, cl = grid.shape
    return {(r, c) for r, c in (np.asarray(nbrhood) + p) \
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

def list_of_objects(obj_coh):
    return reduce(lambda l1, l2: l1 + l2, [[{'points': o, 'color': pv} for o in objs] \
                                    for pv, objs in obj_coh.items()])

def object_cohesion(in_grid):
    grid = np.asarray(in_grid, dtype=np.uint8)
    seeds = stoch_seed_sprinkle(grid)
    regs = merge_regions(grow_regions(regions(grid, seeds), grid))
    prev_regs = {}
    while prev_regs != regs:
        prev_regs = regs
        regs = merge_regions(grow_regions(regs, grid))

    #invariant: np.prod(grid) == sum of pixels in regs
    return regs

def pixel_count(obj_coh: dict):
    return {pv: sum([len(o) for o in objs]) for pv, objs in obj_coh.items()}

def num_pixels(obj_coh: dict):
    return sum(pixel_count(obj_coh).values())

def pixel_percent(obj_coh: dict):
    npx = num_pixels(obj_coh)
    return {pv: sum([len(o) for o in objs]) / npx for pv, objs in obj_coh.items()} \
        if npx > 0 else {pv: 0 for pv, _ in obj_coh.items()}

def pixel_count_desc(pixel_count: dict):
    return sorted(pixel_count.items(), key=lambda t: t[1], reverse=True)

def num_objs(obj_coh: dict):
    return sum([len(objs) for _, objs in obj_coh.items()])

def obj_ratio_per_color(obj_coh: dict):
    pr = {pv: len(regs) / sum([len(r) for r in regs]) for pv, regs in obj_coh.items()}
    return sorted(pr.items(), key=lambda t: t[1])

def edge_points(grid, obj):
    ef = max, lambda t: t[1]
    wf = min, lambda t: t[1]
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
    tl = intersect(width, height, n, 0, w)
    bl = intersect(width, height, s, 0, w)
    br = intersect(width, height, s, 0, e)
    tr = intersect(width, height, n, 0, e)
    return (tl, bl, br, tr)

def rectangle_overlay(grid, obj):
    tl, _, br, _ = corners(grid, obj)
    rectangle = frozenset(reduce(lambda l1, l2: l1 + l2, \
                               [[(r, c) for c in range(tl[1], br[1] + 1)] \
                                              for r in range(tl[0], br[0] + 1)]))
    return rectangle

def rect_overlay_score(grid, obj_list: list):
    for obj in obj_list:
        score = len(obj['points']) / len(rectangle_overlay(grid, obj['points']))
        obj.update({'rect_overlay': score})

    return obj_list

def shift_object(obj, gr, gc, xs, ys):
    shift = frozenset({(x + xs, y + ys) for x, y in obj if valid_point(x + xs, y + ys, gr, gc)})
    return shift if len(shift) == len(obj) else obj

def shift_object_to_border(obj, gr, gc, shift_func):
    prev = obj
    curr = shift_func(obj, gr, gc)
    while prev != curr:
        prev = curr
        curr = shift_func(curr, gr, gc)
    return curr

def shift_object_left(obj, gr, gc):
    return shift_object(obj, gr, gc, 0, -1)

def shift_object_up(obj, gr, gc):
    return shift_object(obj, gr, gc, -1, 0)

def shift_object_top_left(obj, gr, gc):
    left = shift_object_to_border(obj, gr, gc, shift_object_left)
    return shift_object_to_border(left, gr, gc, shift_object_up)

def object_equals(o1, o2, gr, gc):
    return len(o1) == len(o2) and \
        shift_object_top_left(o1, gr, gc) == shift_object_top_left(o2, gr, gc)

def remove_background(obj_coh: dict, bg=0):
    return {pv: obj_coh[pv] for pv in set(obj_coh.keys()).difference({bg})}

def group_objects(obj_coh: dict, gr, gc):
    nbg_oc = remove_background(obj_coh)
    if len(nbg_oc) == 0:
        return {}
    else:
        obj_list = reduce(lambda l1, l2: l1 + l2, 
                          [[(pv, o, shift_object_top_left(o, gr, gc)) for o in objs] \
                           for pv, objs in nbg_oc.items()])
        sobj = sorted(obj_list, key=lambda t: t[2])
        return {k: tuple([(e[0], e[1]) for e in g]) \
                for k, g in itertools.groupby(sobj, key=lambda t: t[2])}

def object_distance(o1: frozenset, o2: frozenset):
    """objects must be distinct and in each others 4 connected path"""
    row, col = lambda t: t[0], lambda t: t[1]
    dict_from_gs = lambda gs, sf: {k: sorted(g, key=sf) for k, g in gs}
    block_dist = lambda o1, o2, ax: \
                 min(abs(o1[0][ax] - o2[-1][ax]), abs(o1[-1][ax] - o2[0][ax])) - 1

    o1_rgs = dict_from_gs(itertools.groupby(sorted(o1, key=row), key=row), col)
    o2_rgs = dict_from_gs(itertools.groupby(sorted(o2, key=row), key=row), col)
    o1_cgs = dict_from_gs(itertools.groupby(sorted(o1, key=col), key=col), row)
    o2_cgs = dict_from_gs(itertools.groupby(sorted(o2, key=col), key=col), row)

    ri, ci = frozenset(o1_rgs.keys()).intersection(o2_rgs.keys()), \
             frozenset(o1_cgs.keys()).intersection(o2_cgs.keys())
    inters = [(o1p, o2p, ks, diff_ax) for o1p, o2p, ks, diff_ax in \
              ((o1_rgs, o2_rgs, ri, 1), (o1_cgs, o2_cgs, ci, 0)) if ks]
    
    dist = min({ax: min({k: block_dist(o1_pix[k], o2_pix[k], ax) for k in int_ks}.values()) \
                 for o1_pix, o2_pix, int_ks, ax in inters}.values()) if inters else 0
    return dist

def objects_by_distance(obj_piece, objs: set):
    last = lambda t: t[1]
    dists = sorted([(o, object_distance(obj_piece, o)) for o in objs], key=last)
    defined_dists = [(o, d) for o, d in dists if d > 0]
    comp_dists = defined_dists if defined_dists else dists
    return (dists, min(comp_dists, key=last), max(comp_dists, key=last))

def sparse_object_cohesion(obj_coh: dict, rm_bgrd_func=remove_background):

    def merge_pieces(select_func, objs, merged, dists, deviations=2):
        if objs == set():
            return (merged, dists)
        elif len(objs) == 1:
            return (merged.union(objs), dists)
        else:
            first = select_func(objs)
            rest = objs.difference({first})
            _, no, _ = objects_by_distance(first, rest)
            nearest, d = no
            ds = dists + [d]

            if d == 0 or (len(set(ds)) > 1 and scipy.stats.zscore(ds)[-1] > deviations):
                return merge_pieces(select_func, rest, merged.union({first}), dists)
            else:
                return merge_pieces(select_func, rest.difference({nearest}), 
                                    merged.union({first.union(nearest)}), [d] + dists)

    def merge(objs):
        select_obj = lambda objs: frozenset(sorted([sorted(o) for o in objs])[0])
        prev_mrg = None
        curr_mrg, ds = merge_pieces(select_obj, objs, set(), [])
        while prev_mrg != curr_mrg:
            prev_mrg = curr_mrg
            curr_mrg, ds = merge_pieces(select_obj, prev_mrg, set(), ds)
        return curr_mrg

    return {pv: merge(objs) for pv, objs in rm_bgrd_func(obj_coh).items()}
