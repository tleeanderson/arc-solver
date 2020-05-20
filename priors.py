import numpy as np
import itertools
import random
import functools
import scipy.stats
import operator

def stoch_seed_sprinkle(grid, coverage=1):
    r, c = np.indices(grid.shape)
    ps = np.asarray([(r, c) for r, c in zip(r.reshape(-1), c.reshape(-1))])
    lps = len(ps)
    return ps[random.sample(range(lps), int(coverage * lps))]

def valid_point(r, c, rl, cl):
    return r > -1 and r < rl and c > -1 and c < cl

def neighborhood(p: tuple, grid, nbrhood, pix_func=lambda x: True):
    rl, cl = grid.shape
    return {(r, c) for r, c in (np.asarray(nbrhood) + p) \
            if valid_point(r, c, rl, cl) and pix_func(grid[r][c])}

def eight_conn(p: tuple, grid, constraint_func):
    return neighborhood(p, grid, list(itertools.product((0, 1, -1), repeat=2)), 
                        constraint_func)

def four_conn(p: tuple, grid, constraint_func):
    return neighborhood(p, grid, ((0, 1), (1, 0), (0, -1), (-1, 0)), 
                        constraint_func)

def grow(region, grid):
    result = set()
    for r, c in region:
        result = result.union(eight_conn((r, c), grid, lambda pix: pix == grid[r][c]))
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
        nr = functools.reduce(lambda s1, s2: s1.union(s2), unions) if unions \
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

def edge_points(obj):
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
    n, s, e, w = edge_points(obj)
    width, height = grid.shape[1], grid.shape[0]
    tl = intersect(width, height, n, 0, w)
    bl = intersect(width, height, s, 0, w)
    br = intersect(width, height, s, 0, e)
    tr = intersect(width, height, n, 0, e)
    return (tl, bl, br, tr)

def rectangle_overlay(grid, obj):
    tl, _, br, _ = corners(grid, obj)
    rectangle = frozenset(functools.reduce(lambda l1, l2: l1 + l2, \
                               [[(r, c) for c in range(tl[1], br[1] + 1)] \
                                              for r in range(tl[0], br[0] + 1)]))
    return rectangle

def rect_overlay_score(grid, obj_list: list):
    for obj in obj_list:
        score = len(obj['points']) / len(rectangle_overlay(grid, obj['points']))
        obj.update({'rect_overlay': score})

    return obj_list

def shift_object(obj, gr, gc, rs, cs):
    shift = frozenset({(r + rs, c + cs) for r, c in obj if valid_point(r + rs, c + cs, gr, gc)})
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

def shift_object_right(obj, gr, gc):
    return shift_object(obj, gr, gc, 0, 1)

def shift_object_up(obj, gr, gc):
    return shift_object(obj, gr, gc, -1, 0)

def shift_object_down(obj, gr, gc):
    return shift_object(obj, gr, gc, 1, 0)

def shift_object_top_left(obj, gr, gc):
    left = shift_object_to_border(obj, gr, gc, shift_object_left)
    return shift_object_to_border(left, gr, gc, shift_object_up)

def object_equals(o1, o2, gr, gc):
    return len(o1) == len(o2) and \
        shift_object_top_left(o1, gr, gc) == shift_object_top_left(o2, gr, gc)

def remove_background(obj_coh: dict, bg=0):
    return {pv: obj_coh[pv] for pv in set(obj_coh.keys()).difference({bg})}

def list_of_objects(obj_coh):
    nbg_coh = remove_background(obj_coh)
    all_os = functools.reduce(lambda l1, l2: l1 + l2, [[(pv, o) for o in objs] \
                                    for pv, objs in nbg_coh.items()], [])
    return [(pv, frozenset(o)) for pv, o \
            in sorted([(pv, sorted(o)) for pv, o in all_os], key=lambda t: t[1])]

def object_cohesion_from_list(obj_coh: list):
    return {k: set([pixs for _, pixs in g]) for k, g in \
            itertools.groupby(sorted(obj_coh, key=lambda t: t[0]), key=lambda t: t[0])}

def group_objects(obj_coh: dict, gr, gc):
    nbg_oc = remove_background(obj_coh)
    if len(nbg_oc) == 0:
        return {}
    else:
        obj_list = functools.reduce(lambda l1, l2: l1 + l2, 
                          [[(pv, o, shift_object_top_left(o, gr, gc)) for o in objs] \
                           for pv, objs in nbg_oc.items()])
        sobj = sorted(obj_list, key=lambda t: t[2])
        return {k: {pg: tuple(frozenset(so) for so in sorted([sorted(o) for _, o, _ in pos])) for pg, pos in \
                    itertools.groupby(sorted(g, key=lambda t: t[0]), key=lambda t: t[0])} \
                for k, g in itertools.groupby(sobj, key=lambda t: t[2])}


def points_between(p1, p2, ax):
    p1_first = range(p1[ax]+1, p2[ax])
    p2_first = range(p2[ax]+1, p1[ax])
    if p1_first or p2_first:
        rng = p1_first if p1_first else p2_first
        return frozenset([(v, p1[1]) for v in rng]) if ax == 0 \
            else frozenset([(p1[0], v) for v in rng])
    else:
        return []

def object_distance(o1: frozenset, o2: frozenset):
    """objects must be distinct and in each others 4 connected path"""
    row, col = lambda t: t[0], lambda t: t[1]
    dict_from_gs = lambda gs, sf: {k: sorted(g, key=sf) for k, g in gs}
    points = lambda o1, o2, ax: \
             min([(len(pb), pb) for pb in [points_between(arg1, arg2, ax) for arg1, arg2 \
                                       in ((o1[0], o2[-1]), (o1[-1], o2[0]))]], key=lambda t: t[0])

    o1_rgs = dict_from_gs(itertools.groupby(sorted(o1, key=row), key=row), col)
    o2_rgs = dict_from_gs(itertools.groupby(sorted(o2, key=row), key=row), col)
    o1_cgs = dict_from_gs(itertools.groupby(sorted(o1, key=col), key=col), row)
    o2_cgs = dict_from_gs(itertools.groupby(sorted(o2, key=col), key=col), row)

    ri, ci = frozenset(o1_rgs.keys()).intersection(o2_rgs.keys()), \
             frozenset(o1_cgs.keys()).intersection(o2_cgs.keys())
    inters = [(o1p, o2p, ks, diff_ax) for o1p, o2p, ks, diff_ax in \
              ((o1_rgs, o2_rgs, ri, 1), (o1_cgs, o2_cgs, ci, 0)) if ks]
    
    dist = min({ax: min({k: points(o1_pix[k], o2_pix[k], ax) for k in int_ks}.values(), key=lambda t: t[0]) \
                for o1_pix, o2_pix, int_ks, ax in inters}.values(), key=lambda t: t[0]) if inters else (0, [])

    return dist

def overlap_distance(o1: frozenset, o2: frozenset):
    return object_distance(o1, o2)[0] + len(o1.difference(o2))

def objects_by_distance(obj_piece, objs: set):
    ob_dist = lambda t: t[1][0]
    return {d: list(os) for d, os in itertools.groupby(sorted([(o, object_distance(obj_piece, o)) \
                                                     for o in objs], key=ob_dist), key=ob_dist)}

def sorted_distances(obj_dists: dict):
    defined_dists = sorted(obj_dists.keys() - {0})
    comp_dists = defined_dists if defined_dists else obj_dists.keys()
    return tuple(functools.reduce(lambda l1, l2: l1 + l2, [obj_dists[d] for d in comp_dists], []))

def sparse_object_cohesion(obj_coh: dict, rm_bgrd_func=remove_background, deviations=2):

    def merge_pieces(select_func, objs, merged, dists, deviations):
        if objs == set():
            return (merged, dists)
        elif len(objs) == 1:
            return (merged.union(objs), dists)
        else:
            first = select_func(objs)
            rest = objs.difference({first})
            no = sorted_distances(objects_by_distance(first, rest))[0]
            nearest, ps_d = no
            d, _ = ps_d
            ds = dists + [d]

            if d == 0 or (len(set(ds)) > 1 and scipy.stats.zscore(ds)[-1] > deviations):
                return merge_pieces(select_func, rest, merged.union({first}), dists, deviations)
            else:
                return merge_pieces(select_func, rest.difference({nearest}), 
                                    merged.union({first.union(nearest)}), [d] + dists, deviations)

    def merge(objs, deviations):
        select_obj = lambda objs: frozenset(sorted([sorted(o) for o in objs])[0])
        prev_mrg = None
        curr_mrg, ds = merge_pieces(select_obj, objs, set(), [], deviations=deviations)
        while prev_mrg != curr_mrg:
            prev_mrg = curr_mrg
            curr_mrg, ds = merge_pieces(select_obj, prev_mrg, set(), ds, deviations=deviations)
        return curr_mrg

    return {pv: merge(objs, deviations) for pv, objs in rm_bgrd_func(obj_coh).items()}

def end_points(obj):
    n, s, e, w = edge_points(obj)
    point_axis = ((n, 0, lambda v: n[0] > v),
                  (s, 0, lambda v: s[0] < v),
                  (e, 1, lambda v: e[1] < v), 
                  (w, 1, lambda v: w[1] > v))

    return {(p, ax, f) for p, ax, f in point_axis if sum([1 for op in obj.difference({p}) \
                                                          if op[ax] == p[ax]]) == 0}

def gaps_by_endpoint(obj_piece, objs):
    sorted_dists = sorted_distances(objects_by_distance(obj_piece, objs))
    end_ps = end_points(obj_piece)
    first = lambda t: t[0]
    all_gaps = sorted(functools.reduce(lambda s1, s2: s1 + s2, 
                                       [[(ep, d_ps[0], d_ps[1]) for ob, d_ps in sorted_dists \
                                         if all([cf(op[ax]) for op in ob])] for ep, ax, cf in end_ps], []), key=first)
    def_gaps = {k: sorted(g, key=lambda t: t[1]) for k, g in itertools.groupby(all_gaps, key=first)}
    undef_gaps = {(ep, ax, f) for ep, ax, f in end_ps if ep not in def_gaps.keys()}
    return def_gaps, undef_gaps

def object_bounds(obj: set):
    sing_obj = functools.reduce(lambda s1, s2: s1.union(s2), obj)
    n, s, e, w = edge_points(sing_obj)
    return (range(n[0], s[0]+1), range(w[1], e[1]+1))

def pix_axis(pix: tuple, obj: set):
    pr, pc = pix
    sing_obj = functools.reduce(lambda s1, s2: s1.union(s2), obj)
    n, s, e, w = edge_points(sing_obj)
    rs = range(n[0], s[0]+1)
    cs = range(w[1], e[1]+1)

    return max([(len(pixs.intersection(sing_obj)), ax) for pixs, ax in \
                (({(pr, c) for c in cs}, 0), ({(r, pc) for r in rs}, 1))], key=lambda t: t[0])[1]

def gaps(obj):

    def traverse_obj(obj, obj_bs, in_between, traveled):
        if obj.difference(traveled) == set():
            return in_between
        else:
            first = next(iter(obj.difference(traveled)))
            rest = obj.difference({first})
            gpe, undef_gpe = gaps_by_endpoint(first, rest)
            between = frozenset()
            if gpe:
                if len(first) == 1:
                    p = next(iter(first))
                    pa = pix_axis(p, obj)
                    between = [ps for _, _, ps in gpe[next(iter(gpe))] if all([bp[pa] == p[pa] for bp in ps])]
                else:
                    bet = {lis[0][2] for ep, lis in gpe.items()}
                    len_bs = [len(b) for b in obj_bs]
                    enum_args = lambda ax, p: ([p[not ax]]*len_bs[ax], obj_bs[ax]) if ax \
                                else (obj_bs[ax], [p[not ax]]*len_bs[ax])
                    eps = {frozenset(functools.reduce(lambda s1, s2: s1.union(s2),
                                           [{op for op in zip(*enum_args(ax, p)) if f(op[ax])} \
                                            for p, ax, f in undef_gpe]) if undef_gpe else {})}
                    between = bet.union(eps)

            ib_ps = functools.reduce(lambda s1, s2: s1.union(s2), between, frozenset())
            return traverse_obj(obj, obj_bs, in_between.union(ib_ps), traveled.union({first}))

    bounds = object_bounds(obj)
    sing_obj = functools.reduce(lambda s1, s2: s1.union(s2), obj)
    corners = {cp for cp in ((min(bounds[0]), min(bounds[1])), (max(bounds[0]), min(bounds[1])), 
                             (min(bounds[0]), max(bounds[1])), (max(bounds[0]), max(bounds[1]))) \
               if cp not in sing_obj}
    gap_points = traverse_obj(obj, bounds, frozenset(), frozenset())
    return gap_points.union(corners).difference(sing_obj)

def object_gaps(grp_obj_coh: dict):
    nbg_coh = remove_background(grp_obj_coh)
    return {pv: {(o, gaps(o)) for o in objs} for pv, objs in nbg_coh.items()}

def group_pieces(obj: set, obj_pieces: set):
    pcs = {piece for piece in obj_pieces if all([p in obj for p in piece])}
    pcs_obj = functools.reduce(lambda s1, s2: s1.union(s2), pcs)
    return frozenset(pcs) if pcs_obj == obj else None

def group_object_cohesion(object_coh: dict, sparse_object_coh: dict):
    #invariant: group_pieces != None because sparse_object_coh groups object_coh
    return {pv: {group_pieces(o, object_coh[pv]) for o in objs} for pv, objs in sparse_object_coh.items()}
