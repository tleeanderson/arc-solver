import priors
import numpy as np
import utils
import interpreter_funcs as int_funcs

GRID = 'grid'
VAL = 'val'
VAL2 = 'val2'

def convert_value(in_grid):
    return {GRID: in_grid}

def object_cohesion(dat_map):
    in_grid = dat_map[GRID]
    return {GRID: in_grid, VAL: priors.object_cohesion(in_grid)}

def sparse_object_cohesion(dat_map):
    oc = dat_map[VAL]
    return {GRID: dat_map[GRID], VAL: priors.sparse_object_cohesion(oc)}

def sparse_and_object_cohesion(dat_map):
    in_grid = dat_map[GRID]
    oc = priors.object_cohesion(in_grid)
    devs = dat_map[VAL]
    return {GRID: in_grid, VAL: oc, VAL2: priors.sparse_object_cohesion(oc, deviations=devs)}

def group_object_cohesion(dat_map):
    oc = dat_map[VAL]
    sc = dat_map[VAL2]
    return {GRID: dat_map[GRID], VAL: priors.group_object_cohesion(oc, sc)}

def object_gaps(dat_map):
    gc = dat_map[VAL]
    return {GRID: dat_map[GRID], VAL: priors.object_gaps(gc)}

def group_objects(dat_map):
    in_grid = dat_map[GRID]
    r, c = np.asarray(in_grid).shape
    oc = dat_map[VAL]
    return {GRID: in_grid, 
            VAL: priors.group_objects(oc, r, c)}

def unique_object(dat_map):
    in_grid = dat_map[GRID]
    objs = dat_map[VAL]
    pv, obj = list(objs.values())[0][0]
    return {GRID: in_grid,
            VAL: (pv, list(obj))}

def unique_object_norm(dat_map):
    in_grid = dat_map[GRID]
    objs = dat_map[VAL]
    pv, _ = list(objs.values())[0][0]
    o = list(objs.keys())[0]
    return {GRID: in_grid,
            VAL: (pv, o)}

def majority_pixel(dat_map):
    in_grid = dat_map[GRID]
    oc = dat_map[VAL]
    return {GRID: in_grid, 
            VAL: priors.pixel_count_desc(priors.pixel_percent(oc))[0][0]}

def remove_background(dat_map):
    in_grid = dat_map[GRID]
    oc = dat_map[VAL]
    return {GRID: in_grid, 
            VAL: priors.remove_background(oc)}

def lookup_answer(dat_map):
    in_grid = dat_map[GRID]
    pv = dat_map[VAL]
    poss_ans = dat_map[VAL2]
    return {GRID: in_grid, 
            VAL: (poss_ans[pv][0], poss_ans[pv][1])}

def single_obj_image(dat_map):
    in_grid = dat_map[GRID]
    pv, obj = dat_map[VAL]
    o = list(obj)
    rs, cs = zip(*o)
    img = utils.smallest_enclosing_img((rs, cs))
    img[rs, cs] = pv
    return img

def create_image(dat_map):
    in_grid = dat_map[GRID]
    pv = dat_map[VAL]
    img = np.zeros(np.asarray(in_grid).shape, dtype=np.uint8)
    img[:] = pv
    return np.ndarray.tolist(img)

def color_gaps(dat_map):
    in_grid = dat_map[GRID]
    ogs = dat_map[VAL]
    gap_col = dat_map[VAL2]
    out = np.ndarray.tolist(utils.color_gaps(in_grid, gap_col, ogs))
    return out

