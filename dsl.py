import priors
import numpy as np
import utils

GRID = 'grid'
VAL = 'val'
VAL2 = 'val2'

def object_cohesion(in_grid):
    return {GRID: in_grid, VAL: priors.object_cohesion(in_grid)}

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


