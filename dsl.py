import priors
import numpy as np
import utils

GRID = 'grid'
VAL = 'val'

def object_cohesion(in_grid):
    return {GRID: in_grid, VAL: priors.object_cohesion(in_grid)}

def majority_pixel(dat_map):
    in_grid = dat_map[GRID]
    oc = dat_map[VAL]
    return {GRID: in_grid, 
            VAL: priors.pixel_count_desc(priors.pixel_percent(oc))[0][0]}

def create_image(dat_map):
    in_grid = dat_map[GRID]
    pv = dat_map[VAL]
    img = np.zeros(np.asarray(in_grid).shape, dtype=np.uint8)
    img[:] = pv
    return np.ndarray.tolist(img)
