import numpy as np
import priors
import inspect
import json
import itertools
import functools

def read_file(path):
    with open(path, 'r') as f:
        return json.load(f)
    return None

def enumerate_funcs(func, arguments):
    perms = itertools.product(*[[(an, v) for v in avs] for an, avs in arguments.items()])
    return [functools.partial(func, **dict(p)) for p in perms]

def in_out_images(task_data):
    return zip(*[(np.asarray(m['input'], dtype=np.uint8), np.asarray(m['output'], dtype=np.uint8)) \
                 for m in task_data])

def pairs_to_indicies(obj):
    return zip(*list(obj))

def dict_paths(input_dict: dict):
    
    def dfs(inp, cp):
        if type(inp) != dict:
            return (cp,)
        else:
            ps = tuple()
            for k in inp:
                ps += dfs(inp[k], cp + (k,))
            return ps

    return dfs(input_dict, tuple())

def path_value(key_path: tuple, inp_dict: dict):
    ele = inp_dict
    for k in key_path:
        if type(ele) != dict or k not in ele:
            return None
        ele = ele[k]
    return ele

def increasing_subseqs(lis):
    return [lis[:i] for i in range(1, len(lis) + 1)]

def insert_path(kp_vals, inp_dict: dict):
    ele = inp_dict
    for k, v in kp_vals:
        if k not in ele:
            ele[k] = v
        ele = ele[k]

def group_objects_equal(grp_objs1: dict, grp_objs2: dict):
    go1_kps, go2_kps = [frozenset(dict_paths(go)) for go in (grp_objs1, grp_objs2)]
    if go1_kps == go2_kps:
        lens_eq = all([len(path_value(kp, grp_objs1)) \
                       == len(path_value(kp, grp_objs2)) for kp in go1_kps])
        return (lens_eq, go1_kps, go2_kps)
    return (False, go1_kps, go2_kps)

def smallest_enclosing_img(obj: tuple):
    rs, cs = obj
    return np.zeros(((max(rs) - min(rs)) + 1, (max(cs) - min(cs)) + 1), dtype=np.uint8)

def file_funcs(py_file):
    return [m[1] for m in inspect.getmembers(py_file) \
            if inspect.isfunction(m[1]) and m[1].__name__ != '<lambda>']

def color_gaps(grid, gap_col, gap_obj_coh: dict):
    in_grid = np.asarray(grid)
    for pv, objs in gap_obj_coh.items():
        rcs = [rc for rc in [list(zip(*g)) for _, g in objs] if len(rc) == 2]
        if rcs:
            for rs, cs in rcs:
                in_grid[rs, cs] = gap_col

    return in_grid

def image_from_object_cohesion(obj_coh: dict, r, c):
    image = np.zeros((r, c), dtype=np.uint8)
    nbg_oc = priors.remove_background(obj_coh)
    for pv, regs in nbg_oc.items():
        for rs, cs in [pairs_to_indicies(rg) for rg in regs]:
            image[rs, cs] = pv
            
    return image

def tuple_from_image(image):
    return tuple(tuple(l) for l in np.ndarray.tolist(image))
        
