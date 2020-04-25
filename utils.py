import numpy as np
import priors
import inspect

def pairs_to_indicies(obj):
    return zip(*list(obj))

def object_cohesion_lists(inputs, outputs):
    return [[priors.object_cohesion(i) for i in ds] for ds in (inputs, outputs)]

def num_objs(inp_coh, out_coh):
    return [[priors.num_objs(oc) for oc in ds] for ds in (inp_coh, out_coh)]

def pixel_count(inp_coh, out_coh):
    return [[priors.pixel_count(oc) for oc in ds] for ds in (inp_coh, out_coh)]

def pixel_count_desc(inp_coh, out_coh):
    return [[priors.pixel_count_desc(oc) for oc in ds] for ds in (inp_coh, out_coh)]

def dict_paths(input_dict: dict):
    
    def dfs(inp, cp):
        if type(inp) in {list, tuple}:
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

def func_reduce(funcs, iv):
    if len(funcs) <= 0:
        return iv
    return func_reduce(funcs[1:], funcs[0](iv))

def file_funcs(py_file):
    return [m[1] for m in inspect.getmembers(py_file) if inspect.isfunction(m[1])]

def single_object_outputs(cohs):
    _, o = num_objs(cohs, cohs)
    return set(o) == {1}

def get_dims(image: list):
    return np.asarray(image).shape

def image_list_shapes(image_list):
    return [get_dims(i) for i in image_list]

def pairwise_equal(inputs, outputs):
    ins, outs = [np.asarray(image_list_shapes(ds)) for ds in (inputs, outputs)]
    return ins, outs, np.all(ins == outs)

def func_on_iters_va(func, *iterables):
    return [[func(*e) for e in it] for it in iterables]

def func_on_iters(func, *iterables):
    return [[func(e) for e in it] for it in iterables]

def func_on_iters_mapf(func, map_func, *iterables):
    return [[func(*map_func(e)) for e in it] for it in iterables]

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
        
