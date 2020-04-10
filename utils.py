import numpy as np
import priors

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

def smallest_enclosing_img(obj: tuple):
    rs, cs = obj
    return np.zeros(((max(rs) - min(rs)) + 1, (max(cs) - min(cs)) + 1), dtype=np.uint8)

def func_reduce(funcs, iv):
    if len(funcs) <= 0:
        return iv
    return func_reduce(funcs[1:], funcs[0](iv))

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

