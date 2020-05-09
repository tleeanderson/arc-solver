import priors
import cache
import utils
import eval_func
import copy

_cache = {}

def clear():
    global _cache
    _cache = {}

def update_cache(kp, kv):
    global _cache
    value = copy.deepcopy(utils.path_value(kp, _cache))
    if value is None:
        nkv = kv[:-1] + (kv[-1](),)
        utils.insert_path(zip(kp, nkv), _cache)
        value = copy.deepcopy(utils.path_value(kp, _cache))
    return value

def hash_group_objects(grp_objs: dict):
    return frozenset((so, tuple(os.items())) for so, os in grp_objs.items())

def hash_obj_cohesion(obj_coh: dict):
    return frozenset((pv, frozenset(objs)) for pv, objs in obj_coh.items())

def object_cohesion(in_grid):
    kp = ('object_cohesion', utils.tuple_from_image(in_grid))
    kv = ({}, lambda: priors.object_cohesion(in_grid))
    return update_cache(kp, kv)

def one_to_one_diff(img1, img2):
    kp = ('one_to_one_diff', tuple(utils.tuple_from_image(im) for im in (img1, img2)))
    kv = ({}, lambda: eval_func.one_to_one_diff(img1, img2))
    return update_cache(kp, kv)

def overlap_distance(o1: frozenset, o2: frozenset):
    kp = ('overlap_distance', (o1, o2))
    kv = ({}, lambda: priors.overlap_distance(o1, o2))
    return update_cache(kp, kv)

def group_objects(obj_coh: dict, r, c):
    kp = ('group_objects', (hash_obj_cohesion(obj_coh), r, c))
    kv = ({}, lambda: priors.group_objects(obj_coh, r, c))
    return update_cache(kp, kv)

def image_from_object_cohesion(obj_coh: dict, r, c):
    kp = ('image_from_object_cohesion', (hash_obj_cohesion(obj_coh), r, c))
    kv = ({}, lambda: utils.image_from_object_cohesion(obj_coh, r, c))
    return update_cache(kp, kv)
