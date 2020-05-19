import functools
import dsl.base as db
import priors
import utils
import copy

def shift_until_collision(ax, d, os, dat_map):
    grid = dat_map[db.INIT_GRID]
    r, c = grid.shape
    oc = priors.remove_background(dat_map[db.OBJECT_COHESION])
    rem_obj = lambda obj, oc: set.difference(functools.reduce(set.union, oc.values(), set()), {obj})
    alone = lambda obj, os: 0 not in {priors.object_distance(obj, o, -1)[0] for o in os}

    shift_objs = []
    shift_vals = [0, 0]
    shift_vals[ax] = d
    shift_args = {'rs': shift_vals[0], 'cs': shift_vals[1]}
    for pv, ob in os:
        other_objs = rem_obj(ob, oc)
        po = None
        a = alone(ob, other_objs)
        while a and po != ob:
            po = ob
            ob = priors.shift_object(ob, r, c, **shift_args)
            a = alone(ob, other_objs)
        shift_objs.append((pv, ob))

    return {os[i]: shift_objs[i] for i in range(len(os))}

FUNC_SPACE = {shift_until_collision: utils.enumerate_funcs(shift_until_collision, 
                                                               {'ax': {0, 1}, 
                                                                'd': {-1, 1}})}
