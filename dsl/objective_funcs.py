import functools
import dsl.base as db
import priors
import utils

def shift_equal_axis(ax, o1, o2, dat_map):
    _, ob1 = o1
    pv_o2, ob2 = o2
    grid = dat_map[db.INIT_GRID]
    r, c = grid.shape
    o1_tl, o2_tl = priors.corners(grid, ob1)[0][ax], priors.corners(grid, ob2)[0][ax]
    shift = [0] * 2
    shift[ax] = o1_tl - o2_tl
    rs, cs = shift
    return {'o1': o1, 'o2': (pv_o2, priors.shift_object(ob2, r, c, rs, cs))}

def shift_until_obj_collision(ax, d, o1, dat_map):
    pv_ob1, ob1 = o1
    grid = dat_map[db.INIT_GRID]
    r, c = grid.shape
    oc = priors.remove_background(dat_map[db.OBJECT_COHESION])
    shift_obj = ob1
    safe = lambda o: 0 not in functools.reduce(set.union, 
                                              [{priors.object_distance(o, go)[0] for go \
                                                in set.difference(objs, o)} \
                                                     for _, objs in oc.items()], {0})
    shift_obj = ob1
    while safe(shift_obj):
        shift_obj = priors.shift_object(ob1, r, c, rs + d, cs) if ax == 0 \
                    else priors.shift_object(ob1, r, c, rs, cs + d)
    return {'o1': (pv_ob1, shift_obj)}
        

FUNC_SPACE = {shift_equal_axis: [functools.partial(shift_equal_axis, **{'ax': av}) for av in (0, 1)], 
              shift_until_obj_collision: utils.enumerate_funcs(shift_until_obj_collision, 
                                                               {'ax': {0, 1}, 
                                                                'd': {-1, 1}})}
