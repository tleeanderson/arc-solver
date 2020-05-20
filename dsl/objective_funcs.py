import functools
import dsl.base as db
import priors

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

FUNC_SPACE = {shift_equal_axis: [functools.partial(shift_equal_axis, **{'ax': av}) for av in (0, 1)]}
