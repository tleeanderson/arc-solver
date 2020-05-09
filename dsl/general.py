import dsl.base as bdsl
import utils
import dsl.filter_funcs as dff
import dsl.objective_funcs as dof
import functools
import operator as op

def binary_func(filt_func1, filt_func2, objective_func, dat_map):
    lis_obs = dat_map[bdsl.LIST_OF_OBJS]
    objs = functools.reduce(op.add, [[(pv, obj) for pv, obj in lis_obs if ff(pv=pv, obj=obj)] \
                                     for ff in (filt_func1, filt_func2)], [])
    lo = len(objs)
    if lo == 2:
        o1, o2 = objs
        no1, no2 = objective_func(o1=o1[1], o2=o2[1], dat_map=dat_map)
        mod_objs = {o1: (o1[0], no1), o2: (o2[0], no2)}
    else:
        mod_objs = {o: o for o in objs}

    dat_map[bdsl.MODIFIED_OBJS] = mod_objs
    return dat_map

FUNC_SPACE = lambda lib: functools.reduce(lambda l1, l2: l1 + l2, 
                                              [lib.FUNC_SPACE[f] for f in utils.file_funcs(lib)])
ARG_SPACE = {binary_func: {'filt_func1': FUNC_SPACE(dff),
                           'filt_func2': FUNC_SPACE(dff),
                           'objective_func': FUNC_SPACE(dof)}}

