import dsl.base as bdsl
import utils
import dsl.filter_funcs as dff
import dsl.objective_funcs as dof
import functools
import operator as op
import inspect

def general_func(**kwargs):
    filt_funcs = [v for k, v in kwargs.items() if k.startswith('filt_func')]
    dat_map = kwargs['dat_map']
    objective_func = kwargs['objective_func']

    lis_obs = dat_map[bdsl.LIST_OF_OBJS]
    filt_objs = functools.reduce(op.add, [[(pv, obj) for pv, obj in lis_obs if ff(pv=pv, obj=obj)] \
                                     for ff in filt_funcs], [])
    fo_ks = ['o' + str(i) for i in range(1, len(filt_objs) + 1)]
    fo_args = {k: o for k, o in zip(fo_ks, filt_objs)}
    obj_func_args = {k for k in inspect.getfullargspec(objective_func).kwonlyargs \
                     if k.startswith('o')}
    if obj_func_args == set(fo_args):
        out_objs = objective_func(**fo_args, dat_map=dat_map)
        mod_objs = {fo_args[k]: o for k, o in out_objs.items()}
    else:
        mod_objs = {o: o for _, o in fo_args.items()}

    dat_map[bdsl.MODIFIED_OBJS] = mod_objs
    return dat_map


FUNC_SPACE = lambda lib: functools.reduce(lambda l1, l2: l1 + l2, 
                                              [lib.FUNC_SPACE[f] for f in utils.file_funcs(lib)])
ARG_SPACE = {general_func: {'filt_func1': FUNC_SPACE(dff),
                            'filt_func2': FUNC_SPACE(dff),
                            'objective_func': FUNC_SPACE(dof)}}

