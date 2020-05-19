import dsl.base as bdsl
import utils
import dsl.filter_funcs as dff
import dsl.list_funcs as lf
import dsl.nary_funcs as nf
import functools
import operator as op
import inspect

def nary_func(**kwargs):
    dat_map = kwargs['dat_map']
    objective_func = kwargs['objective_func']

    ok_from_ff = lambda ff_nm: 'o' + ff_nm[ff_nm.rindex("_")+1:]
    objct_func_obj_args = {k for k in inspect.getfullargspec(objective_func).kwonlyargs \
                           if k.startswith('o')}
    filt_funcs = [v for k, v in kwargs.items() if k.startswith('filt_func_') \
                  and ok_from_ff(k) in objct_func_obj_args]
    lis_obs = dat_map[bdsl.LIST_OF_OBJS]
    filt_objs = functools.reduce(op.add, [[(pv, obj) for pv, obj in lis_obs if ff(pv=pv, obj=obj)] \
                                     for ff in filt_funcs], [])
    if len(filt_objs) == len(objct_func_obj_args):
        fo_ks = ['o' + str(i) for i in range(1, len(filt_objs) + 1)]
        fo_args = {k: o for k, o in zip(fo_ks, filt_objs)}
        out_objs = objective_func(**fo_args, dat_map=dat_map)
        dat_map[bdsl.MODIFIED_OBJS] = {fo_args[k]: o for k, o in out_objs.items()}

    return dat_map

def list_func(**kwargs):
    dat_map = kwargs['dat_map']
    objective_func = kwargs['objective_func']
    filt_func = kwargs['filt_func_1']
    
    filt_objs = [(pv, obj) for pv, obj in dat_map[bdsl.LIST_OF_OBJS] \
                 if filt_func(pv=pv, obj=obj)]
    if filt_objs:
        dat_map[bdsl.MODIFIED_OBJS] = objective_func(os=filt_objs, dat_map=dat_map)
    return dat_map

FUNC_SPACE = lambda lib: functools.reduce(op.add, 
                                          [lib.FUNC_SPACE[f] for f in utils.file_funcs(lib)])

ARG_SPACE = {nary_func: {'filt_func_1': FUNC_SPACE(dff),
                         'filt_func_2': FUNC_SPACE(dff),
                         'objective_func': FUNC_SPACE(nf)},
             list_func: {'filt_func_1': FUNC_SPACE(dff), 
                         'objective_func': FUNC_SPACE(lf)}}

