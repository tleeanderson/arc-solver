import dsl.base as db
import utils
import eval_func
import functools
import itertools
import cache
import interpreter as intr
import dsl.general as dg
import numpy as np
import priors

def enumerate_funcs(func, arguments):
    perms = itertools.product(*[[(an, v) for v in avs] for an, avs in arguments.items()])
    return [functools.partial(func, **dict(p)) for p in perms]

def token_space(lib_const: dict):
    flatten = lambda lis: functools.reduce(lambda l1, l2: l1 + l2, lis, [])
    return flatten([flatten([enumerate_funcs(f, lib_const[lib][f]) for f in sfs]) for lib, sfs \
                                              in [(l, utils.file_funcs(l)) for l in lib_const]])

def filter_func_space(fs, func_const: dict):
    apply_cs = lambda kw, cs: all([cs[k](kw[k]) for k in set(kw)\
                                   .intersection(set(func_const))])
    return [f for f in fs if apply_cs(f.keywords, func_const)]

def filter_func_args(func_args, func_arg_const: dict):
    return {an: avs for an, avs in {an: filter_func_space(avs, func_arg_const[an]) \
                                    if an in func_arg_const else avs for an, avs \
                                    in func_args.items()}.items() if avs}

def filter_arg_space(arg_space, constraints: dict):
    return {f: filter_func_args(args, constraints[f]) if f in constraints \
            else args for f, args in arg_space.items()}

def arg_space(libs, lib_const: dict):
    return {li: filter_arg_space(li.ARG_SPACE, lib_const[li]) for li in libs}

def general_program(in_images, out_images, eval_func, prob_id):

    def evaluate(examples, prog, prob_id):
        return np.asarray([intr.evaluate(img, prog, ef, prob_id + "_" + str(pid))[1] \
                    for img, _, _, ef, pid in examples])

    def traverse(examples, tokens, prog, prob_id):
        valid_score = lambda s: len([e for e in np.ndarray.tolist(s) \
                                     if e is not None]) == np.size(s)
        curr_score = evaluate(examples, prog, prob_id)
        if valid_score(curr_score) and np.sum(curr_score) == 0:
            return prog
        imp_all = lambda dg: np.all(curr_score - dg > 0)
        scores = [(p, evaluate(examples, p, prob_id)) for p in [prog + [t] for t in tokens]]
        filt_scores = sorted([(p, s, np.sum(s)) for p, s in scores if valid_score(s) and imp_all(s)], 
                             key=lambda t: t[2])
        if filt_scores:
            grp_scores = {k: list(g) for k, g in itertools.groupby(filt_scores, key=lambda t: t[2])}
            for p, _, _ in grp_scores[min(grp_scores)]:
                return traverse(examples, tokens, p + [db.sync_mod_objs], prob_id)
        return prog

    examples = [(i, o, cache.object_cohesion(i), functools.partial(eval_func, o), pid) \
                for i, o, pid in zip(*[in_images, out_images, range(len(in_images))])]
    pvs = functools.reduce(set.union, [set(priors.remove_background(oc)) for _, _, oc, _, _ in examples])
    filt_pvs = pvs if pvs else range(10)
    lib_const = {dg: {dg.binary_func: {'filt_func1': {'v': lambda v: v in filt_pvs},
                                 'filt_func2': {'v': lambda v: v in filt_pvs}}}}
    filt_as = arg_space([dg], lib_const)
    tokens = token_space(filt_as)
    prog = [db.init_grid, db.object_cohesion, db.list_of_objects]
    return traverse(examples, tokens, prog, prob_id)
