import copy
import dsl.base as db

_code_cache = {}

def func_reduce(funcs, iv):
    if len(funcs) <= 0:
        return iv
    return func_reduce(funcs[1:], funcs[0](dat_map=iv))

def cached_subseq(prog, cache):
    for i in range(1, len(prog)):
        subseq = prog[:-i]
        if subseq in cache:
            return (subseq, -i)
    return (None, -1)

def replace_with_value(prog, subseq_end, value):
    return (lambda dat_map: copy.deepcopy(value),) + prog[subseq_end:]

def exec_engine(cache, image, prog, prob_id, eval_toks):
    if prob_id not in cache:
        cache[prob_id] = {}
    tup_prog = tuple(prog)
    if tup_prog in cache[prob_id]:
        return cache[prob_id][tup_prog]
    else:
        ss, ind = cached_subseq(tup_prog, cache[prob_id])
        run_prog = replace_with_value(tup_prog, ind, cache[prob_id][ss]) if ss \
                   else tup_prog
        fv = func_reduce(run_prog, image)
        cache[prob_id][tup_prog] = fv
        let = len(eval_toks)
        out_prog = replace_with_value(run_prog + eval_toks, -let, fv) if let > 0 \
                   else run_prog
        out = func_reduce(out_prog, image)
        return out

def exec_cache(image, prog, prob_id, eval_toks):
    return exec_engine(_code_cache, image, prog, prob_id, eval_toks)

def evaluate(image, prog, eval_func, prob_id, eval_toks=(db.sync_mod_objs, db.output)):
    out = exec_cache(image, prog, prob_id, eval_toks)
    score = eval_func(out[db.OUTPUT])
    return (out, score)

def eval_no_cache(image, prog, eval_func, eval_toks=(db.sync_mod_objs, db.output)):
    out = func_reduce(tuple(prog) + eval_toks, image)
    score = eval_func(out[db.OUTPUT])
    return (out, score)

def eval_no_score(image, prog, prob_id, eval_toks=(db.sync_mod_objs, db.output)):
    return exec_cache(image, prog, prob_id, eval_toks)

def clear_cache():
    global _code_cache
    _code_cache = {}
