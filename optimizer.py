import dsl.new_base as db
import dsl.shift as ds
import interpreter_funcs as int_funcs
import utils
import eval_func
import functools
import itertools
import time
import copy
import cache

TEST_TASK = '1caeab9d.json'
_code_cache = {}

def cached_subseq(prog, cache):
    for i in range(1, len(prog)):
        subseq = prog[:-i]
        if subseq in cache:
            return (subseq, -i)
    return (None, -1)

def replace_with_value(prog, subseq_end, value):
    return (lambda inp: copy.deepcopy(value),) + prog[subseq_end:]

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
        fv = utils.func_reduce(run_prog, image)
        cache[prob_id][tup_prog] = fv
        let = len(eval_toks)
        out_prog = replace_with_value(run_prog + eval_toks, -let, fv) if let > 0 \
                   else run_prog
        out = utils.func_reduce(out_prog, image)
        return out

def exec_cache(image, prog, prob_id, eval_toks):
    return exec_engine(_code_cache, image, prog, prob_id, eval_toks)

def evaluate(image, prog, eval_func, prob_id, eval_toks=(db.sync, db.output)):
    out = exec_cache(image, prog, prob_id, eval_toks)
    score = eval_func(out[db.OUTPUT])
    return (out, score)

def eval_no_cache(image, prog, eval_func, eval_toks=(db.sync, db.output)):
    out = utils.func_reduce(prog + eval_toks, image)
    score = eval_func(out[db.OUTPUT])
    return (out, score)

def eval_no_score(image, prog, prob_id, eval_toks=(db.sync, db.output)):
    return exec_cache(image, prog, prob_id, eval_toks)

def brute_force(image, eval_func, prob_id):

    def shift_obj(cp, tokens, eval_func, image, prob_id, state_val):
        if state_val == 0:
            return (cp, state_val)
        progs = [t for t in [(p, *evaluate(image, p, eval_func, prob_id)) for p in [cp + [t] for t in tokens]] \
                 if t[2] is not None]
        print_progs = [(p[-1], sv) for p, _, sv in progs]
        if progs:
            grps = itertools.groupby(sorted(progs, key=lambda t: t[2]), key=lambda t: t[2])
            scores = {sc: list(g) for sc, g in grps if sc < state_val}
            if scores:
                sub_trees = [shift_obj(p, tokens, eval_func, image, prob_id, sv) \
                             for p, _, sv in scores[min(scores)]]
                return min(sub_trees, key=lambda t: t[1])

        return (cp, state_val)

    def loop_objs(cp, tokens, eval_func, image, prob_id, state_val):
        if state_val == 0:
            return cp
        prog = cp + [db.first]
        obj = eval_no_score(image, prog, prob_id, eval_toks=tuple())[db.FIRST]
        if obj:
            shift, sv = shift_obj(prog, tokens, eval_func, image, prob_id, state_val)
            np, nsv = ((cp + [db.rest]), state_val) if shift == prog \
                        else ((shift + [db.sync, db.remove_shift, db.rest]), sv)
            return loop_objs(np, tokens, eval_func, image, prob_id, nsv)
        else:
            return cp

    shift_tokens = utils.file_funcs(ds)
    init_prog = [db.init_grid, db.object_cohesion, db.list_of_objects]
    return loop_objs(init_prog, shift_tokens, eval_func, image, prob_id, eval_func(image))

def test(times=1):
    i = 0
    while i < times:
        for ind in range(3):
            i += 1
            data = int_funcs.sample_image('1caeab9d.json', ind=ind)
            in_data, _ = data['input']
            out_data, _ = data['output']
            ef = functools.partial(cache.one_to_one_diff, out_data)

            start = time.time()
            out = brute_force(in_data, ef, ind)
            end = time.time()

            memory, score = eval_no_cache(in_data, out, ef, eval_toks=[db.output])
            print("final program, score: {}, i: {}, et: {}ms, prog_len: {}"\
                  .format(score, i, round((end-start)*1000), len(out)))
            if score != 0:
                for k, v in memory.items():
                    print("{}: {}".format(k, v))
                int_funcs.display(memory[db.OUTPUT])
                exit()

            # for k, v in _code_cache.items():
            #     print("{}: {}".format(k, v))
            # input()


if __name__ == '__main__':
    test()

    
