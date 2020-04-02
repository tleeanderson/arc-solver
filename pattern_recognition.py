import utils
import priors
import utils
import numpy as np
import dsl

def unique_object(train_input, train_output):
    in_shapes, out_shapes = utils.func_on_iters(lambda img: np.asarray(img).shape, train_input, train_output)
    input_cohs, output_cohs = utils.func_on_iters(priors.object_cohesion, train_input, train_output)
    inp_groups, out_groups = utils.func_on_iters_mapf(priors.group_objects, lambda s: (s[0], s[1][0], s[1][1]),
                                                 zip(input_cohs, in_shapes), zip(output_cohs, out_shapes))
    inp_uniques = [[(ss, shps[0]) for ss, shps in ig.items() if len(shps) == 1] for ig in inp_groups]
    out_uniques = [list(og.keys()) for og in out_groups]
    one_in_uniq, one_out_uniq = [all(ds) for ds in \
                                 utils.func_on_iters(lambda u: len(u) == 1, inp_uniques, out_uniques)]
    if one_in_uniq and one_out_uniq:
        iqs, oqs = utils.func_on_iters(lambda l: l[0], inp_uniques, out_uniques)
        inp_unique_in_output = all([inp_uniq[0] == oqs[i] for i, inp_uniq in enumerate(iqs)])
        if inp_unique_in_output:
            return [dsl.object_cohesion, dsl.group_objects, dsl.unique_object_norm, dsl.single_obj_image]

def majority(train_input, train_output):
    """Works for task: 5582e5ca"""
    _, _, equal = utils.pairwise_equal(train_input, train_output)
    if equal:
        input_cohs, output_cohs = utils.object_cohesion_lists(train_input, train_output)
        if utils.single_object_outputs(output_cohs):
            maj_inp_pixels = [priors.pixel_count_desc(priors.pixel_percent(oc))[0][0] for oc in input_cohs]
            out_colors = [list(oc.keys())[0] for oc in output_cohs]
            if all([oc == mjip for oc, mjip in zip(out_colors, maj_inp_pixels)]):
                return [dsl.object_cohesion, dsl.majority_pixel, dsl.create_image]
