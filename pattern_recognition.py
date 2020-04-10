import utils
import priors
import utils
import numpy as np
import dsl
import itertools

def gap_filling(train_input, train_output):
    """4612dd53"""
    

def majority_color_assoc(train_input, train_output):
    """d4469b4b"""
    input_cohs, output_cohs = utils.func_on_iters(priors.object_cohesion, train_input, train_output)
    out_shapes = [np.asarray(img).shape for img in train_output]
    out_groups = [priors.group_objects(oc, shp[0], shp[1]) for oc, shp in zip(output_cohs, out_shapes)]
    nbg_in_pix_count = [priors.pixel_count_desc(priors.pixel_percent(priors.remove_background(ic))) \
                        for ic in input_cohs]
    nbg_in_pix_maj = [pc[0][0] for pc in nbg_in_pix_count if len(pc) > 0]
    norm_out_objs = [og.keys() for og in out_groups]
    one_out_obj = all([len(og) == 1 for og in norm_out_objs])
    if one_out_obj:
        color_assoc = [list(g) for _, g in itertools.groupby(sorted(zip(nbg_in_pix_maj,
                                                             [list(ogs.values())[0][0][0] for ogs in out_groups],
                                                             [list(ks)[0] for ks in norm_out_objs])), 
                                                             key=lambda t: t)]
        mult_assoc_per_color = all([len(g) > 1 for g in color_assoc])
        if len(nbg_in_pix_maj) == len(norm_out_objs) and mult_assoc_per_color:
            possible_answers = {dsl.VAL2: {mip: (op, obj) for mip, op, obj \
                                           in [list(set(g))[0] for g in color_assoc]}}
            return (dsl.object_cohesion, dsl.remove_background, dsl.majority_pixel, 
                    lambda dat_map: {**dat_map, **possible_answers}, dsl.lookup_answer, dsl.single_obj_image)

def unique_object(train_input, train_output):
    """88a62173"""
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
            return (dsl.object_cohesion, dsl.group_objects, dsl.unique_object_norm, dsl.single_obj_image)

def majority(train_input, train_output):
    """5582e5ca"""
    _, _, equal = utils.pairwise_equal(train_input, train_output)
    if equal:
        input_cohs, output_cohs = utils.object_cohesion_lists(train_input, train_output)
        if utils.single_object_outputs(output_cohs):
            maj_inp_pixels = [priors.pixel_count_desc(priors.pixel_percent(oc))[0][0] for oc in input_cohs]
            out_colors = [list(oc.keys())[0] for oc in output_cohs]
            if all([oc == mjip for oc, mjip in zip(out_colors, maj_inp_pixels)]):
                return (dsl.object_cohesion, dsl.majority_pixel, dsl.create_image)
