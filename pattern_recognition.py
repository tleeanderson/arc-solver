import utils
import priors
import utils
import numpy as np
import dsl

def majority(train_input, train_output):
    in_shapes, out_shapes = utils.image_list_shapes(train_input), utils.image_list_shapes(train_output)
    if in_shapes == out_shapes:
        input_cohs, output_cohs = utils.object_cohesion(train_input, train_output)
        if utils.single_object_outputs(output_cohs):
            maj_inp_pixels = [priors.pixel_count_desc(priors.pixel_percent(oc))[0][0] for oc in input_cohs]
            out_colors = [list(oc.keys())[0] for oc in output_cohs]
            if all([oc == mjip for oc, mjip in zip(out_colors, maj_inp_pixels)]):
                return [dsl.object_cohesion, dsl.majority_pixel, dsl.create_image]
