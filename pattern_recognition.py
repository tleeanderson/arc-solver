import perception
import priors
import utils
import numpy as np

def alike_pixel_numbers(train_input, train_output, test_input):
    input_cohs, output_cohs = perception.object_cohesion(train_input, train_output)
    _, o = perception.num_objs(input_cohs, output_cohs)
    if perception.single_object_outputs(o):
        out_colors = [list(oc.keys())[0] for oc in output_cohs]
        inp_pcs, _ = perception.pixel_count(input_cohs, output_cohs)
        num_out_color_in_inputs = [ipcs[c] for c, ipcs in zip(out_colors, inp_pcs)]
        if len(set(num_out_color_in_inputs)) == 1:
            ppc = num_out_color_in_inputs[0]

    ti = test_input[0]
    test_inp_coh = priors.object_cohesion(ti)
    predict_color = [pv for pv, c in priors.pixel_count_desc(priors.pixel_count(test_inp_coh)) \
                     if ppc == c][0]
    ti_tens = np.asarray(ti)
    ni = np.zeros(ti_tens.shape, dtype=np.uint8)
    ni[:] = predict_color
    return [np.ndarray.tolist(ni)]
