import priors

def object_cohesion(inputs, outputs):
    return [[priors.object_cohesion(i) for i in ds] for ds in (inputs, outputs)]

def num_objs(inp_coh, out_coh):
    return [[priors.num_objs(oc) for oc in ds] for ds in (inp_coh, out_coh)]

def pixel_count(inp_coh, out_coh):
    return [[priors.pixel_count(oc) for oc in ds] for ds in (inp_coh, out_coh)]

def pixel_count_desc(inp_coh, out_coh):
    return [[priors.pixel_count_desc(oc) for oc in ds] for ds in (inp_coh, out_coh)]

def single_object_outputs(outputs):
    return set(outputs) == {1}
