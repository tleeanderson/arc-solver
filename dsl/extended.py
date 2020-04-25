from . import base

def gap_fill(gap_color, deviations):
    return [base.convert_value, lambda dat_map: {**dat_map, **{base.VAL: deviations}},
            base.sparse_and_object_cohesion, base.group_object_cohesion, 
            base.object_gaps, lambda dat_map: {**dat_map, **{base.VAL2: gap_color}}, base.color_gaps]
