import base_dsl as bdsl

def gap_fill(gap_color, deviations):
    return [bdsl.convert_value, lambda dat_map: {**dat_map, **{bdsl.VAL: deviations}},
            bdsl.sparse_and_object_cohesion, bdsl.group_object_cohesion, 
            bdsl.object_gaps, lambda dat_map: {**dat_map, **{bdsl.VAL2: gap_color}}, bdsl.color_gaps]
