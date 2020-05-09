import functools

def color_filter(v, pv, obj):
    return pv == v

FUNC_SPACE = {color_filter: [functools.partial(color_filter, **{'v': pv}) for pv in range(1, 10)]}
