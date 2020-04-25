import priors
import dsl.new_base as new_base

def shift_left(dat_map):
    r, c = dat_map[new_base.INIT_GRID].shape
    pv, obj = dat_map[new_base.SHIFT if new_base.SHIFT in dat_map \
                      else new_base.FIRST]
    dat_map[new_base.SHIFT] = (pv, priors.shift_object_left(obj, r, c))
    return dat_map

def shift_right(dat_map):
    r, c = dat_map[new_base.INIT_GRID].shape
    pv, obj = dat_map[new_base.SHIFT if new_base.SHIFT in dat_map \
                      else new_base.FIRST]
    dat_map[new_base.SHIFT] = (pv, priors.shift_object_right(obj, r, c))
    return dat_map

def shift_up(dat_map):
    r, c = dat_map[new_base.INIT_GRID].shape
    pv, obj = dat_map[new_base.SHIFT if new_base.SHIFT in dat_map \
                      else new_base.FIRST]
    dat_map[new_base.SHIFT] = (pv, priors.shift_object_up(obj, r, c))
    return dat_map

def shift_down(dat_map):
    r, c = dat_map[new_base.INIT_GRID].shape
    pv, obj = dat_map[new_base.SHIFT if new_base.SHIFT in dat_map \
                      else new_base.FIRST]
    dat_map[new_base.SHIFT] = (pv, priors.shift_object_down(obj, r, c))
    return dat_map



