import priors
import cache
import utils
import copy

INIT_GRID = 'init_grid'
OBJECT_COHESION = 'object_cohesion'
LIST_OF_OBJS = 'list_of_objects'
FIRST = 'first'
NEW_VALUE = 'new_value'
OUTPUT = 'output'
REGISTER1 = 'register1'
SHIFT = 'shift'

def init_grid(in_grid):
    return {INIT_GRID: in_grid}

def object_cohesion(dat_map):
    grid = dat_map[INIT_GRID]
    dat_map[OBJECT_COHESION] = cache.object_cohesion(grid)
    return dat_map

def list_of_objects(dat_map):
    oc = dat_map[OBJECT_COHESION]
    dat_map[LIST_OF_OBJS] = priors.list_of_objects(oc)
    return dat_map

def first(dat_map):
    os = dat_map[LIST_OF_OBJS]
    dat_map[FIRST] = os[0] if os else None
    return dat_map

def register1(dat_map):
    dat_map[REGISTER1] = copy.deepcopy(dat_map[FIRST])
    return dat_map

def remove_shift(dat_map):
    dat_map.pop(SHIFT, None)
    return dat_map

def rest(dat_map):
    os = dat_map[LIST_OF_OBJS]
    dat_map[LIST_OF_OBJS] = os[1:]
    return dat_map

def sync(dat_map):
    first = dat_map[FIRST]
    oc = dat_map[NEW_VALUE if NEW_VALUE in dat_map \
                 else OBJECT_COHESION]
    shift = dat_map[SHIFT]
    new_oc = copy.deepcopy(oc)
    #invariant: first[0] == shift[0]
    new_oc[first[0]].remove(first[1])
    new_oc[shift[0]].add(shift[1])
    dat_map[NEW_VALUE] = new_oc
    return dat_map

def output(dat_map):
    r, c = dat_map[INIT_GRID].shape
    dat_map[OUTPUT] = cache.image_from_object_cohesion(dat_map[NEW_VALUE], r, c)
    return dat_map
