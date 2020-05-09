import priors
import cache
import utils
import numpy as np

def one_to_one_diff(in_img1, in_img2):
    img1, img2 = [np.asarray(im) for im in (in_img1, in_img2)]
    oc_im1, oc_im2 = [cache.object_cohesion(im) for im in (img1, img2)]
    grp_im1, grp_im2 = [priors.group_objects(oc, sh[0], sh[1]) \
                        for oc, sh in ((oc_im1, img1.shape), (oc_im2, img2.shape))]
    lens_eq, kps_im1, _ = utils.group_objects_equal(grp_im1, grp_im2)
    if lens_eq:
        return sum([sum([cache.overlap_distance(o1, o2) for o1, o2 in zip(im1_os, im2_os)]) \
                    for im1_os, im2_os in [(utils.path_value(kp, grp_im1), 
                                            utils.path_value(kp, grp_im2)) for kp in kps_im1]])

def percent_not_matching(in_img1, in_img2):
    return np.sum(in_img1 != in_img2) / np.size(in_img1)
