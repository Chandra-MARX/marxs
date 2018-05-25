import numpy as np
from ...utils import generate_test_photons
from ..analysis import find_best_detector_position, mean_width_2d
from ...simulator import propagate

def test_best_detector_position():
    photons = generate_test_photons(100)
    photons['dir'].data[:, 1] = np.random.rand(100)
    photons['pos'].data[:, 1] = 0.5
    photons = propagate(photons, -10)
    # det_x is along the second axis
    out = find_best_detector_position(photons,
                                      objective_func_args={'colname': 'det_x'})
    assert np.isclose(out.x, 1.)


def test_best_detector_position_rotated():
    '''Now use z axis as optical axis'''
    photons = generate_test_photons(100)
    photons['dir'].data[:, 0] = np.random.rand(100)
    photons['dir'].data[:, 2] = - 1
    photons['pos'].data[:, 2] = 0.5
    photons = propagate(photons, -10)
    # det_x is along the second axis
    out = find_best_detector_position(photons,
                                      objective_func_args={'colname': 'det_x'},
                                      orientation=np.array([[ 0.,  1.,  0.],
                                                            [ 0.,  0.,  1.],
                                                            [ 1.,  0.,  0.]]))
    assert np.isclose(out.x, 0.5)


def test_best_detector_position_2d():
    '''Optimize a 2d distribution on the detector'''
    photons = generate_test_photons(100)
    photons['dir'].data[:, 2] = np.random.rand(100)
    photons['pos'].data[:, 1] = 0.5
    photons = propagate(photons, -10)
    # det_x is along the second axis
    out = find_best_detector_position(photons, mean_width_2d, {})
    assert np.isclose(out.x, 1.0)


# from ..gratings import resolvingpower_per_order

# def test_robust_resolvingpower():
#     '''
#         - Pass in an instance of an optical element (e.g. a
#           `marxs.optics.FlatDetector`).
#         - Pass in a `marxs.design.RowlandTorus`. This function will generate
#           a detector that follows the Rowland circle.
#         - ``None``. A flat detector in the yz plane is used, but the x position
#           for this detector is numerically optimized in each step.
#     '''
#     resolvingpower_per_order(gratings, photons, orders, detector=None,
#                              colname='det_x')
