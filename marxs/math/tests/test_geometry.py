# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from numpy.core.umath_tests import inner1d
from astropy.table import Table
import transforms3d
import transforms3d.euler
import pytest

from ..geometry import Cylinder, FinitePlane
from ...tests import closeornan
from ...math.utils import h2e
from ...design import RowlandTorus


def test_intersect_plane_forwards():

    pos = np.array([[20., 0., 0.8, 1.],  # start above
                    [0., .0, 0.4, 1.],  # start inside
                    [-5., .1, -.2, 1]]) # start below
    dir = np.array([[-.1, 0., 0., 0],
                    [-1., 0, 0.02, 0.],
                    [-2., -0.02, 0.023, 0.]])

    plane = FinitePlane()
    intersect, interpos, inter_local = plane.intersect(dir, pos)
    assert np.all(intersect == [True, True, False])
    intersect, interpos, inter_local = plane.intersect(-dir, pos)
    assert np.all(intersect == [False, True, True])

def test_intersect_plane_zoomed():
    pos = np.array([[1., 4., 0.8, 1.],
                    [1., -4., 8., 1.],
                    [1., -6., -8., 1]])
    dir = np.array([[-.1, 0., 0., 0],
                    [-1., 0, 0.02, 0.],
                    [-2., -0.02, 0.023, 0.]])

    plane = FinitePlane({'zoom': [1, 5, 10]})
    intersect, interpos, inter_local = plane.intersect(dir, pos)
    assert np.all(intersect == [True, True, False])
    assert np.all(interpos[0, :] == [0, 4, .8, 1])
    assert np.all(inter_local[0, ] == [4, .8])

def test_intersect_plane_moved():

    pos = np.array([[4.1, 5.5, 6.8, 1.],
                    [5., 4.1, 4.99, 1.],
                    [4.1, 3., 6., 1]])
    dir = np.array([[-.1, 0., 0., 0],
                    [-1., 0, 0.02, 0.],
                    [-2., -0.02, 0.023, 0.]])

    plane = FinitePlane({'position': [4,5,6]})
    intersect, interpos, inter_local = plane.intersect(dir, pos)
    assert np.all(intersect == [True, True, False])
    assert np.all(interpos[1, :] == [4, 4.1, 5.01, 1])
    assert np.allclose(inter_local[1, :], [-0.9, -.99])

def test_intersect_tube_miss():
    '''Rays passing at larger radius or through the center miss the tube.'''
    circ = Cylinder()
    # Passing at larger radius
    intersect, interpos, inter_local = circ.intersect(np.array([[1., 0., .1, 0],
                                                                [-1., -1., 0., 0.]]),
                                                         np.array([[0, -1.1, 0., 1.],
                                                                   [2., 0., 0., 1.]]))
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

    # Through the center almost parallel to z-axis
    intersect, interpos, inter_local = circ.intersect(np.array([[0., 0.1, 1., 0]]),
                                                         np.array([[0.5, 0.5, 0., 1.]]))
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

    # Repeat with a tube that's moved to make sure we did not mess up local and
    # global coordinates in the implementation
    circ = Cylinder({'position': [0.8, 0.8, 1.2]})
    intersect, interpos, inter_local = circ.intersect(np.array([[1., 0., .0, 0],
                                                                [1., 0., 0., 0.]]),
                                                         np.array([[0., 0., 0., 1.],
                                                                   [0., -0.3, 1., 1.]]))
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

def test_intersect_tube_rays_forward():
    '''Rays only intersect if propagating them forward works'''
    pos = np.array([[20., 0., 0.8, 1.],  # start above
                    [0., .0, 0.4, 1.],  # start inside
                    [-5., .1, -.2, 1]]) # start below
    dir = np.array([[-.1, 0., 0., 0],
                    [-1., 0, 0.02, 0.],
                    [-2., -0.02, 0.023, 0.]])

    circ = Cylinder({'zoom': [1, 1, 1]})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == [True, True, False])
    intersect, interpos, inter_local = circ.intersect(-dir, pos)
    assert np.all(intersect == [False, True, True])


def test_intersect_tube_hitmiss_zoomed():
    '''Rays that hit one tube, but miss a shorter one'''
    dir = np.array([[-.1, 0., 0., 0], [-1., 0, 0.2, 0.]])
    pos = np.array([[20., 0., 0.8, 1.], [2., .0, 0.4, 1.]])

    circ = Cylinder({'zoom': [1, 1, 1]})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)

    circ = Cylinder({'zoom': [1, 1, .5]})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

def test_intersect_tube_2points():
    dir = np.array([[-.1, 0., 0., 0], [-0.5, 0, 0., 0.], [-1., 0., 0., 0.]])
    pos = np.array([[50., 0., 0., 1.], [10., .5, 0., 1.], [2, 0, .3, 1.]])

    circ = Cylinder()
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[1., 0., 0.],
                                                [np.sqrt(0.75), 0.5, 0.],
                                                [1, 0., .3]]))
    assert np.allclose(inter_local, np.array([[0., np.arcsin(0.5), 0.],
                                              [0., 0., 0.3]]).T)

def test_intersect_tube_2points_transformed():
    '''inter_local should be the same as interpos,
    even if we make the tube longer in z'''
    pos = np.array([[50., 0., 0., 1.], [10., .5, 0., 1.], [2, 0, .3, 1.]])
    dir = np.array([[-.1, 0., 0., 0], [-0.5, 0, 0., 0.], [-1., 0., 0., 0.]])

    exp_inter_loc = np.array([[0., np.arcsin(0.5), 0.],
                              [0., 0., 0.3]]).T
    exp_interpos = np.array([[1., 0., 0.],
                             [np.sqrt(0.75), 0.5, 0.],
                             [1, 0., .3]])
    circ = Cylinder({'zoom': [1, 1., 5.]})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), exp_interpos)
    assert np.allclose(inter_local, exp_inter_loc)
    # Rotate: interpos should be the same as in the unrotated case.
    # Beware of rotation direction!
    orient = transforms3d.axangles.axangle2mat([0, 0, 1], -.3)
    circ = Cylinder({'orientation': orient})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), exp_interpos)
    assert np.allclose(inter_local, np.array([[0.3, 0.3 + np.arcsin(0.5), 0.3],
                                              [0., 0., 0.3]]).T)

    # Shift: inter_local is the same, but interpos changes
    circ = Cylinder({'position': [-.3, 0, 0]})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[.7, 0, 0],
                                                [np.sqrt(0.75) -.3, 0.5, 0.],
                                                [.7, 0., .3]]))
    assert np.allclose(inter_local, exp_inter_loc)

def test_intersect_tube_2points_outside():
    '''Setting phi_lim

    These are the same numbers, but with the phi_lim set, all intersection are
    at the "bottom" of the cylinder.'''
    pos = np.array([[50., 0., 0., 1.], [10., .5, 0., 1.], [2, 0, .3, 1.]])
    dir = np.array([[-.1, 0., 0., 0], [-0.5, 0, 0., 0.], [-1., 0., 0., 0.]])

    circ = Cylinder({'phi_lim': [.6, np.pi]})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[-1., 0., 0.],
                                                [-np.sqrt(0.75), 0.5, 0.],
                                                [-1, 0., .3]]))
    assert np.allclose(inter_local, np.array([[np.pi, np.pi - np.arcsin(0.5), np.pi],
                                              [0., 0., 0.3]]).T)
    # phi_lims that guarantee a miss
    circ = Cylinder({'phi_lim': [-.4, -.3]})
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

@pytest.mark.xfail
def test_phi_lim_verification():
    '''Check Errors for wrong phi_lim format.'''
    with pytest.raises(ValueError) as e:
        circ = Cylinder({'phi_lim': [-.4, -.3, .4]})
    assert '[lower limit, upper limit]' in str(e.value)

    with pytest.raises(ValueError) as e:
        circ = Cylinder({'phi_lim': [-.4, -.5]})
    assert '[lower limit, upper limit]' in str(e.value)

    with pytest.raises(ValueError) as e:
        circ = Cylinder({'phi_lim': [-.4, 5]})
    assert 'range -pi to +pi' in str(e.value)


def test_intersect_tube_2points_translation():
    '''Repeat test above with a tube that's moved and zoomed to make sure we
    did not mess up local and global coordinates in the implementation
    '''
    circ = Cylinder({'position': [0.8, 0.8, 1.2], 'zoom': [1, 1, 2]})
    intersect, interpos, inter_local = circ.intersect(np.array([[1., 0., .0, 0],
                                                                [1., 0., 0., 0.]]),
                                                         np.array([[0., 0., 0., 1.],
                                                                   [0., 0., 1., 1.]]))
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[.2, 0., 0.],
                                                [.2, 0., 1.]]))

def test_tube_parametric():
    '''Generate points on surface using parametric. Then make rays for intersection
    which should return those points as intersection points.'''
    circ = Cylinder({'zoom': [2., 3., 4.]})
    parametric = circ.parametric_surface(phi=[-.1, 0., 1.])
    parametric = parametric.reshape((6, 4))
    # select dir so that it's in the right direction to recover these points.
    dir = np.tile([1, 0, 0, 0], (6, 1))
    intersect, interpos, inter_local = circ.intersect(dir, parametric)
    assert np.allclose(parametric, interpos)

@pytest.mark.parametrize("geom", [FinitePlane, Cylinder])
def test_local_coordsys(geom):
    '''Ensure local coordiante systems are orthonormal'''
    rot = transforms3d.euler.euler2mat(*(np.pi * 2 * np.random.rand(3)))
    g = geom({'rotation': rot,
              'position': np.random.rand(3)})

    x, y, z = g.get_local_euklid_bases(np.random.rand(5, 2))

    assert np.allclose(inner1d(x, y), 0)
    assert np.allclose(inner1d(x, z), 0)

    for vec in [x, y, z]:
        assert np.allclose(np.linalg.norm(vec, axis=1), 1.)
