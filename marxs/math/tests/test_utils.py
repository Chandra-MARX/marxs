# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from ..import utils
from transforms3d.affines import compose

def test_random_mat():
    '''Compare multiplication of trans and zoom matrixes with transfomrs3d.

    This test ensures that the same conventions are used for ordering.
    '''
    rot = np.eye(3)
    trans = np.random.rand(3)
    zoom = np.random.rand(3)
    assert np.allclose(np.dot(utils.translation2aff(trans), np.dot(utils.mat2aff(rot), utils.zoom2aff(zoom))),
                       compose(trans, rot, zoom))

def test_random_mat_scalar_zoom():
    '''Compare multiplication of trans and zoom matrixes with transfomrs3d.

    This test ensures that the same conventions are used for ordering.
    '''
    rot = np.eye(3)
    trans = np.random.rand(3)
    zoom = np.random.rand()
    zoommat = np.ones(3) * zoom
    assert np.allclose(np.dot(utils.translation2aff(trans), np.dot(utils.mat2aff(rot), utils.zoom2aff(zoom))),
                       compose(trans, rot, zoommat))

def test_normalize():
    '''Currently only works with 2 d input.'''
    normvec = np.array([[1.,0,0]])
    assert np.all(normvec == utils.norm_vector(normvec))

    vec = np.arange(15, dtype=float).reshape((5,3))
    vecout = np.ones_like(vec)
    for i in range(5):
        vecout[i, :] = vec[i, :] / np.linalg.norm(vec[i, :])
    assert np.allclose(vecout, utils.norm_vector(vec))

def test_anglediff():
    assert np.isclose(utils.anglediff([0, 3.]), 3.)
    assert np.isclose(utils.anglediff([-1, 1]), 2.)
    assert np.isclose(utils.anglediff([1., -1.]), 2 * np.pi - 2.)

def test_angle_between():
    assert utils.angle_between(.1, 0, 2.)
    assert np.all(utils.angle_between(np.array([-.1, 0., 6.1, .2, 6.5]), 5., 1.))
    assert utils.angle_between(2., 6., 1.) == False
    assert utils.angle_between(4., 1., 2.) == False
    assert utils.angle_between(-.5, .5, -3.1) == False
