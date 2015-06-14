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
