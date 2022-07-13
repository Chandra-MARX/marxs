import numpy as np

from ..arcus import defaultconf

def test_layout_to_EdHertz():
    '''Compare derived torus tilt angle to number calcualted by
    Ed Hertz and Casey.'''
    assert np.isclose(np.rad2deg(np.sin(defaultconf['rowland_central'].pos4d[2, 1])), 2.89214, rtol=1e-3)
