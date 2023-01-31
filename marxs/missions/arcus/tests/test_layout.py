import numpy as np
import pytest

from ..arcus import defaultconf

@pytest.mark.xfail(reason="Arcus configuration is changing, so this might fail.")
def test_layout_to_EdHertz():
    '''Compare derived torus tilt angle to number calculated by
    Ed Hertz and Casey.'''
    assert np.isclose(np.rad2deg(np.sin(defaultconf['rowland_central'].pos4d[2, 1])), 2.89214, rtol=1e-3)
