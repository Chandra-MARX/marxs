import numpy as np

from ..aperture import CircleAperture
from ...utils import generate_test_photons

def test_circle_phi():
    p = generate_test_photons(500)
    p.remove_column('pos')
    c = CircleAperture(phi=[0, np.pi / 2])
    p = c(p)
    assert np.all(p['pos'][1] >= 0)
    assert np.all(p['pos'][2] >= 0)
