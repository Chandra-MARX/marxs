import numpy as np

from ..aperture import CircleAperture, RectangleAperture, MultiAperture
from ...utils import generate_test_photons

def test_circle_phi():
    p = generate_test_photons(500)
    p.remove_column('pos')
    c = CircleAperture(phi=[0, np.pi / 2])
    p = c(p)
    assert np.all(p['pos'][1] >= 0)
    assert np.all(p['pos'][2] >= 0)

def test_MultiAperture():
    p = generate_test_photons(1000)
    p.remove_column('pos')

    a1 = CircleAperture()
    a2 = RectangleAperture(position=[0, 4, 0], zoom=2)
    a = MultiAperture(elements=[a1, a2], id_col='aper')

    p = a(p)

    # All photons go through some aperture
    assert (p['aper'] == 0).sum() + (p['aper'] == 1).sum() == 1000
    # Most photons go through the bigger aperture
    assert (p['pos'][1:] > 1).sum() > 650
