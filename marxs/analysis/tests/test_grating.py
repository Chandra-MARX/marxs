import numpy as np

from ...utils import generate_test_photons
from ...optics import ThinLens, FlatGrating, FlatDetector, OrderSelector
from ..gratings import (resolvingpower_from_photonlist,
                        effectivearea_from_photonlist,
                        identify_photon_in_supaperture
                        )

def test_resolvingpower_from_photonlist():
    '''Regression test.

    [#159] left behind an undefined ``filterfunc``. This is fixed and a
    regression test added [#165].
    '''
    p = generate_test_photons(1000)
    for i in [1, 2]:
        p['pos'][:, i] = -0.1 + 0.2 * np.random.random(len(p))
    lens = ThinLens(focallength=1)
    p = lens(p)
    orders = np.array([-1, 0, 1])
    grat = FlatGrating(position=[-0.01, 0, 0], d=1e-4,
                       order_selector=OrderSelector(orders))
    p = grat(p)
    det = FlatDetector(position=[-1, 0, 0])
    p = det(p)
    res0, pos0, std0 = resolvingpower_from_photonlist(p, orders, col='det_x')
    assert res0[0] > 10
    assert res0[1] < 1
    assert res0[2] > 10
    res, pos, std = resolvingpower_from_photonlist(p, orders, col='det_x', zeropos=0)
    assert res[0] > 10
    assert res[1] < 1
    assert res[2] > 10
    # y is the cross-dispersion direction...
    resy, posy, stdy = resolvingpower_from_photonlist(p, orders, col='det_y', zeropos=0)
    assert np.all(resy < 1.)
    assert np.allclose(posy, 0, atol=1e-4)


def test_aeff_from_photonlist():
    '''Check the effective area calculation'''
    p = generate_test_photons(5)
    p['a'] = [0, 1, 2, 2, 1]
    p['probability'] = [.1, 1., 1., .3, .5]
    # orders in random order, just in case
    aeff = effectivearea_from_photonlist(p, [2, 1, 0, -1], 5,
                                         A_geom=2., ordercol='a')
    assert aeff[3] == 0
    assert np.isclose(aeff[2], .04)
    assert np.isclose(aeff[1], .6)


def test_identify_photon_in_supaperture():
    angle = [0, 59, 61, 180, 289]
    inaper = identify_photon_in_supaperture(np.deg2rad(angle),
                                            np.deg2rad(30), ang_0=np.pi / 2)
    assert np.all(inaper == [False, False, True, False, True])