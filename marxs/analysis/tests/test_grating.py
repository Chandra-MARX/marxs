import numpy as np
from astropy.table import Table
import pytest

from ...utils import generate_test_photons
from ...optics import ThinLens, FlatGrating, FlatDetector, OrderSelector
from ..gratings import (resolvingpower_from_photonlist,
                        resolvingpower_from_photonlist_robust,
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


def test_resolvingpower_from_photonlist_robust():
    orders = [1, 2, 3]
    size = 10000
    photons = Table({'col1': 2000 + np.random.normal(size=size),
                     'col2': 1000 + np.random.normal(size=size),
                     'order': np.random.choice(orders, size=size)})
    res, pos, std = resolvingpower_from_photonlist_robust(photons, orders,
                                                          ['col1', 'col2'],
                                                          [0, 0])
    assert 1000 / 2.3548 * np.ones(3) == pytest.approx(res, rel=1e-1)
    assert np.ones(3) == pytest.approx(std, rel=1e-1)
    assert 1000 * np.ones(3) == pytest.approx(pos, rel=1e-1)

    res, pos, std = resolvingpower_from_photonlist_robust(photons, orders,
                                                          ['col1', 'col2'],
                                                          [0, -1000])
    assert 2000 / 2.3548 * np.ones(3) == pytest.approx(res, rel=1e-1)

def test_resolvingpower_from_photonlist_robust2():
    '''Now give zeropos for one of the distributions'''
    size=10000
    photons = Table({'col1': 2000 + np.random.normal(size=size),
                     'col2': 1000 + np.random.normal(size=size),
                     'ord': np.random.choice([0, 1, 2], size=size)})
    photons['col1'][photons['ord'] == 0] = 0
    res, pos, std = resolvingpower_from_photonlist_robust(photons, [1,2],
                                                          ['col1', 'col2'],
                                                          [None, 10000],
                                                          ordercol='ord')
    assert 2000 / 2.3548 * np.ones(2) == pytest.approx(res, rel=1e-1)
    assert np.ones(2) == pytest.approx(std, rel=1e-1)
    assert 2000 * np.ones(2) == pytest.approx(pos, rel=1e-1)



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