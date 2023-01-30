import numpy as np
from astropy.table import Table
import pytest

from ...utils import generate_test_photons
from ...optics import ThinLens, FlatGrating, FlatDetector, OrderSelector
from ..gratings import (resolvingpower_from_photonlist,
                        resolvingpower_from_photonlist_robust,
                        effectivearea_from_photonlist,
                        identify_photon_in_subaperture,
                        average_R_Aeff,
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
    res, pos, std = resolvingpower_from_photonlist_robust([photons, photons],
                                                          orders,
                                                          ['col1', 'col2'],
                                                          [0, 0])
    assert 1000 / 2.3548 * np.ones(3) == pytest.approx(res, rel=1e-1)
    assert np.ones(3) == pytest.approx(std, rel=1e-1)
    assert 1000 * np.ones(3) == pytest.approx(pos, rel=1e-1)

    res, pos, std = resolvingpower_from_photonlist_robust([photons['order', 'col1'],
                                                           photons['order', 'col2']],
                                                           orders,
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
    res, pos, std = resolvingpower_from_photonlist_robust([photons, photons],
                                                          [1,2],
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


def test_identify_photon_in_subaperture():
    angle = [0, 59, 61, 180, 289]
    inaper = identify_photon_in_subaperture(np.deg2rad(angle),
                                            np.deg2rad(30), ang_0=np.pi / 2)
    assert np.all(inaper == [False, False, True, False, True])


def test_average_R_Aeff():
    r_out, aeff_out = average_R_Aeff(np.array([np.nan, np.nan]),
                                     np.array([1, 2]))
    assert r_out is np.ma.masked
    assert aeff_out == 3
    r_out, aeff_out = average_R_Aeff(np.array([5., 5.]),
                                     np.array([1, 2]))
    assert r_out == pytest.approx(5)
    r_out, aeff_out = average_R_Aeff(np.array([5., 2.]),
                                     np.array([1, 2]))
    assert r_out == pytest.approx(3)

    nan = np.nan
    r = np.array([
       [[3.72047331e+03, nan, nan, 1.22772588e-02],
        [3.68709325e+03,            nan,            nan, 5.16674211e-02],
        [3.78030863e+03,            nan,            nan, 5.68255010e-02],
        [3.67658767e+03,            nan, 1.20839326e+03, 2.84897083e-02]],

       [[3.63193038e+03,            nan,            nan, 6.43244337e-05],
        [           nan,            nan,            nan, 2.55943724e-02],
        [3.81094245e+03,            nan,            nan, 3.20472833e-02],
        [3.62704572e+03,            nan,            nan, 4.04451453e-02]]])
    aeff = np.array([
       [[45.43284984,  0.        ,  0.        ,  4.14357052],
        [46.01057067,  0.        ,  0.        ,  3.95845344],
        [45.35928564,  0.        ,  0.        ,  4.00130456],
        [44.54923056,  0.        ,  8.66382063,  4.11383244]],

       [[44.93080722,  0.        ,  0.        ,  4.35276434],
        [ 0.        ,  0.        ,  0.        ,  3.96159335],
        [45.1660785 ,  0.        ,  0.        ,  3.68142596],
        [44.43211355,  0.        ,  0.        ,  4.3622082 ]]])
    expected_r = np.ma.masked_invalid(
       [[3716.1898154882915, nan, 1208.3932636420263, 0.03699616674607531],
        [3690.417678741742, nan, nan, 0.024213505777004152]])
    expected_aeff = np.array([
        [181.35193671,   0.        ,   8.66382063,  16.21716095],
        [134.52899927,   0.        ,   0.        ,  16.35799185]])
    r_out, aeff_out =  average_R_Aeff(r, aeff, axis=1)
    assert np.allclose(r_out, expected_r, equal_nan=True)
    assert np.allclose(aeff_out, expected_aeff, equal_nan=True)
