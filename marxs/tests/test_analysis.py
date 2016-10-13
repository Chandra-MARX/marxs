import numpy as np
import transforms3d
from astropy.table import Table

from ..analysis import (measure_FWHM, find_best_detector_position,
                        resolvingpower_per_order)
from ..math.pluecker import e2h
from ..source import PointSource, FixedPointing
from ..simulator import Sequence
from ..optics import (CATGrating,
                      CircleAperture, PerfectLens, RadialMirrorScatter,
                      FlatDetector)
from ..design import RowlandTorus, GratingArrayStructure


def test_FWHM():
    '''For a Gaussian distributed variable the real stddev is close
    to the results from measure_FWHM.'''
    d = np.random.normal(size=1000)
    rel_diff = measure_FWHM(d) / (np.std(d) * 2*np.sqrt(2*np.log(2))) - 1.
    assert np.abs(rel_diff) < 0.05

def test_detector_position():
    '''Check that the optimal detector position is found at the convergent point.'''
    n = 1000
    convergent_point = np.array([3., 5., 7.])
    pos = np.random.rand(n, 3) * 100. + 10.
    dir = pos - convergent_point[np.newaxis, :]
    photons = Table({'pos': e2h(pos, 1), 'dir': e2h(dir, 0),
                     'energy': np.ones(n), 'polarization': np.ones(n), 'probability': np.ones(n)})
    opt = find_best_detector_position(photons)
    assert np.abs(opt.x - 3.) < 0.1

def test_resolvingpower_consistency():
    '''Compare different methods to measure the resolving power.

    This test only ensures consistency, not correctness. However, most of the
    underlying static function are implemented in other packages and tested there.

    This test requires a full pipeline to set up the input photons correctly and it
    thus also serves as an integration test.
    '''
    entrance = np.array([12000., 0., 0.])
    aper = CircleAperture(position=entrance, zoom=100)
    lens = PerfectLens(focallength=12000., position=entrance)
    rms = RadialMirrorScatter(inplanescatter=1e-4,
                              perpplanescatter=1e-5,
                              position=entrance)

    uptomirror = Sequence(elements=[aper, lens, rms])
    # CAT grating with blaze angle ensure that positive and negative orders
    # are defined the same way for all gratings in the GAS.
    blazeang = 1.91
    rowland = RowlandTorus(6000., 6000.)
    blazemat = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]), np.deg2rad(blazeang))
    gas = GratingArrayStructure(rowland=rowland, d_element=30.,
                                x_range=[1e4, 1.4e4],
                                radius=[50, 100],
                                elem_class=CATGrating,
                                elem_args={'d': 1e-4, 'zoom': [1., 10., 10.],
                                           'orientation': blazemat,
                                           'order_selector': None},
                            )
    star = PointSource(coords=(23., 45.), flux=5.)
    pointing = FixedPointing(coords=(23., 45.))
    photons = star.generate_photons(exposuretime=200)
    p = pointing(photons)
    p = uptomirror(p)

    o = np.array([0, -3, -6])

    res1 = resolvingpower_per_order(gas, p.copy(), orders=o, detector=None)
    res2 = resolvingpower_per_order(gas, p.copy(), orders=o, detector=rowland)
    res3 = resolvingpower_per_order(gas, p.copy(), orders=o, detector=FlatDetector(zoom=1000))

    # FWHM is similar
    assert np.isclose(res1[1][0], res2[1][0], atol=0.1)
    assert np.isclose(res1[1][1], res2[1][1], atol=0.2)  # differs stronger here if fit not good
    assert np.isclose(res2[1][0], 1.8, rtol=0.1, atol=0.1)
    # Resolution of 0th order is essentially 0
    assert np.isclose(res1[0][0], 0, atol=0.5)
    assert np.isclose(res2[0][0], 0, atol=0.5)
    assert np.isclose(res3[0][0], 0, atol=0.5)
    # Resolution of higher orders is consistent and higher
    assert np.isclose(res1[0][1], res2[0][1], rtol=0.1)
    assert np.isclose(res1[0][2], res2[0][2], rtol=0.2)
    assert np.isclose(res1[0][1], res3[0][1], rtol=0.1)
    # Resolution is higher at higher orders (approximately linear for small angles)
    assert np.isclose(res1[0][2], 2 * res1[0][1], rtol=0.2)
    assert np.isclose(res2[0][2], 2 * res2[0][1], rtol=0.2)
    # No test for res3 here, since that does not follow Rowland circle.
