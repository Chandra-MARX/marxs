import os
import tempfile

import numpy as np
import pytest
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename

from marxs.design.tolerancing import (oneormoreelements,
                           wiggle, moveglobal, moveindividual, moveelem,
                           varyperiod, varyorderselector, varyattribute,
                           run_tolerances,
                           generate_6d_wigglelist,
                           select_1dof_changed,
                           plot_wiggle, load_and_plot,
                           run_tolerances_for_energies,
                           run_tolerances_for_energies2,
                           )
from marxs.optics import (FlatGrating, OrderSelector, RadialMirrorScatter,
                       RectangleAperture, ThinLens, FlatDetector)

from marxs.design import RowlandTorus, GratingArrayStructure
from marxs.utils import generate_test_photons
from marxs.source import PointSource, FixedPointing
from marxs.simulator import Sequence
from marxs.analysis.gratings import CaptureResAeff

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

mytorus = RowlandTorus(0.5, 0.5, position=[1.5, 0, -3])

def gsa(elem_class=FlatGrating):
    '''make a parallel structure - fresh for every test'''
    g = GratingArrayStructure(rowland=mytorus,
                              d_element=[0.1, 0.1], radius=[0.1,.2],
                              elem_class=elem_class,
                              elem_args={'zoom':0.2, 'd':0.002,
                                         'order_selector': OrderSelector([1])
                                     })
    return g

elempos = np.stack([e.pos4d for e in gsa().elements])

def test_oneormore():
    @oneormoreelements
    def func(a, b, c):
        a.value += 1

    class HoldData():
        def __init__(self, value):
            self.value = value

    obj1 = HoldData(2)
    obj2 = HoldData(4)
    obj3 = HoldData(6)

    listin = [obj2, obj3]

    # First, make sure that func works, otherwise the remaining test is useless.
    func(obj1, 2, c=4)
    assert obj1.value == 3

    func(listin, 'a', None)
    assert listin[0].value == 5
    assert listin[1].value == 7


@pytest.mark.parametrize('function', [wiggle, moveglobal, moveindividual])
def test_change_parallel_elements(function):
    '''Check that parameters work and elements are in fact changed.
    More detailed checks that the type of change is correct are
    implemented as separate tests, but those tests don't check out
    every parameter.
    '''
    g = gsa()
    function(g, 0., 0., 0.)
    assert np.all(np.stack([e.pos4d for e in g.elements]) == elempos)

    for key in ['dx', 'dy', 'dz', 'rx', 'ry', 'rz']:
        d = {key: 1.23}
        function(g, **d)
        assert not np.all(np.stack([e.pos4d for e in g.elements]) == elempos)


def test_moveelements_translate():
    '''Check that the element movers work. If the whole structure is translated
    or individual elements are translated by the same amount, the positions
    should be the same.'''
    g0 = gsa()
    g1 = gsa()
    g2 = gsa()
    moveglobal(g1, dy=-20)
    moveindividual(g2, dy=-20)
    assert np.allclose(np.stack([e.pos4d for e in g1.elements]),
                       np.stack([e.pos4d for e in g2.elements]))
    assert not np.allclose(np.stack([e.pos4d for e in g0.elements]),
                           np.stack([e.pos4d for e in g2.elements]))


def test_moveelements_rotate():
    '''Check that the element movers work.
    Unlike test_moveelements_translate we expect different results because
    there are different center of the rotation.
    This test does not check that the rotation is correct, only that its
    different because the validity of the rotation matrix itself is already
    covered by the tests in the transforms3d package.
    '''
    g1 = gsa()
    g2 = gsa()
    moveglobal(g1, rz=-1, ry=.2)
    moveindividual(g2, rz=-1, ry=.2)
    assert not np.allclose(np.stack([e.pos4d for e in g1.elements]),
                           np.stack([e.pos4d for e in g2.elements]))


def test_moveelem():
    '''Move an individual element'''
    det = FlatDetector(zoom=[1, 100, 100])
    assert det.geometry['center'][2] == 0
    moveelem(det, dz=5)
    assert det.geometry['center'][2] == 5
    assert np.all(det.geometry.pos4d[:3, :3] == np.eye(3) * [1, 100, 100])


def test_wiggle():
    '''Check wiggle function'''
    g = gsa()
    wiggle(g, dx=10, dy=.1)
    diff = elempos - np.stack([e.pos4d for e in g.elements])
    # Given the numbers, wiggle in x must be larger than y
    # This also tests that not all diff numbers are the same
    # (as they would be with move).
    assert np.std(diff[:, 0, 3]) > np.std(diff[:, 1, 3])


@pytest.mark.parametrize('function', [varyperiod, varyorderselector])
def test_errormessage(function):
    '''Check that check is performed for right type of object.
    Some function just set an attribute and there is no function call after
    that that would fail or do anything if called with the wrong type of object.
    Thus, it's very simple to call these with an object where it does not make
    any sense to apply them. So, they have some error check. Here, we check
    this check.
    '''
    with pytest.raises(ValueError) as e:
        # All functions accept two parameters.
        # Error should be raised before they are used, so the value does not
        # matter
        function(gsa, 1., 2.)

    assert 'does not have' in str(e.value)


def test_gratings_d():
    '''Change the grating constant.'''
    g = gsa()
    varyperiod(g.elements, 1., .1)
    periods = [e._d for e in g.elements]
    assert np.std(periods) > 0.01
    assert np.std(periods) < 5.
    assert np.mean(periods) > .5


def test_scatter():
    '''Check that the right properties are set.'''
    scat = RadialMirrorScatter(inplanescatter=1. * u.arcmin,
                               perpplanescatter=.1 * u.arcmin)
    varyattribute(scat, inplanescatter=2. * u.arcsec,
                  perpplanescatter=.2 * u.degree)
    assert scat.inplanescatter == 2. * u.arcsec
    assert scat.perpplanescatter == .2 * u.degree


def test_errormessage_attribute():
    '''Test error message for generic attributechanger'''
    with pytest.raises(ValueError) as e:
        # All functions accept two parameters.
        # Error should be raised before they are used, so the value does not
        # matter
        varyattribute(gsa, attributenotpresent=1., notpresenteither=2.)

    assert 'does not have' in str(e.value)


def test_orderselector():
    '''Test setting the order selector properties.'''
    photons = generate_test_photons(5)
    grat = FlatGrating(d=1., order_selector=OrderSelector([1]))
    p = grat(photons.copy())
    assert np.all(p['order'] == 1)

    varyorderselector(grat, OrderSelector, [2])
    p = grat(photons.copy())
    assert np.all(p['order'] == 2)


def test_runtolerances():
    '''Test the loop with mock functions.
    This is not a complete functional test, just making sure all calling
    signatures work.
    '''
    photons = generate_test_photons(20)
    grat = FlatGrating(d=1., order_selector=OrderSelector([1]))
    parameters =[{'order_selector': OrderSelector, 'orderlist': [2]},
                 {'order_selector': OrderSelector, 'orderlist': [1, 2], 'p': [.8, 0.]}]

    def afunc(photons):
        return {'meanorder': np.nanmean(photons['order'])}

    out = run_tolerances(photons, grat, varyorderselector, grat,
                         parameters, afunc)

    assert out[0]['meanorder'] == 2
    assert out[1]['meanorder'] == 1
    # check parameters are in output
    assert out[1]['orderlist'] == [1, 2]
    # check original parameter is still intact and can be used again
    # Regression test: If results are inserted into the same dict
    # 'meanorder' will appear which is not valid for varyorderselector
    assert 'meanorder' not in parameters[0]

def test_run_tolerances_for_energies():
    '''For this test, we need to define an instrument. The instrument is not
    very realistic (an X-ray mirror with r=0 won't work), but the
    point here is just to check the tolerancing for several
    energies. To make that calculation reasonably fast, we need to
    keep the number of elements in the optical system small.
    '''
    coords = SkyCoord(12. * u.deg, -45 * u.deg)
    src = PointSource(coords=coords)
    pnt = FixedPointing(coords=coords)
    aper = RectangleAperture(position=[5000, 0, 0], zoom=[1, 10, 10])
    lens = ThinLens(position=[4900, 0, 0], zoom=[1, 10, 10], focallength=4900)
    grat = FlatGrating(d=.002, order_selector=OrderSelector([0, 1]),
                       position=[4800, 0, 0], zoom=[1, 10, 10])
    det = FlatDetector(zoom=[1, 100, 100])
    instrum = Sequence(elements=[pnt, aper, lens, grat, det])

    parameters = [{'period_mean': 0.003, 'period_sigma': 0.},
                  {'period_mean': 0.004, 'period_sigma': 0.}]

    res = run_tolerances_for_energies(src, [.1, 1] * u.keV,
                                      Sequence(elements=[pnt, aper, lens]),
                                      Sequence(elements=[grat, det]),
                                      varyperiod, grat,
                                      parameters,
                                      CaptureResAeff(orders=[0, 1, 2]),
                                      reset={'period_mean': 0.005,
                                             'period_sigma': 0.},
                                      t_source=1. * u.ks)
    # Check the reset worked
    assert grat._d == 0.005
    # Check both energy have been calculated
    assert 1 in res['energy']
    assert .1 in res['energy']
    assert len(res) == 4
    # check results are reasonable
    assert np.all(res['R'].data[:, 0] == 0)
    assert not np.any(np.isfinite(res['R'].data[:, 2]))
    assert res['R'].data[2, 1] > res['R'].data[0, 1]


def test_run_tolerances_for_energies2():
    '''Same, as above, but with different calling sequence
    '''
    coords = SkyCoord(12. * u.deg, -45 * u.deg)
    src = PointSource(coords=coords)
    pnt = FixedPointing(coords=coords)
    aper = RectangleAperture(position=[5000, 0, 0], zoom=[1, 10, 10])
    lens = ThinLens(position=[4900, 0, 0], zoom=[1, 10, 10], focallength=4900)
    grat = FlatGrating(d=.002, order_selector=OrderSelector([0, 1]),
                       position=[4800, 0, 0], zoom=[1, 10, 10])
    det = FlatDetector(zoom=[1, 100, 100])
    instrum = Sequence(elements=[pnt, aper, lens, grat, det])

    parameters = [{'period_mean': 0.003, 'period_sigma': 0.},
                  {'period_mean': 0.004, 'period_sigma': 0.}]

    res = run_tolerances_for_energies2(src, [.1, 1] * u.keV,
                                       instrum, FlatGrating,
                                       varyperiod,
                                       parameters,
                                       CaptureResAeff(orders=[0, 1, 2]),
                                       reset={'period_mean': 0.005,
                                              'period_sigma': 0.},
                                       t_source=1. * u.ks)
    # Check the reset worked
    assert grat._d == 0.005
    # Check both energy have been calculated
    assert 1 in res['energy']
    assert .1 in res['energy']
    assert len(res) == 4
    # check results are reasonable
    assert np.all(res['R'].data[:, 0] == 0)
    assert not np.any(np.isfinite(res['R'].data[:, 2]))
    assert res['R'].data[2, 1] > res['R'].data[0, 1]


def test_6dlist():
    '''Check the list of dicts in 3 translations dof and 3 rotations'''
    cglob, cind = generate_6d_wigglelist([0, 1.] * u.cm, [0., 1.] * u.degree,
                            names=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    assert len(cind) == 7
    assert len(cglob) == 13
    assert set(cind[5].keys()) == set(['x', 'y', 'z', 'rx', 'ry', 'rz'])

    tab = Table(cind)
    for col in ['x', 'y', 'z']:
        assert np.max(tab[col]) == 10
        assert np.min(tab[col]) == 0

    for col in ['x', 'y', 'z']:
        assert np.max(tab[col]) == 10
        assert np.min(tab[col]) == 0

    tab = Table(cglob)
    for col in tab.colnames:
        assert - np.min(tab[col]) == np.max(tab[col])

def test_6d_warning():
    with pytest.warns(UserWarning):
        cglob, cind = generate_6d_wigglelist([1.] * u.cm, [0., 1.] * u.degree)

def test_find_changed():
    '''Test that we find the row where only one parameter was changed.'''
    tab = Table({'par1': [-1, -1, 0, 0, 0, 1],
                 'par2': [-1,  0, 0, 3, 0, 0],
                 'id':   [ 0,  1, 2, 3, 4, 5]})
    t = select_1dof_changed(tab, 'par1', parlist=['par1', 'par2'])
    assert set(t['id']) == set([1, 2, 4, 5])

@pytest.mark.skipif('not HAS_MPL')
def test_plot_wiggle():
    '''Test that wiggle plot works. This does not test that the result
    looks correct, only that running through the plot function does not
    raise any errors.
    This is one of the few plotting functions in the entire package, so
    setting up the infrastructure to compare output pixel-by-pixel does not
    seem worth it as this point.
    '''
    fig, ax = plt.subplots()

    tab = Table({'wave': [1, 1],
                 'dd': [0, 1],
                 'Rgrat': [500, 500],
                 'Aeff': [20, 50]})
    plot_wiggle(tab, 'dd', ['dd'], ax, Aeff_col='Aeff')

@pytest.mark.skipif('not HAS_MPL')
def test_plot_wiggle_exception():
    '''Test that exception is raised for parameters not plotted.
    '''
    fig, ax = plt.subplots()

    tab = Table({'wave': [1, 1],
                 'qwe': [0, 1],
                 'Rgrat': [500, 500],
                 'Aeff': [20, 50]})
    with pytest.raises(ValueError, match='Parameter names should start with'):
        plot_wiggle(tab, 'qwe', ['qwe'], ax, Aeff_col='Aeff')

@pytest.mark.skipif('not HAS_MPL')
def test_plot_6dof():

    tab = Table({'wave': [1, 1, 2, 2, 1, 1, 1, 1],
                 'dd': [0, 1, 0, 2, 0, 0, 0, 0],
                 'rr': [0, 0, 0, 0, 2, 4, 6, 8],
                 'R': np.random.rand(8),
                 'Aeffgrat': np.arange(8)})

    with tempfile.TemporaryDirectory() as tmpdirname:
        name = os.path.join(tmpdirname, 'var_global.fits')
        tab.write(name)
        fig, ax = load_and_plot(name, ['dd', 'rr'], R_col='R')

@pytest.mark.skipif('not HAS_MPL')
def test_plot_6dof_real_file():
    '''Repeat previous test with static data file. This is a more realistic file
    but it takes too long to generate every time. This file is used in the docs
    in design/tolerancing (see docs/pyplot/chandra_tolerancing) so if this test
    breaks, the docs will likely have to be changed, too.
    '''
    filename = get_pkg_data_filename('data/wiggle_global.fits', 'marxs.design.tests')
    fig, ax = load_and_plot(filename)
