# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import pytest
from scipy.stats import normaltest
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
import astropy.units as u

from ...source import (SymbolFSource, Source, poisson_process,
                       PointSource,
                       DiskSource, SphericalDiskSource, GaussSource)
from ...optics import RectangleAperture
from ..source import SourceSpecificationError


def test_photons_header():
    '''All generated photons should have some common keywords.

    Just test that some of the keywords are there. It's unlikely
    that I need a full list here.
    '''
    s = SymbolFSource(coords=SkyCoord(-123., -43., unit=u.deg), size=1.*u.deg)
    photons = s.generate_photons(5. * u.s)
    for n in ['EXPOSURE', 'CREATOR', 'MARXSVER', 'SIMTIME', 'SIMUSER']:
        assert n in photons.meta

def test_energy_input_default():
    '''For convenience and testing, defaults for time, energy and pol are set.'''
    s = Source()
    photons = s.generate_photons(5. * u.s)
    assert len(photons) == 5
    assert np.all(photons['energy'] == 1.)
    assert len(set(photons['polangle'])) == 5  # all pol angles are different

def test_flux_input():
    '''Options: contant rate or function'''
    # 1. constant rate
    s = Source(flux=5 / u.hour / u.cm**2)
    photons = s.generate_photons(5. * u.h)
    assert len(photons) == 25
    delta_t = np.diff(photons['time'])
    assert np.allclose(delta_t, delta_t[0]) # constante rate

    # 2. function
    def f(t, geomarea):
        return np.logspace(1, np.log10(t.to(u.s).value)) * u.s

    s = Source(flux=f)
    photons = s.generate_photons(100 * u.s)
    assert np.all(photons['time'] == np.logspace(1, 2))

    # 3. anything else
    s = Source(flux=ValueError())
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5 * u.s)
    assert '`flux` must be' in str(e.value)

def test_energy_input():
    '''Many different options ...'''
    # 1. contant energy
    s = PointSource(coords=SkyCoord("1h12m43.2s +1d12m43s"), energy=2. * u.keV)
    photons = s.generate_photons(5 * u.minute)
    assert np.all(photons['energy'] == 2.)

    # 2. function
    def f(t):
        return (t / u.s * u.keV).decompose()
    s = Source(energy=f)
    photons = s.generate_photons(5 * u.s)
    assert np.all(photons['energy'] == photons['time'])
    # bad function
    s = Source(energy=lambda x: np.array([np.sum(x.value)]) * u.erg)
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5 * u.s)
    assert 'an array of same size as' in str(e.value)

    # 3. array, table, recarray, dict, ... just try some of of the opion with keys
    # spectrum with distinct lines
    engrid = [0.5, 1., 2., 3.] * u.keV
    # first entry (456) will be ignored
    fluxgrid = [456., 1., 0., 2.] / u.s / u.cm**2
    s = Source(energy=QTable({'energy': engrid, 'flux': fluxgrid}))

    photons = s.generate_photons(1000 * u.s)
    en = photons['energy']
    ind0510 = (en >= 0.5) & (en <=1.0)
    ind2030 = (en >= 2.) & (en <=3.0)
    assert (ind0510.sum() + ind2030.sum()) == len(photons)
    assert ind0510.sum() < ind2030.sum()

    # 4. anything else
    s = Source(energy=object())
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5 * u.s)
    assert '`energy` must be' in str(e.value)

def test_polarization_input():
    '''Many differnet options ...'''
    # 1. 100 % polarized flux
    s = Source(polarization=90 * u.deg)
    photons = s.generate_photons(5 * u.s)
    assert np.all(photons['polangle'] == np.pi / 2)

    # 2. function
    def f(t, en):
        return (t / u.s * en / u.keV * u.rad).decompose()
    s = Source(polarization=f, energy=2. * u.keV)
    photons = s.generate_photons(5 * u.s)
    assert np.all(photons['polangle'] == photons['time'] * photons['energy'])
    # bad function
    s = Source(polarization=lambda x, y: [np.dot(x.value, y.value)] * u.deg)
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5 * u.s)
    assert 'an array of same size as' in str(e.value)

    # 3.table
    # spectrum with distinct lines
    polgrid = [0.5, 1., 2., 3.] * u.rad
    probgrid = [456., 1., 0., 2.]  # first entry (456) will be ignored
    s = Source(polarization=QTable({'angle': polgrid, 'probability': probgrid}))
    photons = s.generate_photons(1000 * u.s)
    pol = photons['polangle']
    ind0510 = (pol >= 0.5) & (pol <=1.0)
    ind2030 = (pol >= 2.) & (pol <=3.0)
    assert (ind0510.sum() + ind2030.sum()) == len(photons)
    assert ind0510.sum() < ind2030.sum()

    # 4. None (unpolarized source)
    s = Source(polarization=None)
    photons = s.generate_photons(5 * u.s)
    assert len(set(photons['polangle'])) == len(photons) # all different

    # 5. anything else
    s = Source(polarization=object())
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5 * u.s)
    assert '`polarization` must be' in str(e.value)

def test_poisson_process():
    '''Do some consistency checks for the Poisson process.

    It turns out that this is hard to test properly, without reimplemention the
    scipy version.
    '''
    p = poisson_process(20. / (u.s * u.cm**2))
    times = p(100. * u.s, 1. * u.cm**2).value
    assert (len(times) > 1500) and (len(times) < 2500)
    assert (times[-1] > 99.) and (times[-1] < 100.)

    times = p(100. * u.s, 10. * u.cm**2).value
    assert (len(times) > 18000) and (len(times) < 22000)
    assert (times[-1] > 99.) and (times[-1] < 100.)

def test_poisson_input():
    '''Input must be a scalar.'''
    with pytest.raises(ValueError) as e:
        p = poisson_process(np.arange(5) / u.s / u.cm**2)
    assert 'must be scalar' in str(e.value)

def test_Aeff():
    '''Check that a higher effective area leads to more counts.'''
    a = RectangleAperture(zoom=2)
    s = Source(flux=50 / u.s / u.cm**2, geomarea=a.area)
    photons = s.generate_photons(5. * u.s)
    assert len(photons) == 40

def test_disk_radius():
    pos = SkyCoord(12. * u.degree, -23.*u.degree)
    s = DiskSource(coords=pos, a_inner=50.* u.arcsec,
                   a_outer=10. * u.arcmin)

    photons = s.generate_photons(1e4 * u.s)
    d = pos.separation(SkyCoord(photons['ra'], photons['dec'], unit='deg'))
    assert np.max(d.arcmin <= 10.)
    assert np.min(d.arcmin >= 0.8)

def test_sphericaldisk_radius():
    pos = SkyCoord(12. * u.degree, -23.*u.degree)
    s = SphericalDiskSource(coords=pos, a_inner=50.* u.arcsec,
                   a_outer=10. * u.arcmin)

    photons = s.generate_photons(1e4 * u.s)
    d = pos.separation(SkyCoord(photons['ra'], photons['dec'], unit='deg'))
    assert np.max(d.arcmin <= 10.)
    assert np.min(d.arcmin >= 0.8)

@pytest.mark.parametrize("diskclass,diskpar,n_expected",
                         [(DiskSource, {'a_outer': 30. * u.arcmin}, 2800),
                          (SphericalDiskSource, {'a_outer': 30. * u.arcmin}, 2800),
                          (GaussSource, {'sigma': 1.5 * u.deg}, 150)])
def test_disk_distribution(diskclass, diskpar, n_expected):
    '''This is a separate test from test_disk_radius, because it's a simpler
    to write if we don't have to worry about the inner hole.

    For the test itself: The results should be poisson distributed (or, for large
    numbers this will be almost normal).
    That makes testing it a little awkard in a short run time, thus the limits are
    fairly loose.

    This test is run for several extended sources, incl Gaussian. Stirctly speaking
    it should fail for a Gaussian distribution, but if the sigma is large enough it
    will pass a loose test (and still fail if things to catastrophically wrong,
    e.g. some test circles are outside the source).
    '''

    s = diskclass(coords=SkyCoord(213., -10., unit=u.deg), **diskpar)
    photons = s.generate_photons(1e5 * u.s)

    n = np.empty(20)
    for i in range(len(n)):
        circ = SkyCoord((213. +  np.random.uniform(-0.1, .1)) * u.degree,
                       (- 10. + np.random.uniform(-0.1, .1)) * u.degree)
        d = circ.separation(SkyCoord(photons['ra'], photons['dec'], unit='deg'))
        n[i] = (d < 5. * u.arcmin).sum()
    s, p = normaltest(n)
    # assert a p value here that is soo small that it's never going to be hit
    # by chance.
    assert p > .05
    # better: Test number of expected photons matches
    # Allow large variation so that this is not triggered by chance
    assert np.isclose(n.mean(), n_expected, rtol=.2)
