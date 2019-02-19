# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from scipy.stats import normaltest
import pytest

from ..scatter import RadialMirrorScatter, RandomGaussianScatter
from ..detector import FlatDetector
from ...utils import generate_test_photons

def test_distribution_of_scattered_rays():
    '''Check that scattered rays have a normal distribution.'''
    photons = generate_test_photons(500)
    photons['pos'] = np.tile(np.array([0., 1., 0., 1.]), (500, 1))
    rms = RadialMirrorScatter(inplanescatter=0.1)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)

    assert np.all(p['det_y'] == 0)
    stat, pval = normaltest(p['det_x'])
    assert pval > 0.5
    assert np.allclose(np.std(p['det_x']), np.arctan(0.1), rtol=0.1)

def test_distribution_scatered_rays_turned():
    '''Test that the in-plane scattering plane is correctly calculated.

    The photons are positioned 90 deg from the example above.
    Also, do some perpendicular to the plan scattering, that's missing
    from the test above.'''
    photons = generate_test_photons(500)
    photons['pos'] = np.tile(np.array([0., 0., 1., 1.]), (500, 1))
    rms = RadialMirrorScatter(inplanescatter=0.1, perpplanescatter=0.01)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)

    # This is in-plane-scatter
    assert np.allclose(np.std(p['det_y']), np.arctan(0.1), rtol=0.1)
    # This is scatter perpendicular to the plane.
    assert np.allclose(np.std(p['det_x']), np.arctan(0.01), rtol=0.1)

def test_scatter_0_01():
    '''Test that scatter works if one or both scatter directions are set
    to 0. These cases are special cased, so that need an extra test.'''
    photons = generate_test_photons(500)
    photons['pos'] = np.tile(np.array([0., 1., 0., 1.]), (500, 1))
    rms = RadialMirrorScatter(inplanescatter=0, perpplanescatter=0.1)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)

    assert np.all(p['det_x'] == 1.0)
    assert np.allclose(np.std(p['det_y']), np.arctan(0.1), rtol=0.1)

def test_scatter_0_0():
    '''Test that scatter works if one or both scatter directions are set
    to 0. These cases are special cased, so that need an extra test.'''
    photons = generate_test_photons(500)
    photons['pos'] = np.tile(np.array([0., 1., 0., 1.]), (500, 1))
    rms = RadialMirrorScatter(inplanescatter=0, perpplanescatter=0)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)

    assert np.all(p['det_x'] == 1.0)
    assert np.all(p['det_y'] == 0)


def test_gaussscatter_0():
    '''Test that scatter works if scatter sigma set
    to 0. This case is special cased, so that needs an extra test.'''
    photons = generate_test_photons(500)
    photons['pos'] = np.tile(np.array([0., 1., 0., 1.]), (500, 1))
    rms = RandomGaussianScatter(scatter=0)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)

    assert np.all(p['det_x'] == 1.0)
    assert np.all(p['det_y'] == 0)


def test_gaussiandistribution_of_scattered_rays():
    '''Check that scattered rays have a normal distribution.'''
    photons = generate_test_photons(500)
    photons['pos'] = np.tile(np.array([0., 1., 0., 1.]), (500, 1))
    rms = RandomGaussianScatter(scatter=1e-5)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)
    # Calculate distance from undisturbed position
    d = np.sqrt((p['det_x'] - np.mean(p['det_x']))**2 +
                (p['det_y'] - np.mean(p['det_y']))**2)
    d = d * np.sign(p['det_x'] - np.mean(p['det_x']))

    stat, pval = normaltest(d)
    assert pval > 0.02
    assert np.isclose(np.std(d), 1e-5, rtol=0.1)


def test_scatteredfunction():
    '''Check that scattered rays have a normal distribution.'''
    photons = generate_test_photons(500)
    photons['pos'] = np.tile(np.array([0., 1., 0., 1.]), (500, 1))
    photons['time'] = np.arange(len(photons), dtype=float) / 500.

    def timescatter(photons, intersect, interpos, intercoos):
        return photons['time']

    rms = RandomGaussianScatter(scatter=timescatter)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)
    # Calculate distance from undisturbed position
    d = np.sqrt((p['det_x'] - np.mean(p['det_x']))**2 +
                (p['det_y'] - np.mean(p['det_y']))**2)
    d = d * np.sign(p['det_x'] - np.mean(p['det_x']))

    assert np.std(d[:200]) < 2 * np.std(d[:300])


def test_scatterarg_missing():
    '''Scatter argument is required, unless overriden by a derived class.'''
    with pytest.raises(ValueError) as e:
        rms = RandomGaussianScatter()

    assert 'Keyword "scatter" missing.' == e.value.message

def test_derivedScatterClass():
    '''Check that overriding defaul scatters works.'''
    class Derived(RandomGaussianScatter):
        scatter = 1

    # This should work without scatter keyword
    rms = Derived()

    # This should raise a warning, but set the value
    with pytest.warns(UserWarning, match="Overriding class level"):
        rms = Derived(scatter=5)
    assert rms.scatter == 5
