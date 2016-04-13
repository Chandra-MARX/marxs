import numpy as np
from astropy.table import Table
from scipy.stats import normaltest

from ..scatter import RadialMirrorScatter
from ..detector import FlatDetector

def test_distribution_of_scattered_rays():
    '''Check that scattered rays have a normal distribution.'''
    n = 500
    dir = np.tile(np.array([-1., 0., 0., 0.]), (n, 1))
    pos = np.tile(np.array([0., 1., 0., 1.]), (n, 1))
    photons = Table({'pos': pos,
                     'dir': dir,
                     'energy': np.ones(n),
                     'polarization': np.ones(n),
                     'probability': np.ones(n),
                     })
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
    n = 500
    dir = np.tile(np.array([-1., 0., 0., 0.]), (n, 1))
    pos = np.tile(np.array([0., 0., 1., 1.]), (n, 1))
    photons = Table({'pos': pos,
                     'dir': dir,
                     'energy': np.ones(n),
                     'polarization': np.ones(n),
                     'probability': np.ones(n),
                     })
    rms = RadialMirrorScatter(inplanescatter=0.1, perpplanescatter=0.01)
    det = FlatDetector(position=[-1, 0, 0], zoom=1000)

    p = rms(photons)
    p = det(p)

    # This is in-plane-scatter
    assert np.allclose(np.std(p['det_y']), np.arctan(0.1), rtol=0.1)
    # This is scatter perpendicular to the plane.
    assert np.allclose(np.std(p['det_x']), np.arctan(0.01), rtol=0.1)
