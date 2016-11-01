import numpy as np
from astropy.table import Table
from scipy.stats import pearsonr

from ..source import FixedPointing, JitterPointing


def test_jitter():
    '''test size and randomness of jitter'''
    n = 100000
    ra = np.random.rand(n) * 2 * np.pi
    # Note that my rays are not evenly distributed on the sky.
    # No need to be extra fancy here, it's probably even better for
    # a test to have more scrutiny around the poles.
    dec = (np.random.rand(n) * 2. - 1.) / 2. * np.pi
    time = np.arange(n)
    pol = np.ones_like(ra)
    photons = Table([ra, dec, time, pol], names=['ra', 'dec', 'time', 'polangle'])
    fixed = FixedPointing(coords = (25., -10.))
    jittered = JitterPointing(coords = (25., -10.), jitter=np.deg2rad(1./3600.))
    p_fixed = fixed(photons.copy())
    p_jitter = jittered(photons)

    assert np.allclose(np.linalg.norm(p_fixed['dir'], axis=1), 1.)
    assert np.allclose(np.linalg.norm(p_jitter['dir'], axis=1), 1.)

    prod = np.sum(p_fixed['dir'] * p_jitter['dir'], axis=1)
    # sum can give values > 1 due to rounding errors
    # That would make arccos fail, so catch those here
    ind = prod > 1.
    if ind.sum() > 0:
        prod[ind] = 1.

    alpha = np.arccos(prod)
    # in this formula alpha will always be the abs(angle).
    # Flip some signs to recover input normal distribtution.
    alpha *= np.sign(np.random.rand(n)-0.5)
    # center?
    assert np.abs(np.mean(alpha)) * 3600. < 0.01
    # Is right size?
    assert np.std(np.rad2deg(alpha)) > (0.9 / 3600.)
    assert np.std(np.rad2deg(alpha)) < (1.1 / 3600.)
    # Does it affect y and z independently?
    coeff, p = pearsonr(p_fixed['dir'][:, 1] - p_jitter['dir'][:, 1],
                        p_fixed['dir'][:, 2] - p_jitter['dir'][:, 2])
    assert abs(coeff) < 0.01
