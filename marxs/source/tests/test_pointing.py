import numpy as np
from astropy.table import Table
from scipy.stats import pearsonr

from ...source import FixedPointing, JitterPointing


def test_geomarea_projection():
    '''When a ray sees the aperture under an angle the projected aperture size
    is smaller. This is accounted for by reducing the probability of this photon.'''
    photons = Table()
    photons['ra'] = [0., 45., 90., 135., 315.]
    photons['dec'] = np.zeros(5)
    photons['probability'] = np.ones(5)
    photons['time'] = np.arange(5)
    photons['polangle'] = np.zeros(5)

    fp = FixedPointing(coords=(0., 0.))
    p = fp(photons.copy())

    assert np.allclose(p['probability'], [1., 1./np.sqrt(2), 0, 0, 1./np.sqrt(2)])

    fp = FixedPointing(coords=(0., 0.), geomarea_reference=[0., -2., 0., 0.])
    p = fp(photons.copy())

    assert np.allclose(p['probability'], [ 0, 1./np.sqrt(2), 1., 1./np.sqrt(2), 0.])


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
    prob = np.ones_like(ra)
    photons = Table([ra, dec, time, pol, prob],
                    names=['ra', 'dec', 'time', 'polangle', 'probability'])
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

def test_arcsecinsteadofradianrwarning(recwarn):
    jittered = JitterPointing(coords = (25., -10.), jitter=1./3600.)
    w = recwarn.pop()
    assert 'jitter is expected in radian' in str(w.message)
