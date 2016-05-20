import numpy as np
from astropy.table import Table

from ..analysis import measure_FWHM, find_best_detector_position
from ..math.pluecker import e2h

def test_FWHM():
    '''For a Gaussian distributed variable the real stddev is close to the results from measure_FWHM.'''
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
