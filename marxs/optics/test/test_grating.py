from StringIO import StringIO
import numpy as np
from numpy.random import random
from astropy.table import Table

from ..grating import (FlatGrating,
                       constant_order_factory, uniform_efficiency_factory, EfficiencyFile)
from ...math.pluecker import h2e
from ... import energy2wave

def test_zeros_order():
    '''Photons diffracted into order 0 should just pass through'''
    photons = Table({'pos': random((10, 4)) * 10 - 5,
                     'dir': random((10, 4)) * 10 - 5,
                     'energy': random(10),
                     'polarization': random(10),
                     'probability': np.ones(10),
                     })
    # Make sure homogeneous coordiantes are valid
    # and infall is normal
    photons['pos'][:, 3] = 1
    photons['dir'][:, 1:] = 0

    p = photons.copy()
    def mock_order(x, y):
        return np.zeros_like(x, dtype=np.int), np.ones_like(x)
    g0 = FlatGrating(d=1./500.,
                     order_selector=mock_order,
                     zoom=np.array([1., 5., 5.]))
    p = g0.process_photons(p)
    # Direction unchanged
    d_in = h2e(photons['dir'])
    d_out = h2e(p['dir'])
    # normalize
    d_in =  d_in / np.sqrt(np.sum(d_in**2, axis=-1))[:, None]
    d_out =  d_out / np.sqrt(np.sum(d_out**2, axis=-1))[:, None]
    assert np.allclose(d_in, d_out)
    # all intersection points in y-z plane
    assert np.allclose(p['pos'][:, 0], 0.)
    # no offset between old and new ray
    assert np.allclose(np.cross(h2e(photons['pos']) - h2e(p['pos']), h2e(p['dir'])), 0)

def test_order_dependence():
    '''For small theta, the change in direction is m * dtheta'''
    photons = Table({'pos': np.ones((5,4)),
                     'dir': np.tile([1., 0, 0, 0], (5,1)),
                     'energy': np.ones(5),
                     'polarization': np.ones(5),
                     'probability': np.ones(5),
                     })
    def mock_order(x, y):
        return [-2, -1, 0, 1, 2], np.ones(5)
    g = FlatGrating(d=1./500, order_selector=mock_order)
    p = g.process_photons(photons)
    # grooves run in y direction
    assert np.allclose(p['dir'][:, 1], 0.)
    # positive and negative orders are mirrored
    assert np.allclose(p['dir'][[0,1], 2], - p['dir'][[4,3], 2])
    # order 2 is twice as far as order 1 (for theta  small)
    assert np.abs(p['dir'][4, 2] / p['dir'][3, 2] - 2 ) < 0.00001

def test_energy_dependence():
    photons = Table({'pos': np.ones((5,4)),
                     'dir': np.tile([1., 0, 0, 0], (5,1)),
                     'energy': np.arange(1., 6),
                     'polarization': np.ones(5),
                     'probability': np.ones(5),
                     })
    g = FlatGrating(d=1./500, order_selector=constant_order_factory(1))
    p = g.process_photons(photons)
    # grooves run in y direction
    assert np.allclose(p['dir'][:, 1], 0.)
    # n lambda = d sin(theta)
    lam = energy2wave / p['energy']
    theta = np.arctan2(p['dir'][:, 2], p['dir'][:, 0])
    assert np.allclose(1. * lam, 1./500. * np.sin(theta))

def test_uniform_efficiency_factory():
    '''Uniform efficiency factory should give results from -order to +order'''
    f = uniform_efficiency_factory(2)
    # Make many numbers, to reduce chance of random failure for this test
    testout = f(np.ones(10000), np.ones(10000))
    assert set(testout[0]) == set([-2, -1 , 0, 1, 2])
    assert len(testout[0]) == len(testout[1])

def test_constant_order_factory():
    '''Constant order will give always the same order'''
    f = constant_order_factory(2)
    testout = f(np.ones(15), np.ones(15))
    assert np.all(testout[0] == 2)
    assert len(testout[0]) == len(testout[1])

def test_EfficiencyFile():
    '''Interactively, I see that the probability is not chosen right.

    '''
    data  = StringIO(".5 .1 .1 .1 .4\n1. .1 .1 .1 .5\n1.5 0. .1 .0 .5")
    eff = EfficiencyFile(data, [1, 0, -1, -2])
    testout = eff(np.array([.7, 1.1]), np.zeros(2))
    assert np.allclose(testout[1], np.array([.7, .8]))

    testout = eff(1.6 * np.ones(1000), np.zeros(1000))
    assert np.allclose(testout[1], .6)
    assert set(testout[0]) == set([-2 , 0])
    assert (testout[0] == 0).sum()  < (testout[0] == -2).sum()
