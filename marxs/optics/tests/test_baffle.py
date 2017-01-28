# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Table
from ..baffle import Baffle


def test_photons_through():
    ''' tests that three photons go through or hit the baffle as predicted'''
    pos = np.array([[1., 0., 0., 1],
                    [1., 0., 0., 1],
                    [1., 0., 0., 1]])
    dir = np.array([[-1., 0., 0., 0],
                    [-1., 1.4, 0.5, 0],
                    [-1., -0.7, 0.2, 0]])
    photons = Table({'pos': pos, 'dir': dir, 'energy': [1, 1, 1], 'polarization': [1, 2, 3], 'probability': [0.5, 0.6, 0.7]})
    baf = Baffle(zoom=np.array([1., 1.5, 0.3]))
    photons = baf(photons)
    expected = np.array([0.5, 0., 0.7])
    assert np.all(photons['probability'] == expected)
