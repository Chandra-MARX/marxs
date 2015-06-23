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
	photons = Table({'pos': pos, 'dir': dir, 'energy': [1, 1, 1], 'polarization': [1, 2, 3], 'probability': [1., 1., 1.]})
	baf = Baffle(zoom=np.array([1., 1.5, 0.3]))
	photons = baf.process_photons(photons)
	expected = np.array([1.0, 0.0, 1.0])
	assert np.all(photons['probability'] == expected)
