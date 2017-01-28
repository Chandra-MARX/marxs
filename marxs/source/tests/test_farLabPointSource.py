# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from ..labSource import FarLabPointSource as LabSource


def test_photon_generation():
	'''This is a test for the far source used with an aperture. It checks that
	the starting positions of the photons are within the aperture.
	'''
	pos = [1., 0., 0.]
	rate = 10
	center = [0., 0.5, 0.]
	source = LabSource(pos, flux=rate, energy=5., position = center, zoom = [1., 1., 2.8])

	photons = source.generate_photons(1.)
	for i in range (0, 10):
		assert photons['pos'][i][1] >= -0.5
		assert photons['pos'][i][1] <= 1.5
		assert photons['pos'][i][2] >= -2.8
		assert photons['pos'][i][2] <= 2.8
