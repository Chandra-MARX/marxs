import numpy as np
from ..labSource import FarLabConstantPointSource as LabSource


def test_photon_generation():
	'''This is a test for the far source used with an aperture. It checks that
	the starting positions of the photons are within the aperture.
	'''
	pos = [1., 0., 0.]
	polar = [0., 0., 0.]
	rate = 10
	source = LabSource(pos, polar, rate, 5., 1.)
	
	center = [0., 0.5, 0.]
	v_y = [0., 1.0, 0.]
	v_z = [0., 0., 2.8]
	photons = source.generate_photons(1., center, v_y, v_z)
	for i in range (0, 10):
		assert photons['pos'][i][1] >= -0.5
		assert photons['pos'][i][1] <= 1.5
		assert photons['pos'][i][2] >= -2.8
		assert photons['pos'][i][2] <= 2.8
