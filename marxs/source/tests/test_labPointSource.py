import numpy as np
from ..labSource import LabPointSource as LabSource
from ..labSource import LabPointSourceCone

def test_photon_generation():
	'''This tests the lab point source. It checks that the starting points are all
	at the sources position.
	'''
	pos = [1., 1., 1.]
	rate = 10
	source = LabSource(pos, flux=rate, energy=5.)

	photons = source.generate_photons(1.)
	assert np.all(photons['pos'] == np.ones([10, 4]))

def test_photon_direction():
	'''This tests the lab point source. It checks the optional 'direction' parameter.
	'''
	pos = [1., 1., 1.]
	rate = 10
	source = LabSource(pos, flux=rate, energy=5., direction = '-y')

	photons = source.generate_photons(1.)
	assert np.all(photons['dir'][:, 1] <= 0)

def test_photon_direction_cone():
	"This tests the LabPointSourceCone. Checks the optional 'direction' parameter. It also checks that the fourth element of every direction vector is zero"
	pos = [1., 1., 1.]
	rate = 10
	direction = [1.,1.,1.]
	source = LabPointSourceCone(pos, delta = 1, flux=rate, energy=5., direction = direction)

	photons = source.generate_photons(1.)
	assert np.all(photons['dir'][:, 3] <= 0)
