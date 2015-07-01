import numpy as np
from ..labSource import LabConstantPointSource as LabSource

def test_photon_generation():
	'''This tests the lab point source. It checks that the starting points are all
	at the sources position.
	'''
	pos = [1., 1., 1.] 
	polar = [0., 0., 0.] 
	rate = 10
	source = LabSource(pos, polar, rate, 5., 1.) 
				    
	photons = source.generate_photons(1.)
	assert np.all(photons['pos'] == np.ones([10, 4]))
