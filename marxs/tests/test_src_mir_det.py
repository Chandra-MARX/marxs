# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from marxs.source.labSource import LabPointSource
from marxs.optics.multiLayerMirror import MultiLayerMirror
from marxs.optics.detector import FlatDetector

def test_src_mir_det():
	'''This tests that source, multilayer mirror, and detector run without producing any errors.
	'''
	string1 = 'marxs/optics/data/A12113.txt'
	string2 = 'marxs/optics/data/ALSpolarization2.txt'
	a = 2**(-0.5)
	rotation1 = np.array([[a, 0, -a],
						  [0, 1, 0],
						  [a, 0, a]])
	rotation2 = np.array([[0, 0, 1],
						  [0, 1, 0],
						  [-1, 0, 0]])

	source = LabPointSource([10., 0., 0.], flux=100., energy=-1.)
	mirror = MultiLayerMirror(string1, string2, position=np.array([0., 0., 0.]), orientation=rotation1)
	detector = FlatDetector(1., position=np.array([0., 0., 10.]), orientation=rotation2, zoom = np.array([1, 100, 100]))

	photons = source.generate_photons(100)
	photons = mirror(photons)
	photons = detector(photons)
