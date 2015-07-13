import numpy as np

from ..math.pluecker import *


def polarization_vectors(dir_array, angles):
	n = len(angles)
	polarization = np.zeros((n, 4))
	x = np.array([1., 0., 0.])
	y = np.array([0., 1., 0.])
	for i in range(0, n):
		r = h2e(dir_array[i])
		r /= np.linalg.norm(r)
		if not (np.isclose(r[0], 0.) and np.isclose(r[2], 0.)):
			# polarization relative to positive y at 0              
			v_1 = y - (r * np.dot(r, y))
			v_1 /= np.linalg.norm(v_1)
		else:
			# polarization relative to positive x at 0
			v_1 = x - (r * np.dot(r, x))
			v_1 /= np.linalg.norm(v_1)
			
		# right hand coordinate system is v_1, v_2, r (photon direction)
		v_2 = np.cross(r, v_1)
		polarization[i, 0:3] = v_1 * np.cos(angles[i]) + v_2 * np.sin(angles[i])
		polarization[i, 3] = 0
	
	return polarization