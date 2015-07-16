import numpy as np

from ..math.pluecker import *


def polarization_vectors(dir_array, angles):
	'''Takes angle polarizations and converts them to vectors in the direction of polarization.
	
	- Follows convention: Vector perpendicular to photon direction and closest to +y axis is
	angle 0 for polarization direction, unless photon direction is parallel to the y axis, in
	which case the vector closest to the +x axis is angle 0.
	
	Parameters
    ----------
    dir_array : nx4 np.array
        each row is the homogeneous coordinates for a photon's direction vector
    angles : np.array
    	1D array with the polarization angles
	'''
	n = len(angles)
	polarization = np.zeros((n, 4))
	x = np.array([1., 0., 0.])
	y = np.array([0., 1., 0.])
	
#	NOTE: The commented code works and is more readable, but the current code is faster.
#	for i in range(0, n):
#		r = h2e(dir_array[i])
#		r /= np.linalg.norm(r)
#		if not (np.isclose(r[0], 0.) and np.isclose(r[2], 0.)):
#			# polarization relative to positive y at 0              
#			v_1 = y - (r * np.dot(r, y))
#			v_1 /= np.linalg.norm(v_1)
#		else:
#			# polarization relative to positive x at 0
#			v_1 = x - (r * np.dot(r, x))
#			v_1 /= np.linalg.norm(v_1)
#			
#		# right hand coordinate system is v_1, v_2, r (photon direction)
#		v_2 = np.cross(r, v_1)
#		polarization[i, 0:3] = v_1 * np.cos(angles[i]) + v_2 * np.sin(angles[i])
#		polarization[i, 3] = 0	
	
	r = dir_array.copy()[:,0:3]
	r /= np.linalg.norm(r, axis=1)[:, np.newaxis]
	pol_convention_x = np.isclose(r[:,0], 0.) & np.isclose(r[:,2], 0.)
	# polarization relative to positive y or x at 0
	v_1 = ~pol_convention_x[:, np.newaxis] * (y - r * np.dot(r, y)[:, np.newaxis])
	v_1 += pol_convention_x[:, np.newaxis] * (x - r * np.dot(r, x)[:, np.newaxis])
	v_1 /= np.linalg.norm(v_1, axis=1)[:, np.newaxis]

	# right hand coordinate system is v_1, v_2, r (photon direction)
	v_2 = np.cross(r, v_1)
	polarization[:, 0:3] = v_1 * np.cos(angles)[:, np.newaxis] + v_2 * np.sin(angles)[:, np.newaxis]
	
	return polarization
	
	