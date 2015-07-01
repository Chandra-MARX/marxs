import numpy as np
from astropy.table import Table, Column

from .source import Source


class FarLabConstantPointSource(Source):
	'''Simple in-lab source used with aperture
	
	- assumes point source is far from a rectangular aperture, and only the photons that pass through are tracked
	- photon start positions uniformly distributed within rectangular aperture (reasonable approximation if source is far)
	- photon direction determined by location of source, and selected photon starting position
	- aperture is parallel to y-z plane
	- TODO: figure out how to provide energy distribution

	Parameters
	----------------
	position: 3 element list
		3D coordinates of photon source (not aperture)
	polarization: UNKNOWN
		TODO: determine representation of polarization (needs magnitude and orientation, anything else?)
	rate: float
		photons generated per second
	energy: UNKNOWN
		TODO: determine representation of energy distribution
	center: 3 element list
		3D coordinates of aperture center
	y: float
		half of the width of the aperture in the y direction
	z: float
		half of the width of the aperture in the z direction
	'''
	def __init__(self, position, polarization, rate, energy, center, y, z):
		self.pos = position
		self.polar = polarization
		self.rate = rate
		self.apertureCenter = center
		self.y = y
		self.z = z

	def generate_photons(self, t):
		n = (int)(t * self.rate)

		# randomly choose direction - photons uniformly distributed over baffle plate area
		# coordinate axes: origin at baffle plate, tube center. +x source, +y window, +z up
		# measurements in mm
		y = self.y
		z = self.z
		center = self.apertureCenter
		pos = np.array([center[0] * np.ones(n),
						center[1] + np.random.uniform(-y, y, n),
						center[2] + np.random.uniform(-z, z, n),
						np.ones(n)])
		dir = np.array([pos[0, :] - self.pos[0],
						pos[1, :] - self.pos[1],
						pos[2, :] - self.pos[2],
						np.zeros(n)])
		
		return Table({'pos': pos.T, 'dir': dir.T, 'energy': np.ones(n).T, 'polarization': np.random.uniform(0, 2 * np.pi, n).T, 'probability': np.ones(n).T})


class LabConstantPointSource(Source):
	'''Simple in-lab source for testing purposes

	- point source
	- photons uniformly distributed in all directions
	- photon start position is source position
	- TODO: figure out how to provide energy distribution
	
	Parameters
	----------------
	position: 3 element list
		3D coordinates of photon source
	polarization: UNKNOWN
		TODO: determine representation of polarization (needs magnitude and orientation, anything else?)
	rate: float
		photons generated per second
	energy: UNKNOWN
		TODO: determine representation of energy distribution
	'''
	def __init__(self, position, polarization, rate, energy):
		self.pos = position
		self.polar = polarization
		self.rate = rate
	
	def generate_photons(self, t):
		n = (int)(t * self.rate)

		# assign position to photons
		pos = np.array([self.pos[0] * np.ones(n),
						self.pos[1] * np.ones(n),
						self.pos[2] * np.ones(n),
						np.ones(n)])
		
		# randomly choose direction - photons go in all directions from source
		theta = np.random.uniform(0, 2 * np.pi, n);
		phi = np.arcsin(np.random.uniform(-1, 1, n))
		dir = np.array([np.cos(theta) * np.cos(phi),
						np.sin(theta) * np.cos(phi),
						np.sin(phi),
						np.zeros(n)])

		return Table({'pos': pos.T, 'dir': dir.T, 'energy': np.ones(n).T, 'polarization': np.random.uniform(0, 2 * np.pi, n).T, 'probability': np.ones(n).T})
