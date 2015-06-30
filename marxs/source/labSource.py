import numpy as np
from astropy.table import Table, Column

from .source import Source


class FarLabConstantPointSource(Source):
	'''Simple in-lab source for testing purposes

	- point source
	- TODO: figure out how to provide energy distribution
	- photons uniformly distributed through rectangular space (reasonable approximation if source is far)
	- photon direction determined by location of source, and selected photon starting position
	'''
	def __init__(self, position, polarization, rate, energyMean, energyStdDev):
		self.pos = position
		self.polar = polarization
		self.rate = rate   # photons/sec
		self.meanEnergy = energyMean
		self.stdDevEnergy = energyStdDev

	def generate_photons(self, t, center, v_y, v_z):
		n = (int)(t * self.rate)
		#out = np.empty((6, n))
		#out[0, :] = np.arange(0, t, 1. / self.rate)
		# out[1, :] = self.pos

		# randomly choose direction - photons uniformly distributed over baffle plate area
		# coordinate axes: origin at baffle plate, tube center. +x source, +y window, +z up
		# measurements in mm
		y = np.linalg.norm(v_y)
		z = np.linalg.norm(v_z)
		pos = np.empty((4, n))
		pos = np.array([center[0] * np.ones(n),
						center[1] + np.random.uniform(-y, y, n),
						center[2] + np.random.uniform(-z, z, n),
						np.ones(n)])
		dir = np.empty((4, n))
		dir = np.array([pos[0, :] - self.pos[0],
						pos[1, :] - self.pos[1],
						pos[2, :] - self.pos[2],
						np.zeros(n)])
		#for i in range (0, n):
			#out[1, i] = np.array([center[0], center[1] + np.random.uniform(-y, y), center[2] + np.random.uniform(-z, z), 1])
			#out[2, i] = [out[1, i][0] - self.pos[0], out[1, i][1] - self.pos[1], out[1, i][2] - self.pos[2], 0]
		#out[1, :] = (center[0], center[1] + np.random.uniform(-y, y, n), center[2] + np.random.uniform(-z, z, n), 1)
		#out[2, :] = (out[1, :][0] - self.pos[0], out[1, :][1] - self.pos[1], out[1, :][2] - self.pos[2], 0)
		
		# randomly choose direction - photons go in all directions from source
		# theta = np.random.uniform(0, 2 * np.pi, n);
		# phi = np.arcsin(np.random.uniform(-1, 1, n))
		#out[2, :] = (theta, phi)
		
		# randomly choose photon energy (in keV)
		#out[3, :] = np.random.uniform(0.1, 100, n) # TO BE REPLACED

		# randomly choose polarization
		#out[4, :] = np.random.uniform(0, 2 * np.pi, n)

		#out[5, :] = 1
		
		return Table({'pos': pos.T, 'dir': dir.T, 'energy': np.ones(n).T, 'polarization': np.random.uniform(0, 2 * np.pi, n).T, 'probability': np.ones(n).T})
		#Table(out.T, names = ('time', 'pos', 'dir', 'energy', 'polarization', 'probability'))


class LabConstantPointSource(Source):
	'''Simple in-lab source for testing purposes

	- point source
	- TODO: figure out how to provide energy distribution
	- photons uniformly distributed in all directions
	- photon start position is source position
	'''
	def __init__(self, position, polarization, rate, energyMean, energyStdDev):
		self.pos = position
		#self.polar = polarization
		self.rate = rate   # photons/sec
		#self.meanEnergy = energyMean
		#self.stdDevEnergy = energyStdDev
	
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
