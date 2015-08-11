import numpy as np

from .base import FlatOpticalElement


class Baffle(FlatOpticalElement):
	'''Plate with rectangular hole that allows photons through.

	The probability of photons that miss is set to 0.
	'''
	def process_photons(self, photons):
		intersect, h_intersect, det_coords = self.intersect(photons['dir'].data, photons['pos'].data)
		photons['pos'][intersect] = h_intersect[intersect]
		photons['probability'][~intersect] = 0
		return photons
