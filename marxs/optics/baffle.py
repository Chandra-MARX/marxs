import numpy as np

from .base import FlatOpticalElement


class Baffle(FlatOpticalElement):
	'''Plate with rectangular hole that allows photons through

	Photons that miss are removed from the photon table.
	'''
	def process_photons(self, photons):
		intersect, h_intersect, det_coords = self.intersect(photons['dir'], photons['pos'])
		photons['pos'][intersect] = h_intersect[intersect]
		return photons[intersect]
