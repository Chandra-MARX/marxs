import numpy as np
from astropy.io import ascii

from .base import FlatOpticalElement
from ..math.pluecker import *


class MultiLayerMirror(FlatOpticalElement):
    '''Multilayer mirror with varying layer thickness along one axis
    
    The distance between layers changes along the local y axis.
    All reflectivity data is assumed to be for a single, desired angle. There
    is currently no way to enter varying reflection that depends on the angle
    of incidence.
    There is a default size of 49mm by 24mm, but this can be overridden by
    entering a different value for zoom.
    Provide reflectivity data in a file with columns:
    	'X(mm)' - position along the "changing" axis
    	'Peak lambda' - wavelength with maximum reflection at a given position
    	'Peak' - maximum reflection at a given position
    	'FWHM(nm)' - full width half max, measure of width of reflection Gaussian peaks
    
    Parameters
    ----------
    fileCode: string
    	serial number of mirror, used to open reflection data file
    '''
    def __init__(self, fileCode, **kwargs):
        self.fileName = './marxs/optics/data/' + fileCode + '.txt'
        if ('zoom' not in kwargs):
        	kwargs['zoom'] = np.array([1, 24.5, 12])   # in mm
        super(MultiLayerMirror, self).__init__(**kwargs)

    def process_photons(self, photons):
        # eliminate photons that do not reach the mirror
        doesIntersect, intersection, local = self.intersect(photons['dir'], photons['pos'])
        photons['probability'] *= doesIntersect
        
        # split polarization into parallel and other components
        beam_dir = (photons['dir'][:,0:3]).copy()
    	beam_dir /= np.linalg.norm(beam_dir, axis=1)[:, np.newaxis]
        # v_1 is parallel to plane, v_2 is third direction remaining
        v_1 = np.cross(beam_dir, self.geometry['e_x'][0:3])
        v_2 = np.cross(beam_dir, v_1)
        # find polarization in each direction
        polarization = (photons['polarization'][:,0:3]).copy()
        v_1 /= np.linalg.norm(v_1, axis=1)[:, np.newaxis]
        v_2 /= np.linalg.norm(v_2, axis=1)[:, np.newaxis]
        p_v_1 = np.einsum('ij,ij->i', polarization, v_1)   # just the cosines between the pairs of vectors
        p_v_2 = np.einsum('ij,ij->i', polarization, v_2)
        
        # reflect (change direction)
        directions = photons['dir']
        pos4d_inv = np.linalg.inv(self.pos4d)
        directions = directions.T
        directions = np.dot(pos4d_inv, directions)
        directions[0,:] *= -1
        directions = np.dot(self.pos4d, directions)
        photons['dir'] = directions.T
        
        # adjust polarization vectors after reflection
        # find new v_2
        new_beam_dir = (photons['dir'][:,0:3]).copy()
        new_beam_dir /= np.linalg.norm(new_beam_dir, axis=1)[:, np.newaxis]
        new_v_2 = np.cross(new_beam_dir, v_1)
        photons['polarization'][:,0:3] = np.tile(p_v_1, (3,1)).T * v_1 + np.tile(p_v_2, (3,1)).T * new_v_2
        
        # set position to intersection point
        photons['pos'] = intersection
        
        # read in correct reflecting probability file, now in table format
    	reflectFile = ascii.read(self.fileName)
        
        # find reflectivity adjustment due to polarization in testing
        polarizedFile = ascii.read('./marxs/optics/data/ALSpolarization2.txt')
        tested_polarized_fraction = np.interp(photons['energy'], polarizedFile['Photon energy'] / 1000, polarizedFile['Polarization'])
        
        # find probability of being reflected due to position
        local_intersection = h2e((np.dot(pos4d_inv, intersection.T)).T)
        local_coords_in_file = reflectFile['X(mm)'] / np.linalg.norm(self.geometry['v_y']) - 1
        peak_wavelength = np.interp(local_intersection[:,1], local_coords_in_file, reflectFile['Peak lambda'])
        max_refl = np.interp(local_intersection[:,1], local_coords_in_file, reflectFile['Peak']) / tested_polarized_fraction
        spread_refl = np.interp(local_intersection[:,1], local_coords_in_file, reflectFile['FWHM(nm)'])
        
        wavelength = 1.23984282 / photons['energy']   # wavelength is in nm assuming energy is in keV
        c_squared = (spread_refl ** 2) / (2 * np.log(2))
        refl_prob = max_refl * np.exp(-((wavelength - peak_wavelength) ** 2) / (2 * c_squared))
        
        # find probability of being reflected due to polarization
        # v_1 is parallel to plane, good reflection direction
        refl_prob *= np.einsum('ij,ij->i', polarization, v_1)**2
        
        # multiply probability by probability of reflection
        for i in range(0, len(photons['probability'])):
        	if not (photons['probability'][i] == 0):
        		photons['probability'][i] *= refl_prob[i] / 100
        
        return photons
