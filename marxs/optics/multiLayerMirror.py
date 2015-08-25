import numpy as np
from astropy.io import ascii

from .base import FlatOpticalElement
from ..math.pluecker import *


class MultiLayerMirror(FlatOpticalElement):
    '''Multilayer mirror with varying layer thickness along one axis

    The distance between layers (and thus best reflected energy) changes along
    the local y axis.
    All reflectivity data is assumed to be for a single, desired angle. There
    is currently no way to enter varying reflection that depends on the angle
    of incidence.
    There is a default size of 49mm by 24mm, but this can be overridden by
    entering a different value for zoom.

    Provide reflectivity data in a file with columns:

    - 'X(mm)' - position along the "changing" axis (local y axis)
    - 'Peak lambda' - wavelength with maximum reflection at a given position
    - 'Peak' - maximum reflection at a given position
    - 'FWHM(nm)' - full width half max, measure of width of reflection Gaussian peaks
    
    Provide polarization data in a file with columns:
    
    - 'Photon energy' - energy of the photon in keV
    - 'Polarization' - Fraction polarized in the more reflective direction, so that
    		randomly polarized light would have a value of 0.5.
    
    Parameters
    ----------
    reflFile: string
    	path, filename, and .txt extension for reflection data file
    testedPolarization: string
    	path, filename, and .txt to a text file containing a table with photon energy
    	and fraction polarization for the light used to test the mirrors and create the
    	reflectivity file
    '''
    def __init__(self, reflFile, testedPolarization, **kwargs):
        self.fileName = reflFile
        self.polFile = testedPolarization
        if ('zoom' not in kwargs) and ('pos4d' not in kwargs):
        	kwargs['zoom'] = np.array([1, 24.5, 12])   # in mm
        super(MultiLayerMirror, self).__init__(**kwargs)

    def process_photons(self, photons):
    	# NOTE: This function could be split up into smaller, more readable functions, but I
    	# have done my best to clearly label the specific calculations in the comments.
    	
        # for photons that do not reach the mirror, set probability to zero
        doesIntersect, intersection, local = self.intersect(photons['dir'], photons['pos'])
        photons['probability'] *= doesIntersect
        
        # save the direction of the incoming photons as beam_dir
        beam_dir = (photons['dir'][:,0:3]).copy()
    	beam_dir /= np.linalg.norm(beam_dir, axis=1)[:, np.newaxis]
        
        # reflect the photons (change direction) by transforming to local coordinates
        directions = photons['dir']
        pos4d_inv = np.linalg.inv(self.pos4d)
        directions = directions.T
        directions = np.dot(pos4d_inv, directions)
        directions[0,:] *= -1
        directions = np.dot(self.pos4d, directions)
        photons['dir'] = directions.T
        
        # split polarization into s and p components
        # v_1 is s polarization (perpendicular to plane of incidence), v_2 is p polarization (in the plane of incidence)
        v_1 = np.cross(beam_dir, self.geometry['e_x'][0:3])
        v_1 /= np.linalg.norm(v_1, axis=1)[:, np.newaxis]
        v_2 = np.cross(beam_dir, v_1)
        # find polarization component in each direction, v1 and v2
        polarization = (photons['polarization'][:,0:3]).copy()
        # find the dot product of v1 and v2 for each photon with each photon's polarization vector
        p_v_1 = np.einsum('ij,ij->i', polarization, v_1)   # the cosines between the pairs of vectors (these are unit vectors)
        p_v_2 = np.einsum('ij,ij->i', polarization, v_2)
        
        # adjust polarization vectors after reflection
        # find new v_2
        new_beam_dir = (photons['dir'][:,0:3]).copy()
        new_beam_dir /= np.linalg.norm(new_beam_dir, axis=1)[:, np.newaxis]
        new_v_2 = np.cross(new_beam_dir, v_1)
        photons['polarization'][:,0:3] = -p_v_1[:, np.newaxis] * v_1 + p_v_2[:, np.newaxis] * new_v_2
        
        # set position to intersection point
        photons['pos'] = intersection

        # read in correct reflecting probability file, now in table format
    	reflectFile = ascii.read(self.fileName)
        
        # find reflectivity adjustment due to polarization of light in reflectivity testing
        polarizedFile = ascii.read(self.polFile)
        tested_polarized_fraction = np.interp(photons['energy'], polarizedFile['Photon energy'] / 1000, polarizedFile['Polarization'])

        # find probability of being reflected due to position
        # put the photon positions and the position values from the reflection file into local coordinates
        local_intersection = h2e((np.dot(pos4d_inv, intersection.T)).T)
        local_coords_in_file = reflectFile['X(mm)'] / np.linalg.norm(self.geometry['v_y']) - 1
        # interpolate 'Peak lambda', 'Peak' [reflectivity], and 'FWHM(nm)' to the actual photon positions
        peak_wavelength = np.interp(local_intersection[:,1], local_coords_in_file, reflectFile['Peak lambda'])
        max_refl = np.interp(local_intersection[:,1], local_coords_in_file, reflectFile['Peak']) / tested_polarized_fraction
        spread_refl = np.interp(local_intersection[:,1], local_coords_in_file, reflectFile['FWHM(nm)'])

        wavelength = 1.23984282 / photons['energy']   # wavelength is in nm assuming energy is in keV
        c_squared = (spread_refl ** 2) / (8. * np.log(2))   # the standard deviation squared of the Gaussian reflectivity functions of each photon's wavelength
        c_is_zero = (c_squared == 0)   # skip the case when there is no Gaussian (this is assumed to just be the zero function)
    	
        refl_prob = np.zeros(len(wavelength))
        refl_prob[~c_is_zero] = max_refl[~c_is_zero] * np.exp(-((wavelength[~c_is_zero] - peak_wavelength[~c_is_zero]) ** 2) / (2 * c_squared[~c_is_zero]))

        # find probability of being reflected due to polarization
        # v_1 is s polarization (perpendicular to plane of incidence), the better reflecting polarization
        refl_prob[~c_is_zero] *= np.einsum('ij,ij->i', polarization[~c_is_zero], v_1[~c_is_zero]) ** 2   # dot products of unit vectors
        refl_prob[np.isnan(refl_prob)] = 0
        # multiply probability by probability of reflection
        photons['probability'] *= refl_prob / 100

        return photons
