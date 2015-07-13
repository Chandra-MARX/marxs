import numpy as np
from astropy.table import Table, Column

from .source import Source
from ..optics.base import FlatOpticalElement
from ..optics.polarization import polarization_vectors


class PolarizedSource(Source):
    def __init__(self, **kwargs):
        super(PolarizedSource, self).__init__(**kwargs)
    
    def random_polarization(self, n, dir_array):
        # randomly choose polarization (temporary)
        angles = np.random.uniform(0, 2 * np.pi, n)
        return polarization_vectors(dir_array, angles)


class FarLabConstantPointSource(PolarizedSource, FlatOpticalElement):
    '''Simple in-lab source used with aperture
    
    - assumes point source is far from a rectangular aperture, and only the photons that pass through are tracked
    - photon start positions uniformly distributed within rectangular aperture (reasonable approximation if source is far)
    - photon direction determined by location of source, and selected photon starting position
    - aperture is parallel to y-z plane
    - polarization angle is assumed relative to positive y axis unless direction is exactly along y axis,
      in which case polarization is relative to positive x axis, ALL GLOBAL AXES
    - TODO: figure out how to provide energy distribution

    Parameters
    ----------------
    sourcePos: 3 element list
        3D coordinates of photon source (not aperture)
    polarization: UNKNOWN
        TODO: determine representation of polarization (needs magnitude and orientation, anything else?)
    rate: float
        photons generated per second
    energy: UNKNOWN
        TODO: determine representation of energy distribution
    **kwargs: 'position', 'orientation', 'zoom'
        4x4 pos4d matrix for transformations in homogeneous coords from local coords to global coords 
    '''
    def __init__(self, sourcePos, polarization, rate, energy, **kwargs):
        self.sourcePos = sourcePos
        self.polar = polarization
        self.rate = rate
        super(FarLabConstantPointSource, self).__init__(**kwargs)

    def generate_photons(self, t):
        n = int(t * self.rate)

        # randomly choose direction - photons uniformly distributed over baffle plate area
        # coordinate axes: origin at baffle plate, tube center. +x source, +y window, +z up
        # measurements in mm
        pos = np.dot(self.pos4d, np.array([np.zeros(n),
                                           np.random.uniform(-1, 1, n),
                                           np.random.uniform(-1, 1, n),
                                           np.ones(n)]))

        dir = np.array([pos[0, :] - self.sourcePos[0],
                        pos[1, :] - self.sourcePos[1],
                        pos[2, :] - self.sourcePos[2],
                        np.zeros(n)])
                        
        polarization = self.random_polarization(n, dir.T)
        
        return Table({'pos': pos.T, 'dir': dir.T, 'energy': np.ones(n).T, 'polarization': polarization, 'probability': np.ones(n).T})


class LabConstantPointSource(PolarizedSource):
    '''Simple in-lab source for testing purposes

    - point source
    - photons uniformly distributed in all directions
    - photon start position is source position
    - polarization angle is assumed relative to positive y axis unless direction is exactly along y axis,
      in which case polarization is relative to positive x axis, ALL GLOBAL AXES
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
                        
        polarization = self.random_polarization(n, dir.T)
		# provided polarization table must contain only UNIT VECTORS
        return Table({'pos': pos.T, 'dir': dir.T, 'energy': np.ones(n).T, 'polarization': polarization, 'probability': np.ones(n).T})
