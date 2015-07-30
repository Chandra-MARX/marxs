import numpy as np
from astropy.table import Table, Column

from ..optics.base import FlatOpticalElement
from .source import Source
from ..optics.polarization import polarization_vectors


class FarLabConstantPointSource(Source, FlatOpticalElement):
    '''Simple in-lab source used with aperture

    - assumes point source is far from a rectangular aperture, and only the photons that pass through are tracked
    - photon start positions uniformly distributed within rectangular aperture (reasonable approximation if source is far)
    - photon direction determined by location of source, and selected photon starting position
    - aperture is parallel to y-z plane
    - polarization angle is assumed relative to positive y axis unless direction is exactly along y axis,
      in which case polarization is relative to positive x axis, ALL GLOBAL AXES
    - TODO: figure out how to provide energy distribution

    Parameters
    ----------
    flux, energy, polarization: see `Source`
    sourcePos: 3 element list
        3D coordinates of photon source (not aperture)
    **kwargs: 'position', 'orientation', 'zoom'
        4x4 pos4d matrix for transformations in homogeneous coords from local coords to global coords
    '''
    def __init__(self, sourcepos, **kwargs):
        self.sourcePos = sourcepos
        super(FarLabConstantPointSource, self).__init__(**kwargs)

    def generate_photons(self, exposuretime):
        photons = super(FarLabConstantPointSource, self).generate_photons(exposuretime)
        n = len(photons)
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
        photons['pos'] = pos.T
        photons['dir'] = dir.T
        photons['polarization'] = polarization_vectors(dir.T, photons['polangle'])
        return photons


class LabConstantPointSource(Source):
    '''Simple in-lab source for testing purposes

    - point source
    - photons uniformly distributed in all directions
    - photon start position is source position
    - polarization angle is assumed relative to positive y axis unless direction is exactly along y axis,
      in which case polarization is relative to positive x axis, ALL GLOBAL AXES
    - TODO: figure out how to provide energy distribution
    - TODO: improve direction capability to allow for vetor and angle around that vector

    Parameters
    ----------
    position: 3 element list
        3D coordinates of photon source
    polarization: UNKNOWN
        TODO: determine representation of polarization (needs magnitude and orientation, anything else?)
    rate: float
        photons generated per second
    energy: UNKNOWN
        TODO: determine representation of energy distribution
    direction: string
        hemisphere of photons, format is + or - followed by x, y, or z. Ex: '+x' or '-z'
    '''
    def __init__(self, position, direction=None, **kwargs):
        self.dir = direction
        self.position = position
        super(LabConstantPointSource, self).__init__(**kwargs)

    def generate_photons(self, exposuretime):
        photons = super(LabConstantPointSource, self).generate_photons(exposuretime)
        n = len(photons)

        # assign position to photons
        pos = np.array([self.position[0] * np.ones(n),
                        self.position[1] * np.ones(n),
                        self.position[2] * np.ones(n),
                        np.ones(n)])

        # randomly choose direction - photons go in all directions from source
        theta = np.random.uniform(0, 2 * np.pi, n);
        phi = np.arcsin(np.random.uniform(-1, 1, n))
        dir = np.array([np.cos(theta) * np.cos(phi),
                        np.sin(theta) * np.cos(phi),
                        np.sin(phi),
                        np.zeros(n)])

        if (self.dir != None):
        	if (self.dir[1] == 'x'):
        		col = 0
        	if (self.dir[1] == 'y'):
        		col = 1
        	if (self.dir[1] == 'z'):
        		col = 2
        	dir[col] = abs(dir[col])
        	if (self.dir[0] == '-'):
        		dir[col] *= -1

        photons.add_column(Column(name='pos', data=pos.T))
        photons.add_column(Column(name='dir', data=dir.T))
        photons.add_column(Column(name='polarization', data=polarization_vectors(dir.T, photons['polangle'])))
        return photons
