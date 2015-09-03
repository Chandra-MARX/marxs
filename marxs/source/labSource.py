import numpy as np
from astropy.table import Table, Column

from ..optics.base import FlatOpticalElement
from .source import Source
from ..optics.polarization import polarization_vectors


class FarLabPointSource(Source, FlatOpticalElement):
    '''Simple in-lab point source used with aperture

    - assumes point source is far from a rectangular aperture, and only the photons that pass through are tracked
    - photon start positions uniformly distributed within rectangular aperture (reasonable approximation if source is far)
    - photon direction determined by location of source, and selected photon starting position
	- aperture is in its local y-z plane

    Parameters
    ----------
    sourcePos: 3 element list
        3D coordinates of photon source (not aperture)
    kwargs: ``pos4d`` or ``position``, ``orientation``, and ``zoom`` can be used to set the position,
        size and orientation of the rectangular apeture; see `pos4d` for details.
        Other keyword arguments include ``flux``, ``energy`` and ``polarization``.
        See `Source` for details.
    '''
    def __init__(self, sourcePos, **kwargs):
        self.sourcePos = sourcePos
        super(FarLabPointSource, self).__init__(**kwargs)

    def generate_photons(self, exposuretime):
        photons = super(FarLabPointSource, self).generate_photons(exposuretime)
        n = len(photons)
        # randomly choose direction - photons uniformly distributed over aperture area
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


class LabPointSource(Source):
    '''Simple in-lab point source

    - photons uniformly distributed in all directions
    - photon start position is source position
    - TODO: improve direction capability to allow for vector and angle around that vector

    Parameters
    ----------
    position: 3 element list
        3D coordinates of photon source
    direction: string
        Hemisphere of photons, format is + or - followed by x, y, or z. Ex: '+x' or '-z'.
        In many cases, it is sufficient to simulate photons for one hemisphere. In this case, the
        parameter ``direction`` can reduce the runtime by reducing the number of photons that are
        not relevant for the simulation.
    kwargs : see `Source`
        Other keyword arguments include ``flux``, ``energy`` and ``polarization``.
        See `Source` for details.
    '''
    def __init__(self, position, direction=None, **kwargs):
        self.dir = direction
        self.position = position
        super(LabPointSource, self).__init__(**kwargs)

    def generate_photons(self, exposuretime):
        photons = super(LabPointSource, self).generate_photons(exposuretime)
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
