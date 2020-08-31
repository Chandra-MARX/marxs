# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Column
import astropy.units as u
import transforms3d

from ..optics.base import FlatOpticalElement
from .source import Source
from ..math.polarization import polarization_vectors
from ..math.utils import h2e, e2h


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
        if not 'flux' in kwargs:
            kwargs['flux'] = 1 / u.s
        super(FarLabPointSource, self).__init__(geomarea=None, **kwargs)

    @u.quantity_input
    def generate_photons(self, exposuretime: u.s):
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


class LabPointSourceCone(Source):
    '''In-lab point source

    - Photons are uniformly distributed in all directions in cone (from the point). Cone is meant to refer to the volume swept out by a solid angle in a unit sphere centered at the point source.
    - Photon start position is source position (tip of cone)

    Parameters
    ----------
    position: array-like of shape(3,)
        Eukledian coordinates of photon source. Default is at the origin
        of the coordinate system.
    half_opening: float
        This is half the openning angle of the cone. It is given in steradians.
        The default is pi, which distributes the photons evenly over the entire
        sphere.
    direction: array-like of shape (3,)
        This is the direction of the center of the cone.
        Default is the positive x-axis.
    kwargs : see `Source`
        Other keyword arguments include ``flux``, ``energy`` and ``polarization``.
        See `Source` for details.
    '''
    def __init__(self, position=[0, 0, 0], half_opening=np.pi,
                 direction=[1., 0., 0.],
                 **kwargs):
        if (len(position) != 3) or (len(direction) != 3):
            raise ValueError('Direction and position are expected in Eukledian coordinates.')
        self.dir = e2h(np.asanyarray(direction) / np.linalg.norm(direction), 0)
        self.position = e2h(np.asanyarray(position), 1)
        self.half_opening = half_opening
        if not 'flux' in kwargs:
            kwargs['flux'] = 1 / u.s
        super(LabPointSourceCone, self).__init__(geomarea=None, **kwargs)

    @u.quantity_input
    def generate_photons(self, exposuretime: u.s):
        photons = super(LabPointSourceCone, self).generate_photons(exposuretime)
        n = len(photons)

        # assign position to photons
        pos = np.ones((4, n)) * self.position[:, None]

        # Randomly choose direction - photons directions randomly distributed inside cone.
        # Visualize in arbitrary x-y-z coordinate system (to be reconciled with class parameters later)
        # Angle from pole (z-axis) = phi. Angle from x-axis on x-y plane is theta
        # This cone is temporarily centered about the z axis.
        theta = np.random.uniform(0, 2 * np.pi, n)
        #this is the fractional surface area swept out by delta
        fractionalArea = 2 * np.pi * (1 - np.cos(self.half_opening)) / (4 * np.pi)
        v = np.random.uniform(0, fractionalArea, n)
        phi = np.arccos(1 - 2 * v)
        # For computation of phi see http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php

        dir = np.array([np.cos(theta) * np.sin(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(phi),
                        np.zeros(n)])

        # Now we have all directions for n photons in dir.
        # Now we rotate dir to align with the given direction: self.dir
        # To find axis of rotation: cross self.dir with z
        axis = np.cross(h2e(self.dir), [0, 0, 1])

        angle = np.arccos(self.dir[2]) # Simplified

        rotationMatrix = transforms3d.axangles.axangle2aff(axis, -angle)

        # The aligned directions are:
        dir = np.dot(rotationMatrix, dir)


        photons.add_column(Column(name='pos', data=pos.T))
        photons.add_column(Column(name='dir', data=dir.T))
        photons.add_column(Column(name='polarization', data=polarization_vectors(dir.T, photons['polangle'])))
        return photons
