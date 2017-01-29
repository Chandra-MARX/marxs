# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Column
import transforms3d

from ..optics.base import FlatOpticalElement
from .source import Source
from ..math.polarization import polarization_vectors


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
    def __init__(selv, sourcePos, **kwargs):
        selv.sourcePos = sourcePos
        super(FarLabPointSource, selv).__init__(**kwargs)

    def generate_photons(selv, exposuretime):
        photons = super(FarLabPointSource, selv).generate_photons(exposuretime)
        n = len(photons)
        # randomly choose direction - photons uniformly distributed over aperture area
        # measurements in mm
        pos = np.dot(selv.pos4d, np.array([np.zeros(n),
                                           np.random.uniform(-1, 1, n),
                                           np.random.uniform(-1, 1, n),
                                           np.ones(n)]))

        dir = np.array([pos[0, :] - selv.sourcePos[0],
                        pos[1, :] - selv.sourcePos[1],
                        pos[2, :] - selv.sourcePos[2],
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
    def __init__(selv, position, direction=None, **kwargs):
        selv.dir = direction
        selv.position = position
        super(LabPointSource, selv).__init__(**kwargs)

    def generate_photons(selv, exposuretime):
        photons = super(LabPointSource, selv).generate_photons(exposuretime)
        n = len(photons)

        # assign position to photons
        pos = np.array([selv.position[0] * np.ones(n),
                        selv.position[1] * np.ones(n),
                        selv.position[2] * np.ones(n),
                        np.ones(n)])

        # randomly choose direction - photons go in all directions from source
        theta = np.random.uniform(0, 2 * np.pi, n);
        phi = np.arcsin(np.random.uniform(-1, 1, n))
        dir = np.array([np.cos(theta) * np.cos(phi),
                        np.sin(theta) * np.cos(phi),
                        np.sin(phi),
                        np.zeros(n)])

        if (selv.dir != None):
        	if (selv.dir[1] == 'x'):
        		col = 0
        	if (selv.dir[1] == 'y'):
        		col = 1
        	if (selv.dir[1] == 'z'):
        		col = 2
        	dir[col] = abs(dir[col])
        	if (selv.dir[0] == '-'):
        		dir[col] *= -1

        photons.add_column(Column(name='pos', data=pos.T))
        photons.add_column(Column(name='dir', data=dir.T))
        photons.add_column(Column(name='polarization', data=polarization_vectors(dir.T, photons['polangle'])))
        return photons

class LabPointSourceCone(Source):
    '''In-lab point source

    - Photons are uniformly distributed in all directions in cone (from the point). Cone is meant to refer to the volume swept out by a solid angle in a unit sphere centered at the point source.
    - Photon start position is source position (tip of cone)

    Parameters
    ----------
    direction: 3 element list
        Direction is given by a vector [x,y,z]
        This is the direction of the center of the cone (axis about which cone size is measured in steradians).
        It is sufficient to enter any vector that spans this axis (magnitude does not matter).
    position: 3 element list
        3D coordinates of photon source
    delta: float
        This is half the openning angle of the cone. It is given in steradians.
    kwargs : see `Source`
        Other keyword arguments include ``flux``, ``energy`` and ``polarization``.
        See `Source` for details.
    '''
    def __init__(selv, position, delta, direction=None,  **kwargs):

        selv.dir = direction / np.sqrt(np.dot(direction, direction)) #normalize direction
        selv.position = position
        selv.deltaphi = delta
        super(LabPointSourceCone, selv).__init__(**kwargs)

    def generate_photons(selv, exposuretime):
        photons = super(LabPointSourceCone, selv).generate_photons(exposuretime)
        n = len(photons)

        # assign position to photons
        pos = np.array([selv.position[0] * np.ones(n),
                        selv.position[1] * np.ones(n),
                        selv.position[2] * np.ones(n),
                        np.ones(n)])

        # Randomly choose direction - photons directions randomly distributed inside cone.
        # Visualize in arbitrary x-y-z coordinate system (to be reconciled with class parameters later)
        # Angle from pole (z-axis) = phi. Angle from x-axis on x-y plane is theta
        # This cone is temporarily centered about the z axis.
        theta = np.random.uniform(0, 2 * np.pi, n);
        fractionalArea = 2 * np.pi * (1 - np.cos(selv.deltaphi)) / (4 * np.pi) #this is the fractional surface area swept out by delta
        v = np.random.uniform(0, fractionalArea, n)
        phi = np.arccos(1 - 2 * v)
        # For computation of phi see http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php

        dir = np.array([np.cos(theta) * np.sin(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(phi),
                        np.zeros(n)])

        # Now we have all directions for n photons in dir.
        # Now we rotate dir to align with the given direction: selv.dir
        # To find axis of rotation: cross selv.dir with z
        axis = np.cross(selv.dir, [0, 0, 1])

        angle = np.arccos(selv.dir[2]) # Simplified

        rotationMatrix = transforms3d.axangles.axangle2aff(axis, -angle)

        # The aligned directions are:
        dir = np.dot(rotationMatrix, dir)


        photons.add_column(Column(name='pos', data=pos.T))
        photons.add_column(Column(name='dir', data=dir.T))
        photons.add_column(Column(name='polarization', data=polarization_vectors(dir.T, photons['polangle'])))
        return photons
