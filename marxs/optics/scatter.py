# Licensed under GPL version 3 - see LICENSE.rst
'''Add additional scattering to processed photons.

Classes in this file add additional scattering in a statistical sense.

The classes in this module do not trace the photon to a specific location, they
just add the scatter at the point of the last interaction.  For example,
reflection from a (flat) mirror is implemented as a perfect reflection, but in
practice there is some roughness to the mirror that adds a small Gaussian blur
to the reflection. To represent this, the point of origin of the ray remains
unchanged, but a small random change is added to the direction vector.

'''
import numpy as np
from warnings import warn
import astropy.units as u

from ..math.utils import e2h, h2e, norm_vector
from ..math.rotations import axangle2mat
from ..math.polarization import parallel_transport
from .base import FlatOpticalElement

class RadialMirrorScatter(FlatOpticalElement):
    '''Add scatter to any sort of radial mirror.

    Scatter is added in the plane of reflection, which is defined here
    as the plane which contains (i) the current direction the ray and (ii) the
    vector connecting the center of the `RadialMirrorScatter` element and the
    point of last interaction for the ray.
    Scatter can also be added perpendicular to the plane-of-reflection.

    Parameters
    ----------
    inplanescatter : `astropy.Quantity`
        sigma of Gaussian for in-plane scatter
    perpplanescatter : `astropy.Quantity`
        sigma of Gaussian for scatter perpendicular to the plane of reflection
        (default = 0)
    '''
    def __init__(self, **kwargs):
        self.inplanescatter = kwargs.pop('inplanescatter').to(u.rad).value
        self.perpplanescatter = kwargs.pop('perpplanescatter', 0.).to(u.rad).value
        super(RadialMirrorScatter, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        n = intersect.sum()
        center = self.pos4d[:-1, -1]
        radial = h2e(photons['pos'][intersect].data) - center
        perpplane = np.cross(h2e(photons['dir'][intersect].data), radial)

        # np.random.normal does not work with scale=0
        # so special case that here.
        if self.inplanescatter != 0:
            inplaneangle = np.random.normal(loc=0., scale=self.inplanescatter, size=n)
            rot = axangle2mat(perpplane, inplaneangle)
            outdir = e2h(np.einsum('...ij,...i->...j', rot, h2e(photons['dir'][intersect])), 0)
        else:
            inplaneangle = np.zeros(n)
            outdir = photons['dir'][intersect]

        if self.perpplanescatter !=0:
            perpangle = np.random.normal(loc=0., scale=self.perpplanescatter, size=n)
            rot = axangle2mat(radial, perpangle)
            outdir = e2h(np.einsum('...ij,...i->...j', rot, h2e(outdir)), 0)
        else:
            perpangle = np.zeros_like(inplaneangle)

        pol = parallel_transport(photons['dir'].data[intersect, :], outdir,
                                 photons['polarization'].data[intersect, :])

        return {'dir': outdir, 'polarization': pol,
                'inplanescatter': inplaneangle, 'perpplanescatter': perpangle}


class RandomGaussianScatter(FlatOpticalElement):
    '''Add scatter to any sort of radial mirror.

    This element scatters rays by a small angle, drawn from a Gaussian
    distribution. The direction of the scatter is random.

    Parameters
    ----------
    scatter : `astropy.Quantitiy` or callable
        This this is a number, scattering angles will be drawn from a Gaussian
        with the given sigma. For a variable scatter, this can be a
        function with the following call signature: ``angle = func(photons,
        intersect, interpos, intercoos)``. The function should return an
        `astropy.Quantity` array, containing one angle for each intersecting
        photon. A function passed in for this parameter can makes the
        scattering time, location, or energy-dependent.
    '''
    scattername = 'scatter'

    def __init__(self, **kwargs):
        if 'scatter' in kwargs:
            if hasattr(self, 'scatter'):
                warn('Overriding class level "scatter" definition.')
            self.scatter = kwargs.pop('scatter') # in rad
        else:
            if not hasattr(self, 'scatter'):
                raise ValueError('Keyword "scatter" missing.')
        super().__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        n = intersect.sum()
        # np.random.normal does not work with scale=0
        # so special case that here.
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            scatterzero = self.scatter == 0
        if scatterzero:
            angle = np.zeros(n)
            out = {}
        else:
            pdir = norm_vector(h2e(photons['dir'][intersect].data))
            # Now, find a direction that is perpendicular to the photon direction
            # Any perpendicular direction will do
            # Start by making a set of vectors that at least are not parallel
            # to the photon direction
            guessvec = np.zeros_like(pdir)
            ind = np.abs(pdir[:, 0]) < 0.99999
            guessvec[ind, 0] = 1
            guessvec[~ind, 1] = 1
            perpvec = np.cross(pdir, guessvec)
            if callable(self.scatter):
                angle = self.scatter(photons, intersect,
                                     interpos, intercoos).to(u.rad).value
            else:
                angle = np.random.normal(loc=0.,
                                         scale=self.scatter.to(u.rad).value,
                                         size=n)
            rot = axangle2mat(perpvec, angle)
            outdir = np.einsum('...ij,...i->...j', rot, pdir)
            # Now rotate result by up to 2 pi to randomize direction
            angle2 = np.random.uniform(size=n) * 2 * np.pi
            rot = axangle2mat(pdir, angle2)
            outdir = e2h(np.einsum('...ij,...i->...j', rot, outdir), 0)
            pol = parallel_transport(photons['dir'].data[intersect, :], outdir,
                                     photons['polarization'].data[intersect, :])
            out = {'dir': outdir, 'polarization': pol}
        if self.scattername is not None:
            out[self.scattername] = angle
        return out
