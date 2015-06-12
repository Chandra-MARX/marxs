from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy import optimize

import transforms3d


class RowlandTorus(object):
    '''Torus with z axis as symmetry axis'''
    def __init__(self, R, r):
        self.R = R
        self.r = r

    def quartic(self, x, y, z):
        '''Quartic torus equation.

        Roots of this equation are points on the torus.
        '''
        return (x**2. + y**2. + z**2. + self.R**2. - self.r**2.)**2. - 4. * self.R**2. * (x**2. + y**2.)
    def normal(self, x, y, z):
        '''Return the gradient vector field'''
        if not np.allclose(self.quartic(x, y, z), 0.):
            raise ValueError('Gradient vector field is only defined for points on torus surface.')
        factor = 4. * (x**2. + y**2. + z**2. + self.R**2. - self.r**2)
        dFdx = factor * x - 8. * self.R**2 * x
        dFdy = factor * y - 8. * self.R**2 * y
        dFdz = factor * z
        return [dFdx, dFdy, dFdz]

class GratingArrayStructure(object):
    '''
    Parameters
    ----------
    radius : list of 2 floats float
        Inner and outer radius of the GAS as measured in the yz-plane from the
        origin.
    phi : list of 2 floats
        Bounding angles for a segment covered by the GSA. :math:`\phi=0`
        is on the positive y axis. The segment fills the space from ``phi1`` to
        ``phi2`` in the usual mathematical way (counterclockwise).
        Angles are given in radian. Note that ``phi[1] < phi[0]`` is possible if
        the segment crosses the y axis.
    d_facet : float
        Size of the edge of a facet, which is assumed to be flat and square.
        (``d_facet`` can be larger than the actual size of the silicon membrane to
        accommodate a minimum thickness of the surrounding frame.)
    z_min, z_max: TBD
    '''
    def __init__(self, d_facet, phi, radius, z_min, z_max):
        if not (radius[1] > radius[0]):
            raise ValueError('Outer radius must be larger than inner radius.')
        if np.min(radius) < 0:
            raise ValueError('Radius must be positive.')
        self.radius = radius

        if np.max(np.abs(phi)) > 10:
            raise ValueError('Input angles >> 2 pi. Did you use degrees (radian expected)?')
        self.phi = phi

        self.d_facet = d_facet

        self.z_min = z_min
        self.z_max = z_max

        self.place_facets()

    def max_facets_on_arc(self, radius):
        '''Calculate maximal; number of facets that can be placed at a certain radius.'''
        anglediff = (self.phi[1] - self.phi[0]) % (2. * np.pi)
        return radius * anglediff // self.d_facet

    def distribute_facets_on_arc(self, radius):
        '''Distribute facets on an arc.

        The facets are distributed as evenly as possible over the arc.

        Parameters
        ----------
        radius : float
            radius of arc where the facets are to be distributed.

        Returns
        -------
        centerangles : array
            The phi angles for centers of the facets at ``radius``.
        '''
        # arc is most crowded on inner radius
        n = self.max_facets_on_arc(radius - self.d_facet / 2)
        # total angle available
        anglediff = (self.phi[1] - self.phi[0]) % (2. * np.pi)
        facet_angle = self.d_facet / (2. * np.pi * radius)
        # thickness of space between facets, distributed equally
        d_between = (anglediff - n * facet_angle) / (n + 1)
        return d_between + 0.5 * facet_angle + np.arange(n) * (d_between + facet_angle)

    def max_facets_on_radius(self):
        return (self.radius[1] - self.radius[0]) % self.d_facet

    def select_radii(self):
        '''Distributes facets as evenly as possible along a radius'''
        n = self.max_facets_on_radius()
        d_between = (self.radius[1] - self.radius[0] - n * self.d_facet) / (n + 1)
        return d_between + 0.5 * self.d_facet + np.arange(n) * (d_between + self.d_facet)

    def place_facets(self):
        '''
        Returns
        -------
        pos4d : list of arrays
            List of affine transformations that bring an optical element centered
            on the origin of the coordinate system with the active plane in the
            yz-plane to the required facet position on the Rowland torus.
        '''
        pos4d = []
        radii = self.select_radii()
        for r in radii:
            angles = self.distribute_facets_on_arc(r)
            for a in angles:
                y = r * np.sin(a)
                z = r * np.cos(a)
                def f(x):
                    return self.rowland.quartic(x, y, z)
                x = optimize.brentq(f, self.z_min, self.z_max)
                # TBD: Error checking
                facet_pos = np.array([x, y, z])
                facet_normal = np.array(self.rowland.normal(x, y, z))
                # Find the rotation between [1, 0, 0] and the new normal
                rot_ax = np.cross([1., 0., 0.], facet_normal)
                if np.all(np.allclose(rot_ax), 0):
                    rot_mat = np.eye(4)  #vectors are parallel
                else:
                    rot_ang = np.arcsin(norm(rot_ax) / norm(facet_normal))
                    rot_mat = transforms3d.axangles.axangle2aff(rot_ax, rot_ang)
                pos4d.append(transforms3d.affines.compose(facet_pos, rot_mat[:3, :3], np.ones(3)))
        return pos4d

