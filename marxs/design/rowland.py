from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy import optimize
from astropy import table
import transforms3d
from transforms3d.affines import decompose44

from ..optics.base import OpticalElement, _parse_position_keywords
from ..optics import FlatDetector
from ..math.utils import translation2aff, zoom2aff, mat2aff


def find_radius_of_photon_shell(photons, mirror_shell, x, percentile=[1,99]):
    '''Find the radius the photons coming from a single mirror shell have.

    For nested Wolter Type I mirrors the ray of photons reflected from a single mirror
    shell essentially form a cone in space. The tip of the cone is at the focal point
    and the base is at the mirror. There is a certain thickness to this cone depending
    on where exactly on the mirror the individual photon was reflected.

    This function takes a photon list of photons after passing through the mirror and
    calculates the radius range that this photon cone covers at a specific distance from
    the focal point. This information can help to design the placement of gratings.

    Parameters
    ----------
    photons : `~astropy.table.Table`
        Photon list with position and direction of photons leaving the mirror
    mirror_shell : int
        Select mirror shell to look at (uses column ``mirror_shell`` in ``photons``
        for filtering).
    x : float
        Distance along the optical axis (assumed to coincide with the x axis with focal point
        at 0).
    percentile : list of floats
        The radius is calculated at the given percentiles. ``50`` would give the median radius.
        The default of ``[1, 99]`` gives a radius range excluding extrem outliers such as
        stray rays scattered into the extreme wing of the PSF.
    '''
    p = photons[:]
    mdet = FlatDetector(position=np.array([x, 0, 0]), zoom=1e8, pixsize=1.)
    p = mdet.process_photons(p)
    ind = (p['probability'] > 0) & (p['mirror_shell'] == 0)
    r = np.sqrt(p['det_x'][ind]**2+p['det_y'][ind]**2.)
    return np.percentile(r, percentile)


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
        # For r,R  >> 1 even marginal differences lead to large
        # numbers on the quartic because of R**4 -> normalize
        if not np.allclose(self.quartic(x, y, z) / self.R**4., 0.):
            raise ValueError('Gradient vector field is only defined for points on torus surface.')
        factor = 4. * (x**2. + y**2. + z**2. + self.R**2. - self.r**2)
        dFdx = factor * x - 8. * self.R**2 * x
        dFdy = factor * y - 8. * self.R**2 * y
        dFdz = factor * z
        return [dFdx, dFdy, dFdz]


class FacetPlacementError(Exception):
    pass


class GratingArrayStructure(OpticalElement):
    '''

    When a ``GratingArrayStructure`` (GAS) is initialized, it places as many
    facets as possible in the space available. Those facets are positioned
    on the Rowland circle.

    After generation, individual facet positions can be adjusted by hand by
    editing the :attribute:`facet_pos`.
    Also, additional misalingments for each facet can be introduced by
    editing :attribute:`facet_uncertainty`, e.g. to represent uncertainties
    in the manufacturing process. This attribute holds a list of affine
    transformation matrices.
    The global position and rotation of the total GAS can be changed with
    :attribute:`uncertainty`, e.g. the represent the reproducibility of
    inserting the gratings into the beam for separate observations. The
    uncertainty is expressed as an affine transformation matrix.

    All uncertianty metrices should only consist of translation and rotations
    and all uncertainties should be relatively small.

    After any of the :attribute:`facet_pos`, :attribute:`facet_uncertainty` or
    :attribute:`uncertainty` is changed, :method:`generate_facets` needs to be
    called to renerate the facets on the GAS.
    This mechanism can be used to estimate the influence of manufacturing
    uncertainties. First, run a simulation with an ideal GAS, then change
    the values, regenrate the facets and rerun te simulation. Comparing the
    results will allow you to estiamte the effect of the manufacturing
    misalignments.

    The order in which all the transformations are applied to the facet is
    chosen, such that all rotations are done around the actual center of the
    facet or GAS respoectively. "Uncertainty" roations are always done *after*
    all other rotations are accounted for.

    Use ``pos4d`` in ``facetargs`` to set a fixed rotation for all facets, e.g.
    for a CAT grating.

    Parameters
    ----------
    rowland : RowlandTorus
    d_facet : float
        Size of the edge of a facet, which is assumed to be flat and square.
        (``d_facet`` can be larger than the actual size of the silicon membrane to
        accommodate a minimum thickness of the surrounding frame.)
    x_range: list of 2 floats
        Minimum and maximum of the x coordinate that is searched for an intersection
        with the torus.
    radius : list of 2 floats float
        Inner and outer radius of the GAS as measured in the yz-plane from the
        origin.
    phi : list of 2 floats
        Bounding angles for a segment covered by the GSA. :math:`\phi=0`
        is on the positive y axis. The segment fills the space from ``phi1`` to
        ``phi2`` in the usual mathematical way (counterclockwise).
        Angles are given in radian. Note that ``phi[1] < phi[0]`` is possible if
        the segment crosses the y axis.
    '''
    def __init__(self, rowland, d_facet, x_range, radius, phi=[0., 2*np.pi], **kwargs):
        self.rowland = rowland
        if not (radius[1] > radius[0]):
            raise ValueError('Outer radius must be larger than inner radius.')
        if np.min(radius) < 0:
            raise ValueError('Radius must be positive.')
        self.radius = radius

        if np.max(np.abs(phi)) > 10:
            raise ValueError('Input angles >> 2 pi. Did you use degrees (radian expected)?')
        self.phi = phi
        self.x_range = x_range
        self.d_facet = d_facet
        self.facet_class = kwargs.pop('facetclass')
        self.facet_args = kwargs.pop('facetargs', {})

        super(GratingArrayStructure, self).__init__(**kwargs)

        self.uncertainty = np.eye(4)
        self.facet_pos = self.facet_position()
        self.facet_uncertainty = [np.eye(4)] * len(self.facet_pos)
        self.generate_facets(self.facet_class, self.facet_args)

    def calc_ideal_center(self):
        '''Position of the center of the GSA, assuming placement on the Rowland circle.'''
        anglediff = (self.phi[1] - self.phi[0]) % (2. * np.pi)
        a = (self.phi[0] + anglediff / 2 ) % (2. * np.pi)
        r = sum(self.radius) / 2
        return self.xyz_from_ra(r, a)

    def anglediff(self):
        '''Angles range covered by facets, accounting for 2 pi properly'''
        anglediff = (self.phi[1] - self.phi[0])
        if (anglediff < 0.) or (anglediff > (2. * np.pi)):
            # If anglediff == 2 pi exactly, presumably the user want to cover the full circle.
            anglediff = anglediff % (2. * np.pi)
        return anglediff

    def max_facets_on_arc(self, radius):
        '''Calculate maximal; number of facets that can be placed at a certain radius.'''
        return radius * self.anglediff() // self.d_facet

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
        facet_angle = self.d_facet / (2. * np.pi * radius)
        # thickness of space between facets, distributed equally
        d_between = (self.anglediff() - n * facet_angle) / (n + 1)
        centerangles = d_between + 0.5 * facet_angle + np.arange(n) * (d_between + facet_angle)
        return (self.phi[0] + centerangles) % (2. * np.pi)

    def max_facets_on_radius(self):
        return (self.radius[1] - self.radius[0]) // self.d_facet

    def distribute_facets_on_radius(self):
        '''Distributes facets as evenly as possible along a radius'''
        n = self.max_facets_on_radius()
        d_between = (self.radius[1] - self.radius[0] - n * self.d_facet) / (n + 1)
        return self.radius[0] + d_between + 0.5 * self.d_facet + np.arange(n) * (d_between + self.d_facet)

    def xyz_from_ra(self, radius, angle):
        '''Get Cartesian coordiantes for radius, angle and the rowland circle.

        y,z are calculated from the radius and angle of polar coordiantes in a plane;
        then x is determined from the condition that the point lies on the Rowland circle.
        '''
        y = radius * np.sin(angle)
        z = radius * np.cos(angle)
        def f(x):
            return self.rowland.quartic(x, y, z)
        x, brent_out = optimize.brentq(f, self.x_range[0], self.x_range[1], full_output=True)
        if not brent_out.converged:
            raise Exception('Intersection with torus not found.')
        return x, y, z

    def facet_position(self):
        '''
        Returns
        -------
        pos4d : list of arrays
            List of affine transformations that bring an optical element centered
            on the origin of the coordinate system with the active plane in the
            yz-plane to the required facet position on the Rowland torus.
        '''
        pos4d = []
        radii = self.distribute_facets_on_radius()
        for r in radii:
            angles = self.distribute_facets_on_arc(r)
            for a in angles:
                x, y, z = self.xyz_from_ra(r, a)
                facet_pos = np.array([x, y, z])
                facet_normal = np.array(self.rowland.normal(x, y, z))
                # Find the rotation between [1, 0, 0] and the new normal
                rot_ax = np.cross([1., 0., 0.], facet_normal)
                if np.allclose(rot_ax, 0):
                    rot_mat = np.eye(4)  # vectors are parallel
                else:
                    rot_ang = np.arcsin(norm(rot_ax) / norm(facet_normal))
                    rot_mat = transforms3d.axangles.axangle2aff(rot_ax, rot_ang)
                pos4d.append(transforms3d.affines.compose(facet_pos, rot_mat[:3, :3], np.ones(3)))
        return pos4d

    def generate_facets(self, facet_class, facet_args={}):
        '''
        Example
        -------
        from marxs.optics.grating import FlatGrating
        gsa = GSA( ... args ...)
        gsa.generate_facets(FlatGrating, {'d': 0.002})
        '''
        self.facets = []

        facet_pos4d = _parse_position_keywords(facet_args)
        tfacet, rfacet, zfacet, Sfacet = decompose44(facet_pos4d)
        if not np.allclose(Sfacet, 0.):
            raise ValueError('pos4 for facet includes shear, which is not supported for gratings.')
        name = facet_args.pop('name', '')

        gas_center = self.calc_ideal_center()
        Tgas = translation2aff(gas_center)

        for i in range(len(self.facet_pos)):
            f_center, rrown, ztemp, stemp = decompose44(self.facet_pos[i])
            Tfacetgas = translation2aff(-np.array(gas_center) + f_center)
            tsigfacet, rsigfacet, ztemp, stemp = decompose44(self.facet_uncertainty[i])
            if not np.allclose(ztemp, 1.):
                raise FacetPlacementError('Zoom is not supported in the facet uncertainty.')
            if not np.allclose(stemp, 0.):
                raise FacetPlacementError('Shear is not supported in the facet uncertainty.')
            # Will be able to write this so much better in python 3.5,
            # but for now I don't want to nest np.dot too much so here it goes
            f_pos4d = np.eye(4)
            for m in reversed([self.pos4d,  # any change between GAS system and global
                      # coordiantes, e.g. if x is not optical axis
                      Tgas,  # move to center of GAS
                      self.uncertainty,  # uncertainty in GAS positioning
                      Tfacetgas,  # translate facet center to GAS center
                      translation2aff(tsigfacet),  # uncertaintig in translation for facet
                      translation2aff(tfacet),  # any additional offset of facet. Probably 0
                      mat2aff(rsigfacet),  # uncertainty in rotation for facet
                      mat2aff(rfacet),  # Any rotation of facet, e.g. for CAT gratings
                      mat2aff(rrown),   # rotate grating normal to be normal to Rowland torus
                      zoom2aff(zfacet),  # sets size of grating
                     ]):
                assert m.shape == (4, 4)
                f_pos4d = np.dot(m, f_pos4d)
            self.facets.append(facet_class(pos4d = f_pos4d, name='{0}  Facet {1} in GAS {2}'.format(name, i, self.name), **facet_args))

    def process_photons(self, photons):
        '''

        This is a simple brute-force implementation. It does assume that every
        photon will interact with one grating at most, but is not any more clever
        than that. This means that a lot of relatively expansive intersection
        calculations are done.

        In those designs that we study, the GAS is almost flat (because the Rowland
        circle is large), so I could implement e.g.:

        - a much faster "approximate" intersection test by projection on a x = const plane
          (possibly as a kdtree from scipy.spatial).
        '''
        photons_out = []
        for i, f in enumerate(self.facets):
            intersect, interpos, temp = f.intersect(photons['dir'], photons['pos'])
            p_out = f.process_photons(photons[intersect], interpos[intersect])
            p_out['facet'] = [i] * intersect.sum()
            photons_out.append(p_out)
            photons = photons[~intersect]
        # append photons that did not intersect a facet
        photons_out.append(photons)
        return table.vstack(photons_out)

    def intersect(self, photons):
        raise NotImplementedError
