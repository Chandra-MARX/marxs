from __future__ import division

import numpy as np
from scipy import optimize
from astropy import table
import transforms3d
from transforms3d.affines import decompose44

from ..optics.base import OpticalElement
from ..base import _parse_position_keywords, MarxsElement
from ..optics import FlatDetector
from ..math.utils import translation2aff, zoom2aff, mat2aff
from ..math.rotations import ex2vec_fix
from ..math.pluecker import e2h, h2e


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


class RowlandTorus(MarxsElement):
    '''Torus with z axis as symmetry axis

    Parameters
    ----------
    R : float
        Radius of Rowland torus. ``r`` determines the radius of the Rowland circle,
        ``R`` is then used to rotate that circle around the axis of symmetry of the torus.
    r : float
        Radius of Rowland circle
    '''
    def __init__(self, R, r, **kwargs):
        self.R = R
        self.r = r
        self.pos4d = _parse_position_keywords(kwargs)
        super(RowlandTorus, self).__init__(**kwargs)

    def quartic(self, xyz, transform=True):
        '''Quartic torus equation.

        Roots of this equation are points on the torus.

        Parameters
        ----------
        xyz : np.array of shape (N, 3) or (3)
            Coordinates of points in euklidean space. The quartic is calculated for
            those points.
        transform : bool
            If ``True`` transform ``xyz`` from the global coordinate system into the
            local coordinate system of the torus. If this transformation is done in the
            calling function already, set to ``False``.

        Returns
        -------
        q : np.array of shape (N) or scalar
            Quartic at the input location
        '''
        if xyz.shape[-1] != 3:
            raise ValueError('Input coordinates must be defined in Eukledian space.')

        if transform:
            invpos4d = np.linalg.inv(self.pos4d)
            xyz = h2e(np.einsum('...ij,...j', invpos4d, e2h(xyz, 1)))
        return ((xyz**2).sum(axis=-1) + self.R**2. - self.r**2.)**2. - 4. * self.R**2. * (xyz[..., :2]**2).sum(axis=-1)

    def solve_quartic(self, x=None, y=None, z=None, interval=[0, 1]):
        '''Solve the quartic for points on the Rowland torus in Cartesian coordinates.

        This method solves the quartic equation for positions on the Rowland Torus for
        cases where two of the Cartesian coordinates are fixed (e.g. y and z) and the third
        one (e.g. x) needs to be computed. This function is intended as a convenience for a
        common use case. In more general cases, evaluate the :meth:`RowlandTorus.quartic` and
        search for the roots of that function.

        Parameters
        ----------
        x, y, z : float or None
            Set two of these coordinates to fixed numbers. This method will solve for the
            coordinate set to ``None``.
            x, y, z are defined in the global coordinate system.
        interval : np.array
            [min, max] for the search. The quartic can have up to for solutions because a
            line can intersect a torus in four points and this interval must bracket one and only
            one solution.

        Returns
        -------
        coo : float
            Value of the fitted coordinate.
        '''
        n_Nones = 0
        for i, c in enumerate([x, y, z]):
            if c is None:
                n_Nones +=1
                ind = i
        if n_Nones != 1:
            raise ValueError('Exactly one of the input numbers for x,y,z must be None.')
        # Need to give it a number for vstack to work
        if ind == 0: x = 0.
        if ind == 1: y = 0.
        if ind == 2: z = 0.

        xyz = np.vstack([x,y,z]).T
        def f(val_in):
            xyz[..., ind] = val_in
            return self.quartic(xyz)
        val_out, brent_out = optimize.brentq(f, interval[0], interval[1], full_output=True)
        if not brent_out.converged:
            raise Exception('Intersection with torus not found.')
        return val_out


    def normal(self, xyz):
        '''Return the gradient vector field.

        Parameters
        ----------
        xyz : np.array of shape (N, 3) or (3)
            Coordinates of points in euklidean space. The quartic is calculated for
            those points. All points need to be on the surface of the torus.

        Returns
        -------
        gradient : np.array
            Gradient vector field in euklidean coordinates. One vector corresponds to each
            input point. The shape of ``gradient`` is the same as the shape of ``xyz``.
        '''
        # For r,R  >> 1 even marginal differences lead to large
        # numbers on the quartic because of R**4 -> normalize
        invpos4d = np.linalg.inv(self.pos4d)
        xyz = h2e(np.einsum('...ij,...j', invpos4d, e2h(xyz, 1)))

        if not np.allclose(self.quartic(xyz, transform=False) / self.R**4., 0.):
            raise ValueError('Gradient vector field is only defined for points on torus surface.')
        factor = 4. * ((xyz**2).sum(axis=-1) + self.R**2. - self.r**2)
        dFdx = factor * xyz[..., 0] - 8. * self.R**2 * xyz[..., 0]
        dFdy = factor * xyz[..., 1] - 8. * self.R**2 * xyz[..., 1]
        dFdz = factor * xyz[..., 2]
        gradient = np.vstack([dFdx, dFdy, dFdz]).T
        return h2e(np.einsum('...ij,...j', self.pos4d, e2h(gradient, 0)))

def design_tilted_torus(f, alpha, beta):
    '''Design a torus with specifications similar to Heilmann et al. 2010

    A `RowlandTorus` is fully specified with the parameters ``r``, ``R`` and ``pos4d``.
    However, in practice, these numbers might be derived from other values.
    This function calculates the parameters of a RowlandTorus, based on a different
    set of input values.

    Parameters
    ----------
    f : float
        distance between focal point and on-axis grating. Should be as large as
        possible given the limitations of the spacecraft to increase the resolution.
    alpha : float (in radian)
        angle between optical axis and the line (on-axis grating - center of Rowland circle).
        A typical value could be twice the blaze angle.
    beta : float (in radian)
        angle between optical axis and the line (on-axis grating - hinge), where the hinge
        is a point on the Rowland circle. The Rowland Torus will be constructed by rotating
        the Rowland Circle around the axis (focal point - hinge).
        The center of the Rowland Torus will be the point where the line
        (on-axis grating - center of Rowland circle) intersects the line
        (focal point - hinge).

    Returns
    -------
    R : float
        Radius of Rowland torus. ``r`` determines the radius of the Rowland circle,
        ``R`` is then used to rotate that circle around the axis of symmetry of the torus.
    r : float
        Radius of Rowland circle
    pos4d : np.array of shape (4, 4)

    Notes
    -----
    The geometry used here really needs o be explained in a figure.
    However, some notes to explain at least the meaning of the symbols on the code
    are in order:

    - Cat : position of on-axis CAT grating (where the Rowland circle intersects the on-axis beam)
    - H : position of hinge
    - Ct : Center of Rowland Torus
    - F : Focal point on axis (at the origin of the coordinate system)
    - CatH, HF, FCt, etc. : distance between Cat and H, F and Ct, etc.
    - gamma : see sketch.
    '''
    r = f / (2. * np.cos(alpha))
    CatH = r * np.sqrt(2 * (1 + np.cos(2 * (beta - alpha))))
    HF = np.sqrt(f**2 + CatH**2 - 2 * f * CatH * np.cos(beta))
    gamma = np.arccos(HF / (2 * r))
    R = f / np.sin(np.pi - alpha - (alpha + gamma)) * np.sin(alpha + gamma) - r
    FCt = f / np.sin(np.pi - alpha - (alpha + gamma)) * np.sin(alpha)
    x_Ct = FCt * np.cos(alpha + gamma)
    y_Ct = 0
    z_Ct = FCt * np.sin(alpha + gamma)
    orientation = transforms3d.axangles.axangle2mat([0,1,0], np.pi/2 - alpha - gamma)
    pos4d = transforms3d.affines.compose([x_Ct, y_Ct, z_Ct], orientation, np.ones(3))
    return R, r, pos4d

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
    output_cols = ['facet']

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
        return self.xyz_from_ra(r, a).flatten()

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
        x = self.rowland.solve_quartic(y=y,z=z, interval=self.x_range)
        return np.vstack([x,y,z]).T

    def facet_position(self):
        '''Calculate ideal facet positions based on rowland geometry.

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
                facet_pos = self.xyz_from_ra(r, a).flatten()
                #facet_normal = np.array(self.rowland.normal(x, y, z))
                # Find the rotation between [1, 0, 0] and the new normal
                # Keep grooves (along e_y) parallel to e_y
                rot_mat = ex2vec_fix(facet_pos, np.array([0., 1., 0.]))

                pos4d.append(transforms3d.affines.compose(facet_pos, rot_mat, np.ones(3)))
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
        # _parse_position_keywords pops off keywords, thus operate on a copy here
        facet_args = facet_args.copy()
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
                      mat2aff(rrown),   # rotate grating normal to be normal to Rowland torus
                      mat2aff(rfacet),  # Any rotation of facet, e.g. for CAT gratings
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
        self.add_output_cols(photons)

        for i, f in enumerate(self.facets):
            intersect, interpos, intercoos = f.intersect(photons['dir'].data, photons['pos'].data)
            p_out = f.process_photons(photons, intersect, interpos, intercoos)
            p_out['facet'] = i

        return photons

    def intersect(self, photons):
        raise NotImplementedError
