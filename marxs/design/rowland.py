# Licensed under GPL version 3 - see LICENSE.rst
'''Tools for setting up instruments in the Rowland Torus geometry.

This includes an object that represents the Rowland torus itself (imaginatively
called `~marxs.design.rowland.RowlandTorus`), some helper functions used to set
the right parameters for the torus and a few classes that are derived from
`marxs.simulator.ParallelCalculated`, each placing elements such as gratings
or detectors on the Rowland torus. There are many ways to do that, e.g. the
limits can be defined in x,y,z coordinates or in theta, phi coordinates on the
torus, gratings can be ordered in concentric circles or pack densely in a
rectangular area etc. At this point, the different classes do not exploit all
these possibilities, they merely give a few of many possible ways to set up an
instrument.

These classes may be generalized in the future.
'''
import numpy as np
from scipy import optimize
import transforms3d

from ..optics.base import OpticalElement
from ..base import MarxsElement
from ..optics import FlatDetector
from ..math.utils import e2h, h2e, anglediff
from ..simulator import ParallelCalculated
from ..math.geometry import Geometry

__all__ = ['find_radius_of_photon_shell',
           'design_tilted_torus',
           'double_rowland_from_channel_distance',
           'add_offset_double_rowland_channels',
           'RowlandTorus',
           'ElementsOnTorus',
           'GratingArrayStructure',
           'RectangularGrid', 'CircularMeshGrid',
           ]


def find_radius_of_photon_shell(photons, mirror_shell, x, percentile=[1, 99]):
    '''Find the radius the photons coming from a single mirror shell have.

    For nested Wolter Type I mirrors the ray of photons reflected from a single
    mirror shell essentially form a cone in space. The tip of the cone is at
    the focal point and the base is at the mirror. There is a certain thickness
    to this cone depending on where exactly on the mirror the individual photon
    was reflected.

    This function takes a photon list of photons after passing through the
    mirror and calculates the radius range that this photon cone covers at a
    specific distance from the focal point. This information can help to design
    the placement of gratings.

    Parameters
    ----------
    photons : `~astropy.table.Table`
        Photon list with position and direction of photons leaving the mirror
    mirror_shell : int
        Select mirror shell to look at (uses column ``mirror_shell`` in
        ``photons`` for filtering).
    x : float
        Distance along the optical axis (assumed to coincide with the x axis
        with focal point at 0).
    percentile : list of floats
        The radius is calculated at the given percentiles. ``50`` would give
        the median radius. The default of ``[1, 99]`` gives a radius range
        excluding extreme outliers such as stray rays scattered into the extreme
        wing of the PSF.

    '''
    p = photons.copy()
    mdet = FlatDetector(position=np.array([x, 0, 0]), zoom=1e8, pixsize=1.)
    p = mdet(p)
    ind = (p['probability'] > 0) & (p['mirror_shell'] == mirror_shell)
    r = np.sqrt(p['det_x'][ind]**2+p['det_y'][ind]**2.)
    return np.percentile(r, percentile)


class RowlandTorus(MarxsElement, Geometry):
    '''Torus with y axis as symmetry axis.

    Note that the origin of the torus is the focal point, which
    **may or may not** be the same as the center of the torus.

    Parameters
    ----------
    R : float
        Radius of Rowland torus. ``r`` determines the radius of the Rowland
        circle, ``R`` is then used to rotate that circle around the axis of
        symmetry of the torus.
    r : float
        Radius of Rowland circle
    '''

    display = {'color': (1., 0.3, 0.3),
               'opacity': 0.2,
               'shape': 'torus; surface',
               'coo1': np.linspace(0, 2 * np.pi, 60),
               'coo2': np.linspace(0, 2 * np.pi, 60)}

    def __init__(self, R, r, **kwargs):
        self.R = R
        self.r = r
        # super does not work, because the various classes have different signature
        Geometry.__init__(self, kwargs)
        MarxsElement.__init__(self, **kwargs)

    def quartic(self, xyz, transform=True):
        '''Quartic torus equation.

        Roots of this equation are points on the torus.

        Parameters
        ----------
        xyz : np.array of shape (N, 3) or (3)
            Coordinates of points in euclidean space. The quartic is calculated
            for those points.
        transform : bool
            If ``True`` transform ``xyz`` from the global coordinate system
            into the local coordinate system of the torus. If this
            transformation is done in the calling function already, set to
        ``False``.

        Returns
        -------
        q : np.array of shape (N) or scalar
            Quartic at the input location
        '''
        if xyz.shape[-1] != 3:
            raise ValueError('Input coordinates must be defined in Euclidean space.')

        if transform:
            invpos4d = np.linalg.inv(self.pos4d)
            xyz = h2e(np.einsum('...ij,...j', invpos4d, e2h(xyz, 1)))
        return ((xyz**2).sum(axis=-1) + self.R**2. - self.r**2.)**2. - 4. * self.R**2. * (xyz[..., [0,2]]**2).sum(axis=-1)

    def solve_quartic(self, origin, v, transform=True):
        '''Solve the quartic on the Rowland torus.

        This method solves the quartic equation for positions on the Rowland
        Torus, i.e. it intersects a line with the torus. To that end, the
        location on the line is varied and the root of the quartic is found
        through numerical optimization.

        Parameters
        ----------
        origin : np.array
            Origin of line as homogeneous coordinate.
            This is also used as approximate starting point for the numerical
            optimization, so it should be reasonably close to the
            solution.
        v : np.array
            Direction of the line as homogeneous coordinate.
        transform : bool
            If ``True`` transform input from the global coordinate system
            into the local coordinate system of the torus. If this
            transformation is done in the calling function already, set to
            ``False``.

        Returns
        -------
        coo : np.array
            Position of intersection as homogeneous coordinate.

        '''
        origin = h2e(origin)
        v = h2e(v)
        def fun(k, origin, v):
            out = self.quartic(origin + k[..., None] * v, transform=transform)
            return out
        out = optimize.root(fun,
                            # The starting guess also serves to set the scale
                            # in the absence of a jacobian.
                            # So, can't start at 0. Instead, need to guess
                            # reasonable scale from torus parameters.
                            # r-R seems like a good guess but does not work at r = R.
                            # In that case, just pick a number that's small
                            # relative to r.
                            max(np.abs(self.r - self.R), self.r * 1e-2) * 0.1,
                            args=(origin, v))
        if not out.success:
            raise Exception('Intersection with torus not found.')
        else:
            return e2h(origin + out.x[..., None] * v, 1)

    def parametric_surface(self, theta, phi, display):
        '''Parametric representation of surface of torus.

        In contrast to `parametric` the input parameters here are 1-d arrays
        and the functions converts that to a rectangular grid.

        Parameters
        ----------
        theta : np.array
            Points on the Rowland circle (specified in rad, where 0 is on the
            optical axis across from the focal point).
        phi : np.array
            ``phi`` rotates the Rowland circle to make up the Rowland torus.

        Returns
        -------
        xyzw : np.array
            Torus coordinates in global homogeneous coordinate system.
        '''
        if (phi.ndim != 1) or (theta.ndim != 1):
            raise ValueError('input parameters have 1-dim shape.')
        theta, phi = np.meshgrid(theta, phi)
        return self.parametric(theta, phi)

    def parametric(self, theta, phi):
        '''Parametric description of points on the torus.

        This is just another way to obtain the shape of the torus, e.g.
        for visualization.

        Parameters
        ----------
        theta : np.array
            Points on the Rowland circle (specified in rad, where 0 is on the
            optical axis across from the focal point).
        phi : np.array
            ``phi`` rotates the Rowland circle to make up the Rowland torus.

        Returns
        -------
        xyzw : np.array
            Torus coordinates in global homogeneous coordinate system.
        '''
        x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        z = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        y = self.r * np.sin(theta)
        w = np.ones_like(z)
        torus = np.array([x, y, z, w]).T
        return np.einsum('...ij,...j', self.pos4d, torus)

    def xyzw2parametric(self, xyzw, transform=True, intersectvalid=True):
        '''Calculate (theta, phi) coordinates for point on torus surface.

        Parameters
        ----------
        xyzw : np.array of shape (N, 4)
            Coordinates of points on the torus surface in homogeneous
            coordinates.

        transform : bool
            If ``True`` transform ``xyz`` from the global coordinate system
            into the local coordinate system of the torus. If this
            transformation is done in the calling function already, set to
            ``False``.

        intersectvalid : bool
            When ``r >=R`` the torus can intersect with itself. At these points,
            phi is not unique. If ``intersectvalid`` is true, those points will
            be filled with an arbitrarily chosen valid value (0.), otherwise
            they will be nan.

        Returns
        -------
        theta, phi: np.array
            Parametric representation for all points on the torus.
        '''
        if transform:
            invpos4d = np.linalg.inv(self.pos4d)
            xyzw = np.einsum('...ij,...j', invpos4d, xyzw)

        xyz = h2e(xyzw)

        if not np.allclose(self.quartic(xyz, transform=False) / self.R**4., 0.):
            raise ValueError('Parametric representation is only defined for points on torus surface.')
        # There are two branches and we can use the sign to distinguish between them
        s = np.sign(np.sqrt(xyz[:,0]**2 + xyz[:, 2]**2) - self.R)
        theta = np.arcsin(s * xyz[:, 1] / self.r) + (s < 0) * np.pi

        factor = self.R + self.r * np.cos(theta)
        phi = np.zeros_like(factor)
        ind = factor != 0
        phi[ind] = np.arctan2(xyz[ind, 2] / factor[ind], xyz[ind, 0] / factor[ind])
        if not intersectvalid:
            phi[~factor] = np.nan

        return theta, phi

    def normal_parametric(self, theta, phi):
        '''Return the gradient vector field.

        Following the usual convention, the vector is pointing outwards
        of the torus volume.

        Parameters
        ----------
        theta, phi : 1-d np.arrays
            Theta and phi coordinates for points on the torus surface.

        Returns
        -------
        gradient : np.array
            Gradient vector field in homogeneous coordinates. One vector
            corresponds to each input point. The shape of ``gradient`` is the
            same as the shape of ``xyz``.
        '''
        if not ((theta.ndim == 1) and (phi.ndim == 1)):
            raise ValueError('theta and phi must be 1-d arrays.')
        normal = np.empty((len(theta), 3))
        normal[:, 0] = np.cos(theta) * np.cos(phi)
        normal[:, 1] = np.sin(theta)
        normal[:, 2] = np.cos(theta) * np.sin(phi)

        return np.einsum('...ij,...j', self.pos4d, e2h(normal, 0))

    def normal(self, xyzw):
        '''Return the gradient vector field.

        Following the usual conventions, the vector is pointing outwards
        of the torus volume.

        Parameters
        ----------
        xyzw : np.array of shape (N, 4)
            Coordinates of points in euclidean space. The quartic is calculated
            for those points. All points need to be on the surface of the torus.

        Returns
        -------
        gradient : np.array
            Gradient vector field in euclidean coordinates. One vector
            corresponds to each input point. The shape of ``gradient`` is the
            same as the shape of ``xyz``.
        '''
        if not len(xyzw.shape) == 2:
            raise ValueError('Shape of input array must be (N, 4).')
        theta, phi = self.xyzw2parametric(xyzw, 1)
        return self.normal_parametric(theta, phi)

    def xyz_from_radiusangle(self, radius, angle, start):
        '''Get Cartesian coordinates for radius, angle on the rowland circle.

        y, z are calculated from the radius and angle of polar coordinates in a
        plane; then x is determined from the condition that the point lies on
        the Rowland circle.  The plane is perpendicular to the optical axis
        that defines the Rowland circle.

        Parameters
        ----------
        radius, angle : float or np.array of shape (n,)
            Polar coordinates in a plane perpendicular to the optical axis
            (where the optical axis is parallel to the x-axis and goes through
            the origin of the `RowlandTorus`.
            ``angle=0`` coincides with the local y-axis.
        start : number
            Starting value for the search. The quartic can have up to four
            solutions because a line can intersect a torus in four points.
            The solution found thus depends on where the numerical root solver
            starts the search.

        Returns
        -------
        xyz : np.array of shape (n, 3)
            Euclidean coordinates in the global coordinate system.
        '''
        y = radius * np.cos(angle)
        z = radius * np.sin(angle)
        # Use np.mean for backwards compatibility.
        # This used to take a bracketing interval instead of a first guess.
        out = self.solve_quartic(np.array([np.mean(start), y, z, 1]),
                               np.array([1, 0, 0, 0]),
                               transform=False)
        return h2e(np.einsum('...ij,...j', self.pos4d, out))


def design_tilted_torus(f, alpha, beta):
    """Design a torus with specifications similar to Heilmann et al. 2010

    A `RowlandTorus` is fully specified with the parameters ``r``, ``R`` and
    ``pos4d``.  However, in practice, these numbers might be derived from other
    values.  This function calculates the parameters of a RowlandTorus, based
    on a different set of input values.

    Parameters
    ----------
    f : float
        distance between focal point and on-axis grating. Should be as large as
        possible given the limitations of the spacecraft to increase the
        resolution.
    alpha : float (in radian)
        angle between optical axis and the line (on-axis grating - center of
        Rowland circle).
        A typical value could be twice the blaze angle.
    beta : float (in radian)
        angle between optical axis and the line (on-axis grating - hinge),
        where the hinge is a point on the Rowland circle. The Rowland Torus
        will be constructed by rotating
        the Rowland Circle around the axis (focal point - hinge).
        The center of the Rowland Torus will be the point where the line
        (on-axis grating - center of Rowland circle) intersects the line
        (focal point - hinge).

    Returns
    -------
    R : float
        Radius of Rowland torus. ``r`` determines the radius of the Rowland
        circle, ``R`` is then used to rotate that circle around the axis of
        symmetry of the torus.
    r : float
        Radius of Rowland circle
    pos4d : np.array of shape (4, 4)

    Notes
    -----
    The geometry used here really needs to be explained in a figure.
    However, some notes to explain at least the meaning of the symbols on the
    code are in order:

    - Cat : position of on-axis CAT grating (where the Rowland circle intersects the on-axis beam)
    """
    r = f / (2. * np.cos(alpha))
    R = r * np.sin(np.pi / 2 - beta)
    orientation = transforms3d.axangles.axangle2mat([0, 0, -1], beta - alpha)
    x_Ct = f / 2 - R * np.cos(beta - alpha)
    y_Ct = f / 2 * np.tan(alpha) + R * np.sin(beta - alpha)
    pos4d = transforms3d.affines.compose([x_Ct, y_Ct, 0], orientation, np.ones(3))
    return R, r, pos4d


def double_rowland_from_channel_distance(d_BF, R, f):
    '''This prescription is borne out of the engineering needs where it turns out
    that the spacing between channels d_BF is actually an important parameter
    so it makes sense to use that here to parameterize the torus, too.

    f is the distance between a theoretical on-axis CAT grating and the focal
    point.  It's default is a little smaller than 12 m, to leave space to mount
    the SPOs.

    Parameters
    ----------
    d_BF : float
        Distance between the optical axes of two channels measured
        in the dispersion direction.
    R : float
        Radius of Rowland torus.
    f : float
        Distance between a theoretical on-axis CAT grating and the focal
        point.

    Returns
    -------
    geometry : dict
        Dictionary with the keys that hold the input parameters,
        derived dimensions of the geometry, and the constructed
        Rowland tori. Three Rowland tori are constructed: One for
        each channel called (rowland_central, rowland_central_m) and
        one for the detector (rowland_detector).

    '''
    d = 0.5 * d_BF
    r = 0.5 * np.sqrt(f**2 + d_BF**2)
    alpha = np.arctan2(d_BF, f)
    pos = [f - (r + R) * np.cos(alpha),
           (r + R) * np.sin(alpha),
           0]
    orient = [[np.cos(alpha), np.sin(alpha), 0],
              [-np.sin(alpha), np.cos(alpha), 0],
              [0., 0., 1]]


    posm = [f - (r + R) * np.cos(-alpha),
           (r + R) * np.sin(-alpha),
            0]

    orientm = [
        [np.cos(-alpha), np.sin(-alpha), 0],
        [-np.sin(-alpha), np.cos(-alpha), 0],
        [0.0, 0.0, 1],
    ]

    geometry = {'d_BF': d_BF,
                'd': d,
                'f': f,
                'rowland_central': RowlandTorus(R=R, r=r, position=pos,
                                                orientation=orient),
                'rowland_central_m': RowlandTorus(R=R, r=r, position=posm,
                                                  orientation=orientm)}

    geometry['pos_det_rowland'] = np.array([0, -d, 0, 1])
    geometry['shift_det_rowland'] = np.eye(4)
    geometry['shift_det_rowland'][:, 3] = geometry['pos_det_rowland']
    geometry['rowland_detector'] = RowlandTorus(R, r,
                                                pos4d=geometry['rowland_central'].pos4d)
    geometry['rowland_detector'].pos4d = np.dot(geometry['shift_det_rowland'],
                                                geometry['rowland_detector'].pos4d)
    geometry['pos_opt_ax'] = {'1': np.array([0, -d, 0, 1]),
                              '1m': np.array([0, d, 0, 1])}
    return geometry


def add_offset_double_rowland_channels(geometry,
                                       offsets={'1': [0, -2.5, -7.5],
                                                '1m': [0, -2.5, -2.5],
                                                '2': [0, 2.5, 2.5],
                                                '2m': [0, 2.5, 7.5],
                                                }):
    '''Generate Rowland tori for multiple spectral channels in a double-tori design.

    While this function can generate four separate channels (splitting the
    original and the original mirrored channel in two each).
    Rowland tori are added to the input geometry dictionary.

    Parameters
    ----------
    geometry : dict
        Dictionary with the keys from ` double_rowland_from_channel_distance`.
    offsets : dict
        Dictionary with the keys that hold the channel ids and the
        values that hold the offsets in the x, y, and z direction.
        The default value is set for a total of four channels, where each
        of the original and the original mirrored channel is split in two.
        However, the function can be used to generate any number of channels
        with the naming convention that channels 1, 2, 3, ... are generated
        from the original channel and channels 1m, 2m, 3m, ... are generated
        from the original mirrored channel.
    '''
    d = geometry['d']
    geometry['pos_opt_ax'] = {}
    for k in offsets:
        sign = 1 if 'm' in k else -1
        geometry['pos_opt_ax'][k] = np.array([0, sign * d, 0, 1])

    for k, v in offsets.items():
        geometry['pos_opt_ax'][k][:3] += v
    # Now offset that Rowland torus in a z axis by a few mm.
    # Shift is measured from point on symmetry plane.
    for channel in geometry['pos_opt_ax']:
        shift_matrix = np.eye(4)
        shift_matrix[:, 3] = geometry['pos_opt_ax'][channel][:]
        if channel in ['1', '2']:
            base_rowland = 'rowland_central'
        elif channel in ['1m', '2m']:
            base_rowland = 'rowland_central_m'
        namer = 'rowland_' + channel
        geometry[namer] = RowlandTorus(geometry[base_rowland].R,
                                       geometry[base_rowland].r,
                                       pos4d=geometry[base_rowland].pos4d)
        geometry[namer].pos4d = np.dot(shift_matrix,
                                       geometry[namer].pos4d)


class ElementsOnTorus(ParallelCalculated, OpticalElement):
    '''A collection of elements on a Rowland torus.

    When initialized, it places elements in the space available on the
    Rowland torus, most commonly, this class is used to place grating facets.

    After generation, individual facet positions can be adjusted by hand by
    editing the attributes `elem_pos` or `elem_uncertainty`. See
    `marxs.simulation.Parallel` for details.

    After any of the `elem_pos`, `elem_uncertainty` or
    `uncertainty` is changed, `generate_elements` needs to be
    called to regenerate the facets on the GAS.

    Parameters
    ----------
    rowland : `marxs.design.rowland.RowlandTorus`
        Torus on which the elements are placed.
    d_element : list of two floats
        Size of the edge of elements along the two (y and z in canonical marxs orientation)
        edges.
        ``d_element`` can be larger than the actual size of the silicon
        membrane or the active area of a CCD chip to accommodate a minimum
        thickness of the surrounding frame.
    guess_distance : float
        A ray can intersect a torus in up to four
        points. ``guess_distance`` specifies the starting distance for the numerical search for
        the intersection point to resolve this ambiguity.
        Default is the start close to outer rim of the torus.
    optimize_axis : np.array
        Homogeneous coordinate of the axis along which elements will be moved. This will
        usually coincide with the optical axis of the telescope.
        Default is the x-axis in the coordinate system of the torus.
    normal_spec : np.array or callable
        see `~marxs.simulator.simulator.ParallelCalculated` for details.
        Default for this class: Pointing **inward** into the Rowlandtorus,
        as one would do for an array of e.g. CCDs.
    parallel_spec : np.array or callable
        see `~marxs.simulator.simulator.ParallelCalculated` for details.
        Default for this class: Symmetry axis of the torus.
    '''
    def __init__(self, **kwargs):
        self.rowland = kwargs.pop('rowland')
        # Default to a slightly positive number
        self.guess_distance = kwargs.pop('guess_distance',
                                         self.rowland.r + self.rowland.R)
        self.d_element = kwargs.pop('d_element')
        self.optimize_axis = np.asanyarray(kwargs.pop('optimize_axis',
                                           self.rowland.pos4d @ np.array([1, 0, 0, 0])))
        self.id_col = kwargs.pop('id_col', 'facet')

        # Defaults changes from parent classes. Document! Or remove?
        if 'normal_spec' not in kwargs.keys():
            kwargs['normal_spec'] = lambda xyzw: -self.rowland.normal(xyzw)
        if 'parallel_spec' not in kwargs.keys():
            # If not given, make parallel to Rowland circle at origin.
            kwargs['parallel_spec'] = self.rowland.pos4d @ np.array([0, 1, 0, 0])
        kwargs['pos_spec'] = self.elempos

        super().__init__(**kwargs)

    def elemposyz(self):
        '''Specify the element position in the yz plane

        For the torus, the Rowland circle lies in the xy plane and the symmetry
        axis of the torus is the y-axis. In this class, the position of the
        elements is specified in the yz plane in the coordinates of the
        torus and the remaining coordinate.

        This function will be customized by derived classes.

        Returns
        -------
        ypos, zpos : np.array
            1D arrays of y and z positions for the elements distributed in 2D.
        '''
        raise NotImplementedError

    def elempos(self):
        '''Generate 3D positions of elements

        For each element, y and z coordinates in the torus coordinate system
        are generated from ``self.elemposyz`` and the elements are placed at
        x=0. Then, that initial coordinate is transformed into the global
        coordinate system and moved by
        ``self.guess_distance * self.optimize_axis``. The resulting position
        is the starting point for numerical optimization along
        ``self.optimize_axis`` to find the on-torus location.
        '''
        ypos, zpos = self.elemposyz()
        posyz = np.vstack([np.zeros_like(ypos), ypos, zpos])
        origin = np.einsum('...ij,...j', self.rowland.pos4d, e2h(posyz.T, 1)) + self.guess_distance * self.optimize_axis
        # solve_quartic is not vectorized (because scipy.optimize is not)
        # so need to pass in one element at a time.

        # now, origin is in the global coordinate system.
        # Because transform=True is the default in solve_quartic that works.
        return np.vstack([self.rowland.solve_quartic(o, self.optimize_axis) for o in origin])


class RectangularGrid(ElementsOnTorus):
    '''A collection of diffraction gratings on the Rowland torus.

    This class is similar to ``marxs.design.rowland.GratingArrayStructure`` but
    instead of placing elements on concentric circles, they are placed to fill
    a rectangular area.

    When initialized, it places elements in the space available on the
    Rowland circle, most commonly, this class is used to place grating facets.

    After generation, individual facet positions can be adjusted by hand by
    editing the attributes `elem_pos` or `elem_uncertainty`. See
    `marxs.simulation.Parallel` for details.

    After any of the `elem_pos`, `elem_uncertainty` or
    `uncertainty` is changed, `generate_elements` needs to be
    called to regenerate the facets on the GAS.

    Parameters
    ----------
    y_range, z_range: list of two floats
        Limits of the rectangular area where gratings are placed.
        To place only one element, make both limits the same, e.g.
        ``z_range=[5, 5]`` will place one element in each row in z
        centered on $z=5$.
        This is given in the coordinate system of the torus, i.e. y is the
        plane of the Rowland circle and both numbers are measured from the
        center of the Rowland torus.
        Default for z-range: [-1e-10, 1e-10]. This will make one element
        centered on 0 in z-range.

    '''
    def __init__(self, **kwargs):
        self.y_range = kwargs.pop('y_range')
        self.z_range = kwargs.pop('z_range', [-1e-10, 1e-10])

        super().__init__(**kwargs)

    def elemposyz(self):
        n_y = int(np.ceil((self.y_range[1] - self.y_range[0]) / self.d_element[0]))
        n_z = int(np.ceil((self.z_range[1] - self.z_range[0]) / self.d_element[1]))
        n_y = max(1, n_y)
        n_z = max(1, n_z)

        # n_y and n_z are rounded up, so they cover a slightly larger range than y/z_range
        width_y = n_y * self.d_element[0]
        width_z = n_z * self.d_element[1]

        ypos = np.arange(0.5 * (self.y_range[0] - width_y + self.y_range[1] + self.d_element[0]), self.y_range[1], self.d_element[0])
        zpos = np.arange(0.5 * (self.z_range[0] - width_z + self.z_range[1] + self.d_element[1]), self.z_range[1], self.d_element[1])

        ypos, zpos = np.meshgrid(ypos, zpos)

        return ypos.flatten(), zpos.flatten()


class CircularMeshGrid(ElementsOnTorus):
    '''A collection of diffraction gratings on the Rowland torus filling a circle.

    When initialized, it places elements in the space available on the
    Rowland circle, most commonly, this class is used to place grating facets.

    After generation, individual facet positions can be adjusted by hand by
    editing the attributes `elem_pos` or `elem_uncertainty`. See
    `marxs.simulation.Parallel` for details.

    After any of the `elem_pos`, `elem_uncertainty` or
    `uncertainty` is changed, `generate_elements` needs to be
    called to regenerate the facets on the GAS.

    Parameters
    ----------
    radius : list of two floats
        Inner and outer radius of the circle.
    '''

    def __init__(self, **kwargs):
        self.radius = kwargs.pop('radius')
        super().__init__(**kwargs)

    def elemposyz(self):
        n_y = int(np.ceil(2 * self.radius[1] / self.d_element[0]))
        y_width = n_y * self.d_element[0]
        y_pos = np.arange(- y_width / 2, self.radius[1], self.d_element[0])

        ypos = []
        zpos = []

        for y in y_pos:
            if np.abs(y) > self.radius[1]:
                # Outermost layer. Center might be outside outer radius
                z = np.array([0])
            else:
                z_outer = np.sqrt(self.radius[1]**2 - y**2)
                if np.abs(y) > self.radius[0]:
                    n_z = int(np.ceil(2 * z_outer / self.d_element[1]))
                    z_width = n_z * self.d_element[1]
                    z = np.arange(- z_width / 2, z_outer, self.d_element[1])
                else:
                    z_inner = np.sqrt(self.radius[0]**2 - y**2)
                    z_mid = 0.5 * (z_inner + z_outer)
                    n_z = int(np.ceil((z_outer - z_inner) / self.d_element[1]))
                    z_width = n_z * self.d_element[1]
                    z = np.arange(z_mid - z_width / 2, z_outer, self.d_element[1])
                    z = np.hstack([-z, z])

            ypos.extend([y] * len(z))
            zpos.extend(z)
        return ypos, zpos


class GratingArrayStructure(ElementsOnTorus):
    r"""Collection of diffraction gratings on the Rowland Torus

    When a ``GratingArrayStructure`` (GAS) is initialized, it places elements
    in the space available on the Rowland circle, most commonly, this class is
    used to place grating facets.

    After generation, individual facet positions can be adjusted by hand by
    editing the attributes `elem_pos` or `elem_uncertainty`. See `Parallel` for
    details.

    After any of the `elem_pos`, `elem_uncertainty` or `uncertainty` is
    changed, `generate_elements` needs to be called to regenerate the facets on
    the GAS.

    Parameters
    ----------
    radius : list of 2 floats
        Inner and outer radius of the GAS as measured in the yz-plane from the
        origin.

    phi : float or list of 2 floats
        Bounding angles for a segment covered by the GSA. :math:`\phi=0` is on
        the positive y axis. The segment fills the space from ``phi1`` to
        ``phi2`` in the usual mathematical way (counterclockwise).  Angles are
        given in radian. Note that ``phi[1] < phi[0]`` is possible if the
        segment crosses the y axis.
        Alternatively, ``phi`` can just be a single number. In that case, there
        will be exactly one element per radius.

    """
    def __init__(self, **kwargs):
        self.phi = kwargs.pop('phi', [0., 2*np.pi])
        self.radius = kwargs.pop('radius')
        super().__init__(**kwargs)

    def max_elements_on_radius(self, radius):
        '''Distribute elements on a radius.

        Parameters
        ----------
        radius : list of two floats
            inner and outer radius that should be covered by elements

        Returns
        -------
        n : int
            Number of elements needed to cover a given radius segment.
            Elements might reach beyond the radius limits if the difference between
            inner and outer radius is not an integer multiple of the element size.
        '''
        return int(np.ceil((radius[1] - radius[0]) / self.d_element[0]))

    def distribute_elements_on_radius(self):
        '''Distributes elements as evenly as possible along a radius.

        .. note::
           Unlike `distribute_elements_on_arc`, this function will have
           elements reaching beyond the limits of the radius, if the distance
           between inner and outer radius is not an integer multiple of the
           element size.

        Returns
        -------
        radii : np.ndarray
            Radii of the element *center* positions.

        '''
        radii = []
        for i in range(len(self.radius) // 2):
            radiusbracket = self.radius[2 * i: 2 * i + 2]
            n = self.max_elements_on_radius(radiusbracket)
            radii.append(np.mean(radiusbracket) +
                         np.arange(- n / 2 + 0.5, n / 2 + 0.5) * self.d_element[0])
        return np.hstack(radii)


    def max_elements_on_arc(self, radius):
        '''Calculate maximal number of elements that can be placed at a certain radius.

        Parameters
        ----------
        radius : float
            Radius of circle where the centers of all elements will be placed.
        '''
        return radius * anglediff(self.phi) // self.d_element[1]

    def distribute_elements_on_arc(self, radius):
        '''Distribute elements on an arc.

        The elements are distributed as evenly as possible over the arc.

        .. note::

          Contrary to `distribute_elements_on_radius`, elements never stretch
          beyond the limits set by the ``phi`` parameter of the GAS. If an arc
          segment is not wide enough to accommodate at least a single element,
          it will go empty.
          (The exception to that is is ``phi`` is a single value. In that case,
          there will be exactly one element.)

        Parameters
        ----------
        radius : float
            radius of arc where the elements are to be distributed.

        Returns
        -------
        centerangles : array
            The phi angles for centers of the elements at ``radius``.

        '''
        if len(self.phi) == 1:
            return self.phi
        # arc is most crowded on inner radius
        n = self.max_elements_on_arc(radius - self.d_element[0] / 2)
        element_angle = self.d_element[1] / (2. * np.pi * radius)
        # thickness of space between elements, distributed equally
        d_between = (anglediff(self.phi) - n * element_angle) / (n + 1)
        centerangles = d_between + 0.5 * element_angle + np.arange(n) * (d_between + element_angle)
        return (self.phi[0] + centerangles) % (2. * np.pi)

    def elemposyz(self):
        angles = []
        radii = self.distribute_elements_on_radius()
        for r in radii:
            angles.append(self.distribute_elements_on_arc(r))
        # Turn lists of lists into flat numpy array
        radii = np.concatenate([[radii[i]] * len(a) for i, a in enumerate(angles)])
        angles = np.concatenate(angles)
        return radii * np.sin(angles), radii * np.cos(angles)
