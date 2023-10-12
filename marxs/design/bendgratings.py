# Licensed under GPL version 3 - see LICENSE.rst
'''This module contains functions to modify gratings.

After flat simple gratings have been placed on a Rowland-torus, they can be
changed to more complicated geometries or set-ups. For example, placing just
the center of the grating on the Rowland-torus, can leave the edges to deviate.
In this case, gratings might be bend in one dimension or a more complex
arrangement of grating bars (a "chirp") could be specified. Conceptually, it
is often useful to think of this as a multi-step process for the simulation,
so in practice a chirp would have to be known in grating manufacturing.

To support this conceptual idea (place the gratings, then rotate it, then do X,
then chirp it) the functions in this module operate on exciting grating objects.
'''
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import RectBivariateSpline
from transforms3d.affines import decompose, decompose44

from marxs.math.geometry import Cylinder
from marxs.math.utils import h2e, e2h, norm_vector
from marxs.utils import generate_test_photons
from marxs.optics import OrderSelector


def bend_gratings(gratings, radius):
    '''Bend gratings to follow the Rowland cirle.

    Gratings are bend in one direction (the dispersion direction) only.

    The numerical procedure used to calculate the bending takes the central ray as
    a fixed ppoint assuming that the central ray always goes to the correct position!

    Parameters
    ----------
    gratings : list
        List of gratings to be bend
    radius : float
        Radius of the newly bend gratings
    '''
    for e in gratings:
        t, rot, z, s = decompose(e.geometry.pos4d)
        d_phi = np.arctan(z[1] / radius)
        c = Cylinder({'position': t - radius * h2e(e.geometry['e_x']),
                      'orientation': rot,
                      'zoom': [radius, radius, z[2]],
                      'phi_lim': [-d_phi, d_phi]})
        c._geometry = e.geometry._geometry
        e.geometry = c
        e.display['shape'] = 'surface'
        for e1 in e.elements:
            # can't be the same geometry, because groove_angle is part of _geometry and that's different
            # Maybe need to get that out again and make the geometry strictly the geometry
            # But for now, make a new cylinder of each of them
            # Even now, not sure that's needed, since intersect it run by FlatStack
            c = Cylinder({'position': t - radius * h2e(e.geometry['e_x']),
                          'orientation': rot,
                          'zoom': [radius, radius, z[2]],
                          'phi_lim': [-d_phi, d_phi]})
            c._geometry = e1.geometry._geometry
            e1.geometry = c
            e1.display['shape'] = 'surface'


class NumericalChirpFinder():
    '''Optimizer to determine optimal chirp by ray-tracing individual rays

    This object passes a ray through the center of a grating with the specified energy and
    diffraction order. It records the position of this center ray. Then, rays are passed through the grating at different positions. For each position, the grating period is optimized numerically, such that these test rays hit the same position as the center ray, when the object is called is called.

    The purpose of wrapping this in an object (as opposed to a simple function) is that certain settings such as energy and order of diffraction are set when the oject is initialized



    Assumes that the focal point (=position of the 0th order) is at the
    origin of the coordinate system.

    Parameters
    ----------
    detector : marxs element
        Mamrx
    colname : string
        Name of column in photon list that the detector ``detector`` writes in
    '''
    uv = [0, 0]

    def __init__(self, detector, order, energy, d=0.0002,
                 colname='detcirc_phi'):
        self.photon = generate_test_photons(1)
        self.detector = detector
        self.energy = energy
        self.order = order
        self.base_d = d
        self.colname = colname

    def set_grat(self, grat):
        self.grat = grat
        self.calc_goal()

    def set_uv(self, uv):
        self.uv = uv
        self.init_photon()

    def calc_goal(self):
        if not hasattr(self.grat, 'original_orderselector'):
            self.grat.original_orderselector = self.grat.order_selector
        self.grat.order_selector = OrderSelector([self.order])
        self.grat._d = self.base_d
        self.set_uv([0., 0.])
        self.run_photon()
        self.goal = self.photon[self.colname][0]

    def posongrat(self):
        pos = h2e(self.grat.geometry['center'] +
                  self.uv[0] * self.grat.geometry['v_y'] +
                  self.uv[1] * self.grat.geometry['v_z'])
        return pos

    def init_photon(self):
        pos = self.posongrat()
        self.pos = e2h(1.1 * pos, 1)
        self.dir = norm_vector(- e2h(pos.reshape(1, 3), 0))
        self.reset_photon()

    def reset_photon(self):
        self.photon['pos'] = self.pos
        self.photon['dir'] = self.dir
        self.photon['probability'] = 1

    def run_photon(self):
        self.reset_photon()
        self.photon = self.grat(self.photon)
        self.photon = self.detector(self.photon)

    def optimize_func(self, d):
        self.grat._d = d * self.base_d
        self.run_photon()
        return np.abs(self.photon[self.colname][0] - self.goal)

    def correction_on_d(self, uarray=np.array([-.999, 0, .999]),
                        varray=np.array([0])):
        corr = np.ones((len(uarray), len(varray)))
        for j, u in enumerate(uarray):
            for k, v in enumerate(varray):
                self.set_uv([u, v])
                corr[j, k] = minimize_scalar(self.optimize_func,
                                             bracket=(.99, 1., 1.01)).x
        return corr

    def __call__(self, grat, *args, **kwargs):
        self.set_grat(grat)
        return self.correction_on_d(*args, **kwargs)


class BendNumericalChirpFinder(NumericalChirpFinder):
    def posongrat(self):
        trans, rot, zoom, shear = decompose44(self.grat.geometry.pos4d)
        p_lim = self.grat.geometry.phi_limits
        p_center = np.mean(p_lim)
        # Not accounting for 2 pi wrap and crazy stuff
        p_half = (p_lim[1] - p_lim[0]) / 2
        pos = h2e(self.grat.geometry.parametric_surface([p_center + self.uv[0] * p_half],
                                                        [self.uv[1] * zoom[2]]))
        return pos[0, 0, :]


def chirp_gratings(gratings, optimizer, d,
                   uarray=np.array([-.999, 0, .999]), varray=np.array([0])):
    '''
    Parameters
    ----------
    gratings : list

    optimizer : callable
    '''
    for grat in gratings:
        corr = optimizer(grat, uarray, varray)
        corr = np.tile(corr[:, 0], (3, 1)).T
        ly = np.linalg.norm(grat.geometry['v_y'])
        lz = np.linalg.norm(grat.geometry['v_z'])
        grat.fitted_d_corr = corr
        grat.spline = RectBivariateSpline(ly * uarray, lz * uarray,
                                          d * corr,
                                          bbox=[-ly, ly, -lz, lz],
                                          kx=2, ky=2)

        def func(self, intercoos):
            return self.spline(intercoos[:, 0], intercoos[:, 1], grid=False)

        # Invoking the descriptor protocol to create a bound method
        # see https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        grat._d = func.__get__(grat)
        grat.order_selector = grat.original_orderselector
