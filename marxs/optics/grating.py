'''Gratings and efficiency files'''

import numpy as np
from astropy.table import Column
from transforms3d import affines

from ..math.pluecker import *
from ..math.utils import norm_vector
from .. import energy2wave
from .base import FlatOpticalElement

def uniform_efficiency_factory(max_order = 3):
    '''Uniform grating efficiency

    This returns a callable that assigns grating orders between ``-max_order``
    and ``+max_order`` (boundaries included) with uniform efficiency.

    Parameters
    ----------
    max_order : int
        Grating orders are chosen between ``-max_order`` and ``+max_order``
        (boundaries included).

    Returns
    -------
    uniform efficiency : callable
        A callable that always returns ``order`` for every photon input.
    '''
    def uniform_efficiency(energy, polarization):
        if np.isscalar(energy):
            return np.random.randint(-max_order, max_order + 1), 1.
        else:
            return np.random.randint(-max_order, max_order + 1, len(energy)), np.ones_like(energy)
    return uniform_efficiency


def constant_order_factory(order = 1):
    '''Select the same grating order for every photon.

    This is useful for testing and to quickly generate large photon numbers in
    a specific region of the detector, for example if the input spectrum has to
    narrow lines close to each other and you want to see if they can be
    resolved in any specific order. In that case, it is just a waste of time to
    simulate photons that diffract all over the detector.

    Parameters
    ----------
    order : int
        The grating order that will be assigned to all photons.

    Returns
    -------
    select_constant_order : callable
        A callable that always returns ``order`` for every photon input.
    '''
    def select_constant_order(energy, polarization):
        '''Always select the same order'''
        return np.ones_like(energy, dtype=int) * order, np.ones_like(energy)

    return select_constant_order


class EfficiencyFile(object):
    def __init__(self, filename, orders):
        dat = np.loadtxt(filename)
        self.energy = dat[:, 0]
        if len(orders) != (dat.shape[1] - 1):
            raise ValueError('orders has len={0}, but data files has {1} order columns.'.format(len(orders), dat.shape[1] - 1))
        self.orders = np.array(orders)
        # Probability to end up in any order
        self.totalprob = np.sum(dat[:, 1:], axis=1)
        # Cumulative probability for orders, normalized to 1.
        self.cumprob = np.cumsum(dat[:, 1:], axis=1) / self.totalprob[:, None]

    def __call__(self, energies, polarization):
        orderind = np.empty(len(energies), dtype=int)
        ind = np.empty(len(energies), dtype=int)
        for i, e in enumerate(energies):
            ind[i] = np.argmin(np.abs(self.energy - e))
            orderind[i] = np.min(np.nonzero(self.cumprob[ind[i]] > np.random.rand()))
        return self.orders[orderind], self.totalprob[ind]



class FlatGrating(FlatOpticalElement):
    '''Flat grating with grooves parallel to the y axis.

    The grating is assumed to be geometrically thin, i.e. all photons enter on
    the face of the grating, not through the sides.

    The sign convention for grating orders is determined by the ``order_sign_convenction``
    attribute. If this is ``None``, the following, somewhat arbitrary convention is chosen:
    Positive grating orders will are displaced along the local :math:`\hat e_z` vector,
    negative orders in the opposite direction. If the grating is rotated by :math:`-\pi`
    the physical situation is the same, but the sign of the grating order will be reversed.
    In this sense, the convention chosen is arbitrarily. However, it has some practical
    advantages: The implementation is fast and all photons passing through the grating
    in the same diffraction order are displaced in the same way. (Contrary to the
    convention in :class:`CATGrating`.)
    If ``order_sign_convention`` is not ``None`` is has to be a callable that accepts the
    photons table as input and returns an a float (e.g. ``+1``) or an array filled with -1
    or +1.

    Parameters
    ----------
    d : float
        grating constant
    order_selector : callable
        A function or callable object that accepts photon energy and
        polarization as input and returns a grating order (integer)
        and a probability (float).
    transmission : bool
        Set to ``True`` for a transmission grating and to ``False`` for a
        reflection grating. *(Default: ``True`` )*

        .. warning::
           Reflection gratings are untested so far!
    '''
    output_columns = ['order', 'grat_y', 'grat_z']
    order_sign_convention = None

    def __init__(self, **kwargs):
        self.order_selector = kwargs.pop('order_selector')
        self.transmission = kwargs.pop('transmission', True)
        self.d = kwargs.pop('d', None)
        if self.d is None:
            raise ValueError('Input parameter "d" (Grating constant) is required.')
        super(FlatGrating, self).__init__(**kwargs)

    def diffract_photons(self, photons):
        '''Vectorized implementation'''
        p = norm_vector(h2e(photons['dir']))
        n = self.geometry['plane'][:3]
        l = h2e(self.geometry['e_y'])
        d = h2e(self.geometry['e_z'])

        wave = energy2wave / photons['energy']
        m, prob = self.order_selector(photons['energy'], photons['polarization'])
        # The idea to calculate the components in the (d,l,n) system separately
        # is taken from MARX
        if self.order_sign_convention is None:
            sign = 1.
        else:
            sign = self.order_sign_convention(photons)
        p_d = np.dot(p, h2e(self.geometry['e_z'])) + sign * m * wave / self.d
        p_l = np.dot(p, h2e(self.geometry['e_y']))
        # The norm for p_n can be derived, but the direction needs to be chosen.
        p_n = 1. - np.sqrt(p_d**2 + p_l**2)
        # Check if the photons have same direction compared to normal before
        direction = np.sign(np.dot(p, n), dtype=np.float)
        if not self.transmission:
            direction *= -1
        dir = e2h(p_d[:, None] * d[None, :] + p_l[:, None] * l[None, :] + (direction * p_n)[:, None] * n[None, :], 0)
        return dir, m, prob

    def process_photons(self, photons, intersect=None, interpos=None, intercoos=None):
        '''
        Other Parameters
        ----------------
        intersect, interpos, intercoos : array (N, 4)
            These parameters are here for performance reasons. In many cases, the
            intersection point between the grating and the rays has been calculated
            by the calling routine to decide which photon is processed by which
            grating and only photons intersecting this grating are passed in.
            The array ``interpos`` contains the intersection points in the global
            coordinate system, ``intercoos`` in the local (y,z) system of the grating.
            If not all three of ``intersect``, ``interpos`` and ``intercoos` are passed in, they are
            calculated here. No checks are done on passed-in values.
        '''
        if (interpos is None) or (intercoos is None) or (intersect is None):
            intersect, interpos, intercoos = self.intersect(photons['dir'], photons['pos'])
        self.add_output_cols(photons)
        if intersect.sum() > 0:
            dir, m, p = self.diffract_photons(photons[intersect])
            photons['pos'][intersect] = interpos[intersect]
            photons['dir'][intersect] = dir
            photons['order'][intersect] = m
            photons['grat_y'][intersect] = intercoos[intersect, 0]
            photons['grat_z'][intersect] = intercoos[intersect, 1]
            photons['probability'][intersect] = photons['probability'][intersect] * p
        return photons

class CATGrating(FlatGrating):
    '''Critical-Angle-Transmission Grating

    CAT gratings are a special case of :class:`FlatGrating` and accept the same arguments.

    They differ from a :class:`FlatGrating` in the sign convention of the
    grating orders: Blazing happens on the side of the negative orders. Obviously, this
    convention is only meaningful if the photons do not arrive perpendicular to the grating.
    '''


    def order_sign_convention(self, photons):
        '''Convention to chose the sign for CAT grating orders

        Blazing happens on the side of the negative orders. Obviously, this
        convention is only meaningful if the photons do not arrive perpendicular to the grating.
        '''
        p = h2e(photons['dir'])
        d = h2e(self.geometry['e_z'])
        dotproduct = np.dot(p, d)
        sign = np.sign(dotproduct)
        sign[sign == 0] = 1
        return sign
