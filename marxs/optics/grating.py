import numpy as np
from astropy.table import Column
from transforms3d import affines

from ..math.pluecker import *
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
            return np.random.randint(max_order - 1, max_order + 1)
        else:
            # scalar input
            return np.random.randint(max_order - 1, max_order + 1, len(energy))
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
        return np.ones_like(energy, dtype=int) * order

    return select_constant_order


class FlatGrating(FlatOpticalElement):
    '''Flat grating with grooves parallel to the y axis.

    The grating is assumed to be geometrically thin, i.e. all photons enter on
    the face of the grating, not through the sides.

    Parameters
    ----------
    d : float
        grating constant
    order_selector : callable
        A function or callable object that accepts photon energy and
        polarization as input and returns a grating order (integer).
    transmission : bool
        Set to ``True`` for a transmission grating and to ``False`` for a
        reflection grating. *(Default: ``True``)*

        .. warning::
           Reflection gratings are untested so far!

        .. todo::
           Check reflection gratings.
    '''
    output_columns = ['order']

    def __init__(self, **kwargs):
        self.order_selector = kwargs.pop('order_selector')
        self.transmission = kwargs.pop('transmission', True)

        self.d = kwargs.pop('d', None)
        if self.d is None:
            raise ValueError('Input parameter "d" (Grating constant) is required.')
        super(FlatGrating, self).__init__(**kwargs)

    def diffract_photons(self, photons):
        '''Vectorized implementation'''
        p = h2e(photons['dir'])
        # Check if p is normalized
        length2 = np.sum(p*p, axis=-1)
        if not np.allclose(length2, 1.):
            p = p / np.sqrt(length2)[:, None]
        n = self.geometry['plane'][:3]
        l = h2e(self.geometry['e_y'])
        d = h2e(self.geometry['e_z'])

        wave = energy2wave / photons['energy']
        m = self.order_selector(photons['energy'], photons['polarization'])
        # The idea to calculate the components in the (d,l,n) system separately
        # is taken from MARX
        p_d = np.dot(p, h2e(self.geometry['e_z'])) + m * wave / self.d
        p_l = np.dot(p, h2e(self.geometry['e_y']))
        # The norm for p_n can be derived, but the direction need to be chosen.
        p_n = 1. - np.sqrt(p_d**2 + p_l**2)
        # Check if the photons have same direction compared to normal before
        direction = np.sign(np.dot(p, n), dtype=np.float)
        if not self.transmission:
            direction *= -1
        dir = e2h(p_d[:, None] * d[None, :] + p_l[:, None] * l[None, :] + (direction * p_n)[:, None] * n[None, :], 0)
        return dir, m

    def process_photons(self, photons, interpos=None):
        '''
        Additional Parameters
        ---------------------
        interpos : array (N, 4)
            This parameter is here for performance reasons. In many cases, the
            intersection point between the grating and the rays has been calculated
            by the calling routine to decide which photon is processed by which
            grating and only photons intersecting this grating are passed in.
            If ``interpos`` is  not passed in, they are
            calculated here. No checks are done on passed-in values.
        '''
        if (interpos is None):
            intersect, interpos, intercoos = self.intersect(photons['dir'], photons['pos'])
        else:
            intersect = np.ones(len(phtons), dtype=bool)
        self.add_output_cols(photons)
        dir, m = self.diffract_photons(photons[intersect])
        photons['pos'][intersect] = interpos
        photons['dir'][intersect] = dir
        photons['order'][intersect] = m
        return photons
