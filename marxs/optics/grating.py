'''Gratings and efficiency files'''
import math
import numpy as np

from ..math.pluecker import h2e, e2h
from ..math.utils import norm_vector
from ..math.polarization import parallel_transport
from .. import energy2wave
from .base import FlatOpticalElement

class OrderSelector(object):
    '''Select from a list of order number independent of energy.

    Parameters
    ----------
    orderlist : array
        This are the order numbers to chose from. They must be integers.
    p : None or array
        Probability for a photon to end up in each order. The sum of all probabilities
        can be smaller than 1, if a certain fraction of photons is absorbed by
        the grating. If this is ``None`` equal probability is assigned to all orders.

    Examples
    --------
    Two common use cases for testing are a grating where every photon gets
    diffracted into the same order or where all photons get distributed with
    equal probability into a set of orders.

    >>> import numpy as np
    >>> from marxs.optics import FlatGrating, OrderSelector
    >>> singleordergrating = FlatGrating(d=1e-4, order_selector=OrderSelector([3]))
    >>> setofordersgrating = FlatGrating(d=1e-4, order_selector=OrderSelector(np.arange(-3, 4)))
    '''
    def __init__(selv, orderlist, p=None):
        selv.orderlist = orderlist
        if p is None:
            selv.p = np.ones(len(orderlist)) / len(orderlist)
        else:
            p = np.asanyarray(p)
            if len(p) != len(orderlist):
                raise ValueError('Number of elements in orderlist and probabilities does not match.')
            if p.sum() > 1.:
                raise ValueError('Sum of all probabilities must be <= 1.')
            if np.any(p < 0):
                raise ValueError('Probabilities cannot be negative')
            selv.p = p

    def __call__(selv, energy, *args):
        p_sum = selv.p.sum()
        p = selv.p / p_sum
        if np.isscalar(energy):
            return np.random.choice(selv.orderlist, p=p), selv.p.sum()
        else:
            return np.random.choice(selv.orderlist, size=len(energy), p=p), p_sum * np.ones_like(energy)


class EfficiencyFile(object):
    '''Select grating order from a probability distribution in a data file.

    The file format supported by this class is as follows:
    The first colum contains energy values in keV, all remaining columns have the probability
    that a photons with this energy is diffracted into the respective order. The probabilities
    for each order do not have to add up to 1.

    Parameters
    ----------
    filename : string
        Path to the efficiency file.
    orders : list
        List of orders in the file. Must match the number of columns with probabilities.
    '''
    def __init__(selv, filename, orders):
        dat = np.loadtxt(filename)
        selv.energy = dat[:, 0]
        if len(orders) != (dat.shape[1] - 1):
            raise ValueError('orders has len={0}, but data files has {1} order columns.'.format(len(orders), dat.shape[1] - 1))
        selv.orders = np.array(orders)
        selv.prob = dat[:, 1:]
        # Probability to end up in any order
        selv.totalprob = np.sum(selv.prob, axis=1)
        # Cumulative probability for orders, normalized to 1.
        selv.cumprob = np.cumsum(selv.prob, axis=1) / selv.totalprob[:, None]

    def __call__(selv, energies, *args):
        orderind = np.empty(len(energies), dtype=int)
        ind = np.empty(len(energies), dtype=int)
        for i, e in enumerate(energies):
            ind[i] = np.argmin(np.abs(selv.energy - e))
            orderind[i] = np.min(np.nonzero(selv.cumprob[ind[i]] > np.random.rand()))
        return selv.orders[orderind], selv.totalprob[ind]



class FlatGrating(FlatOpticalElement):
    '''Flat grating

    The grating is assumed to be geometrically thin, i.e. all photons enter on
    the face of the grating, not through the sides.

    Parameters
    ----------
    d : float or callable
        grating constant in mm. If ``d`` is callable, then it will be called as
        ``d(intercoos)``, where ``intercoos`` is a (N, 2) array holding the
        positions where photons hit the gratings in the local coordinate system.
        This can be used to simulate manufacturing uncertainties or intentional
        grating period variation. The callable has to return a vector of lengths N
        that contains the grating constant for each photon in mm.
    order_selector : callable
        A function or callable object that accepts arrays of photon energy, polarization
        and the blaze angle
        as input and returns arrays for grating order (integer)
        and probability (float). The probabiliy expresses the chance that the photon passes
        the grating and is not absorbed, e.g. if the probability that a photon at energy E ends
        up in order=[-2, -1, 0, 1, 2] is [0, 0, .5, .3, .0] , then the returned probability for
        all photons should be 0.8.
    transmission : bool
        Set to ``True`` for a transmission grating and to ``False`` for a
        reflection grating. (*Default*: ``True`` )
    groove_angle : float
        Angle between the local z axis and the direction of the grooves in radian.
        (*Default*: ``0.``)

    .. warning::
       Reflection gratings are untested so far!
    '''

    loc_coos_name = ['grat_y', 'grat_z']
    '''name for output columns that contain the interaction point in local coordinates.'''

    def order_sign_convention(selv, p):
        '''Set sign convention for grating orders.

        This sets the following, somewhat arbitrary convention:
        Positive grating orders will are displaced along the local :math:`\hat e_z` vector,
        negative orders in the opposite direction. If the grating is rotated by :math:`-\pi`
        the physical situation is the same, but the sign of the grating order will be reversed.
        In this sense, the convention chosen is arbitrarily. However, it has some practical
        advantages: The implementation is fast and all photons passing through the grating
        in the same diffraction order are displaced in the same way. (Contrary to the
        convention in :class:`CATGrating`.)
        ``order_sign_convention`` has to be a callable that accepts an array
        of eukledian direction vectors as input and returns ``+1``, ``-1``,
        or an array filled with ``-1`` or ``+1``.

        Parameters
        ----------
        p : np.array
            Array of Eucleadian direction vectors
        '''
        # -1 because n, l, d should be right-handed coordinate system
        # while n = e_x, l = e_x, and d = e_y would be left-handed.
        return -1

    def __init__(selv, **kwargs):
        selv.order_selector = kwargs.pop('order_selector')
        selv.transmission = kwargs.pop('transmission', True)
        if 'd' not in kwargs:
            raise ValueError('Input parameter "d" (Grating constant) is required.')
        selv._d = kwargs.pop('d')
        groove = kwargs.pop('groove_angle', 0.)

        super(FlatGrating, selv).__init__(**kwargs)

        selv._geometry['e_groove'] = np.array([0., math.sin(-groove), math.cos(-groove), 0.])
        selv._geometry['e_perp_groove'] = np.array([0., math.cos(-groove), -math.sin(-groove), 0.])

    def d(selv, intercoos):
        '''Method that returns the grating constant at given positions.

        For a grating with constant grating constant, this will just return the
        number input as ``d`` when the element was initialized. For gratings
        where the grating constant varies with position on the facet, this
        calculates the appropriate number for every position.
        '''
        if not callable(selv._d):
            return selv._d
        else:
            return selv._d(intercoos)

    def diffract_photons(selv, photons, intersect, interpos, intercoos):
        '''Vectorized implementation'''
        p = norm_vector(h2e(photons['dir'].data[intersect]))
        n = selv.geometry('plane')[:3]
        l = h2e(selv.geometry('e_groove'))
        # Minus sign here because we want n, l, d to be a right-handed coordinate system
        d = -h2e(selv.geometry('e_perp_groove'))

        wave = energy2wave / photons['energy'].data[intersect]
        # calculate angle between normal and (ray projected in plane perpendicular to groove)
        # -> this is the blaze angle
        p_perp_to_grooves = norm_vector(p - np.dot(p, l)[:, np.newaxis] * l)
        # Use abs here so that blaze angle is always in 0..pi/2
        # independent of the relative orientation of p and n.
        blazeangle = np.arccos(np.abs(np.dot(p_perp_to_grooves, n)))
        m, prob = selv.order_selector(photons['energy'].data[intersect],
                                      photons['polarization'].data[intersect],
                                      blazeangle)

        # The idea to calculate the components in the (d,l,n) system separately
        # is taken from MARX
        sign = selv.order_sign_convention(p)
        p_d = np.dot(p, d) + sign * m * wave / selv.d(intercoos[intersect, :])
        p_l = np.dot(p, l)
        # The norm for p_n can be derived, but the direction needs to be chosen.
        p_n = np.sqrt(1. - p_d**2 - p_l**2)
        # Check if the photons have same direction compared to normal before
        direction = np.sign(np.dot(p, n), dtype=np.float)
        if not selv.transmission:
            direction *= -1
        dir = e2h(p_d[:, None] * d[None, :] + p_l[:, None] * l[None, :] + (direction * p_n)[:, None] * n[None, :], 0)
        return dir, m, prob, blazeangle

    def specific_process_photons(selv, photons, intersect, interpos, intercoos):

        dir, m, p, blaze = selv.diffract_photons(photons, intersect, interpos, intercoos)
        pol = parallel_transport(photons['dir'].data[intersect, :], dir,
                                 photons['polarization'].data[intersect, :])
        return {'dir': dir, 'probability': p, 'order': m, 'blaze': blaze,
                'polarization': pol}

class CATGrating(FlatGrating):
    '''Critical-Angle-Transmission Grating

    CAT gratings are a special case of :class:`FlatGrating` and accept the same arguments.

    They differ from a :class:`FlatGrating` in the sign convention of the
    grating orders: Blazing happens on the side of the negative orders. Obviously, this
    convention is only meaningful if the photons do not arrive perpendicular to the grating.
    '''


    def order_sign_convention(selv, p):
        '''Convention to chose the sign for CAT grating orders

        Blazing happens on the side of the negative orders. Obviously, this
        convention is only meaningful if the photons do not arrive perpendicular to the grating.
        '''
        # Minus sign here because we want n, l, d to be a right-handed coordinate system
        d = -h2e(selv.geometry('e_perp_groove'))
        dotproduct = np.dot(p, d)
        sign = np.sign(dotproduct)
        sign[sign == 0] = 1
        return sign
