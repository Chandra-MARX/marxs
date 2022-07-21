# Licensed under GPL version 3 - see LICENSE.rst
'''Gratings made by the MKI `Space Nanotechnology Laboratory`_

The MIT Kavli Institute for Astrophysics and Space Research
`Space Nanotechnology Laboratory`_ produces elements for astronomical
instruments in space. One particular field of research are critical
angle transmission (CAT) gratings, see e.g. `Heilmann et al. (2015)`_

.. _Space Nanotechnology Laboratory: http://snl.mit.edu/
.. _Heilmann et al. (2015): http://dx.doi.org/10.1117/12.2188525
'''
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table

from marxs.optics import (CATGrating,
                          OrderSelector, FlatStack,
                          FlatOpticalElement)
from marxs.math.utils import norm_vector
from marxs.optics.scatter import RandomGaussianScatter
from marxs.utils import tablerows_to_2d


__all__ = ['l1transtab', 'l1_order_selector',
           'l1_dims', 'l2_dims',
           'qualityfactor', 'd',
           'InterpolateEfficiencyTable',
           'QualityFactor',
           'L1',
           'L2Abs',
           'L2Diffraction',
           'CATL1L2Stack',
           'NonParallelCATGrating',
           'catsupportbars',
           ]

d = 0.0002
'''Spacing of grating bars'''

l1transtab = Table.read(get_pkg_data_filename('data/SiTransmission.csv'), format='ascii.ecsv')
'''Transmission through 1 mu of Si'''

l1_order_selector = OrderSelector(orderlist=np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
                                  p=np.array([0.006, 0.0135, 0.022, 0.028, 0.861, 0.028, 0.022, 0.0135, 0.006]))
'''Simple order selector for diffraction on L1.

The numbers here are calculated for the 2018 Arcus gratings assuming the L1 structure
is independent from the grating membrane itself (which is not true, but a valid first
approximation.)
'''

l1_dims = {'bardepth': 0.004 * u.mm, 'period': 0.005 * u.mm, 'barwidth': 0.0009 * u.mm}
'''Dimensions of L1 support bars running perpendicular to the grating bars'''

l2_dims = {'bardepth': 0.5 * u.mm, 'period': 0.9622504 * u.mm, 'barwidth': 0.0962250 * u.mm}
'''Dimensions of hexagonal L2 support'''

qualityfactor = {'d': 200. * u.um, 'sigma': 1.75 * u.um}
'''Scaling of grating efficiencies, parameterized as a Debye-Waller factor'''


class InterpolateEfficiencyTable(object):
    '''Order Selector for MARXS using a specific kind of data table.

    Ralf Heilmann from the SNL typically writes simulated grating efficiencies
    into Excel tables. Since Excel is hard to read in Python and not very
    suited to version control with git, those tables are converted to csv files
    of a certain format.
    A short summary of this format is given here, to help reading the code.
    The table contains data in 3-dimensional (wave, n_theta, order) space,
    flattened into a 2d table.

    - Row 1 + 2: Column labels. Not used here.
    - Column A: wavelength in nm.
    - Column B: blaze angle in deg.
    - Rest: data

    For each wavelength there are multiple blaze angles listed, so Column A
    contains
    many duplicates and looks like this: [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3, ...].
    Column B repeats like this: [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3, ...].

    Because the wave, theta grid is regular, this class can use the
    `scipy.interpolate.RectBivariateSpline` for interpolation in each 2-d slice
    (``order`` is an integer and not interpolated).

    Parameters
    ----------
    tab : `astropy.table.Table`
        Table as read in. Useful to access units or other meta data.
    k : int
        Degree of spline. See `scipy.interpolate.RectBivariateSpline`.
    '''

    def __init__(self, tab, k=1):
        wave, theta, orders = tablerows_to_2d(tab)

        theta = theta.to(u.rad)
        # Order is int, we will never interpolate about order,
        self.orders = np.array([int(n) for n in tab.colnames[2:]])
        self.interpolators = [RectBivariateSpline(wave, theta, d, kx=k, ky=k) for d in orders]

    def probabilities(self, energies, pol, blaze):
        '''Obtain the probabilities for photons to go into a particular order.

        This has the same parameters as the ``__call__`` method, but it returns
        the raw probabilities, while ``__call__`` will draw from these
        probabilities and assign an order and a total survival probability to
        each photon.

        Parameters
        ----------
        energies : np.array
            Energy for each photons
        pol : np.array
            Polarization for each photon (not used in this class)
        blaze : np.array
            Blaze angle for each photon

        Returns
        -------
        orders : np.array
            Array of orders
        interpprobs : np.array
            This array contains a probability array for each photon to reach a
            particular order
        '''
        # convert energy in keV to wavelength in nm
        # (nm is the unit of the input table)
        wave = (energies * u.keV).to(u.nm, equivalencies=u.spectral()).value
        interpprobs = np.empty((len(self.orders), len(energies)))
        for i, interp in enumerate(self.interpolators):
            interpprobs[i, :] = interp.ev(wave, blaze)
        return self.orders, interpprobs

    def __call__(self, energies, pol, blaze):
        orders, interpprobs = self.probabilities(energies, pol, blaze)
        totalprob = np.sum(interpprobs, axis=0)
        # Cumulative probability for orders, normalized to 1.
        cumprob = np.cumsum(interpprobs, axis=0) / totalprob
        ind_orders = np.argmax(cumprob > np.random.rand(len(energies)), axis=0)

        return orders[ind_orders], totalprob


class QualityFactor(FlatOpticalElement):
    '''Scale probabilities of theoretical curves to measured values.

    All gratings look better in theory than in practice. This grating quality
    factor scales the calculated diffraction probabilities to the observed
    performance.
    '''

    def __init__(self, qualityfactor=qualityfactor, **kwargs):
        self.factor = np.exp(- (2 * np.pi * qualityfactor['sigma'] /
                                qualityfactor['d'])**2)
        super().__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        return {'probability': self.factor**(photons['order'][intersect]**2)}
        return photons


def check_lx_dims(lx_dims):
    '''Check that dimensions in l1_dims or l2_dims make sense'''
    if not (lx_dims['barwidth'] < lx_dims['period']):
        raise ValueError('Period of grating must be larger than bar width.')


class L1(CATGrating):
    '''A CAT grating representing only the L1 structure

    This is treated independently of the CAT grating layer itself
    although the two gratings are not really in the far-field limit.
    CAT gratings of this class determine (statistically) if a photon
    passes through the grating bars or the L1 support.
    The L1 support is simplified as solid Si layer of 4 mu thickness.

    Parameters
    ----------
    l1_dims : dict
    '''
    blaze_name = 'blaze_L1'
    order_name = 'order_L1'

    def __init__(self, l1_dims=l1_dims, **kwargs):
        check_lx_dims(l1_dims)
        self.openfraction = 1 - l1_dims['barwidth'] / l1_dims['period']
        energy = l1transtab['energy'].to(u.keV, equivalencies=u.spectral())
        trans = np.exp(np.log(l1transtab['transmission']) * l1_dims['bardepth'] / (1 * u.micrometer))
        self.transfunc = interp1d(energy, trans)
        kwargs['d'] = l1_dims['period'].to(u.mm).value
        super().__init__(**kwargs)


    def specific_process_photons(self, photons, intersect,
                                 interpos, intercoos):
        catresult = super().specific_process_photons(photons, intersect, interpos, intercoos)

        # Now select which photons go through the L1 support and
        # set the numbers appropriately.
        # It is easier to have the diffraction calculated for all photons
        # and then re-set numbers for a small fraction here.
        # That, way, I don't have to duplicate the blaze calculation and no
        # crazy tricks are necessary to keep the indices correct.
        l1 = np.random.rand(intersect.sum()) > self.openfraction
        ind = intersect.nonzero()[0][l1]
        catresult['dir'][l1] = photons['dir'].data[ind, :]
        catresult['polarization'][l1] = photons['polarization'].data[ind, :]
        catresult['order_L1'][l1] = 0
        catresult['probability'][l1] = self.transfunc(photons['energy'][ind])
        return catresult


class L2Abs(FlatOpticalElement):
    '''L2 absorption and shadowing

    Some photons may pass through the CAT grating membrane and L1
    support, but are absorbed by the L2 sidewalls. We treat this
    statistically by reducing the overall probability.  This
    implementation ignores the effect that photons might scatter on
    the L2 sidewall surface (those would be scattered away from the
    CCDs anyway for most layouts).

    Note that this does not read the L2 from a file, but calculates it
    directly from the dimensions.

    '''
    def __init__(self, l2_dims=l2_dims, **kwargs):
        check_lx_dims(l2_dims)
        self.bardepth = l2_dims['bardepth']
        self.period = l2_dims['period']
        self.barwidth = l2_dims['barwidth']
        super().__init__(**kwargs)
        self.innerfree = self.period - self.barwidth

    def specific_process_photons(self, photons, intersect,
                                 interpos, intercoos):

        p3 = norm_vector(photons['dir'].data[intersect])
        ex, ey, en = self.geometry.get_local_euklid_bases(intercoos[intersect, :])
        angle = np.arccos(np.abs(np.einsum("ij,ij->i", p3, en)))

        # fractional area NOT covered by the hexagon structure
        openfraction = (self.innerfree / self.period)**2
        # fractional area shadowed by inclined hexagon structure
        shadowarea = (self.bardepth * self.innerfree * np.sin(angle))
        totalarea = self.period**2 / 2 * np.sqrt(3)
        shadowfraction = shadowarea / totalarea

        return {'probability': openfraction - shadowfraction}


class L2Diffraction(RandomGaussianScatter):
    '''Very simple approximation of L2 diffraction effects.

    L2 is a hexagonal pattern, but at such a large spacing, that diffraction
    at higher orders can be safely neglected. The only thing that does
    happen is a slight broadening due to the single-slit function, and again,
    only the core of that matters. So, we simply approximate this here with
    simple Gaussian Scattering using the radius of the Airy disk as estimate
    for the broadening sigma.
    '''
    scattername = 'L2Diffraction'
    def __init__(self, l2_dims=l2_dims, **kwargs):
        check_lx_dims(l2_dims)
        self.innerfree = l2_dims['period'] - l2_dims['barwidth']
        super().__init__(**kwargs)

    def scatter(self, photons, intersect, interpos, intercoos):
        wave = (photons['energy'].data[intersect] * u.keV).to(u.mm, equivalencies=u.spectral())
        # 1.22 from Airy disk formula https://en.wikipedia.org/wiki/Airy_disk
        # 0.4 is approx factor between sigma and r (first minimum)
        sigma = 1.22 * 0.4 * np.arcsin(wave / self.innerfree)
        return np.random.normal(size=intersect.sum()) * sigma


class NonParallelCATGrating(CATGrating):
    '''CAT Grating where the angle of the reflective wall changes.

    This element represents a CAT grating where not all grating bar walls
    are perpendicular to the surface of the grating. This is only
    true for a ray through the center. The angle changes linearly with
    the distance to the center in the dispersion direction.
    Each grating bar has a fixed angle, i.e. no change of the direction
    happens along the grating bars (perpendicular to the dispersion direction).

    Parameters
    ----------
    d_blaze_mm : float
        Change in direction of the reflecting grating bar sidewall, which
        directly translates to a change in blaze angle [rad / mm].
    blaze_center : float
        Blaze angle at center of grating, ``0`` means grating bars are
        perpendicular to element surface. [rad]
    '''
    def __init__(self, **kwargs):
        self.d_blaze_mm = kwargs.pop('d_blaze_mm', 0)
        self.blaze_center = kwargs.pop('blaze_center', 0)
        super().__init__(**kwargs)

    def blaze_angle_modifier(self, intercoos):
        '''
        Parameters
        ----------
        intercoos : np.array
            intercoos coordinates for photons interacting with optical element
        '''
        return self.blaze_center + intercoos[:, 0] * self.d_blaze_mm


def catsupportbars(photons):
    '''Metal structure that holds grating facets will absorb all photons
    that do not pass through a grating facet.

    We might want to call this L3 support ;-)
    '''
    if 'facet' in photons.colnames:
        photons['probability'][photons['facet'] < 0] = 0.
    else:
        photons['probability'] = 0.
    return photons


class CATL1L2Stack(FlatStack):
    '''SNL fabricated CAT grating

    This element combines all parts of a CAT grating into a single object.
    These include the grating membrane and the absorption and diffraction due
    to the L1 and L2 support.
    Approximations are done for all those elements, see the individial classes
    for more details. Except for `order_selector` all other parameters are set with
    defaults defined in module level variables.

    Parameters
    ----------
    order_selector : `marxs.optics.OrderSelector`
        Order selector for the grating membrane
    groove_angle : float
        Goove angle of grating bars (in rad). Default: 0
    l1_order_selector : `marxs.optics.OrderSelector`
        Order selector for L1 dispersion (cross-dispersion direction for grating)
    qualityfactor : dict
        Parameterization of grating quality scaling factor. See model level variable
        for format.
    l1_dims : dict
        Dimensions of L1 support. See module level variable for format.
    l2_dims : dict
        Dimensions of L2 support. See module level variable for format.
    '''
    def __init__(self, l1_dims=l1_dims, l2_dims=l2_dims,
                 l1_order_selector=l1_order_selector,
                 qualityfactor=qualityfactor,
                 **kwargs):
        kwargs['elements'] = [NonParallelCATGrating,
                              QualityFactor,
                              L1,
                              L2Abs,
                              L2Diffraction,
                          ]
        groove_angle = kwargs.pop('groove_angle', 0.)
        kwargs['keywords'] = [{'order_selector': kwargs.pop('order_selector'),
                               'd': kwargs.pop('d', d),
                               'groove_angle': groove_angle},
                              {'qualityfactor': qualityfactor},
                              {'l1_dims': l1_dims,
                               'order_selector': l1_order_selector,
                               'groove_angle': np.pi / 2. + groove_angle},
                               {'l2_dims': l2_dims},
                               {'l2_dims': l2_dims}
                          ]
        super().__init__(**kwargs)
