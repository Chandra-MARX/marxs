import numpy as np
from numpy.random import rand, randn

from marxs.missions.mitsnl.catgrating import InterpolateEfficiencyTable, NonParallelCATGrating
from marxs.design.uncertainties import generate_facet_uncertainty as genfacun
from marxs.design.tolerancing import oneormoreelements
from marxs.simulator import SimulationSetupError

__all__ = ['OrderSelectorWavy',
           'OrderSelectorTopHat',
           'wiggle_and_bar_tilt',
           ]

class OrderSelectorWavy(InterpolateEfficiencyTable):
    '''Add a random number to blaze angle before looking up Ralf Table

    In the lab, it seems that the grating bars are not exactly
    perpendicular to the "surface" of the grating. This class adds
    a random number drawn from a Gaussian distribution to the blaze angle
    before looking up the grating efficiency and selecting the order.

    Parameters
    ----------
    wavysigma : float
        Sigma of Gaussian distribution (in radian)
    '''
    def __init__(self, wavysigma, **kwargs):
        self.sigma = wavysigma
        super().__init__(**kwargs)

    def probabilities(self, energies, pol, blaze):
        return super().probabilities(energies, pol, blaze + self.sigma * randn(len(blaze)))


class OrderSelectorTopHat(InterpolateEfficiencyTable):
    '''Add a random number to blaze angle before looking up Ralf Table

    In the lab, it seems that the grating bars are not exactly
    perpendicular to the "surface" of the grating. This class adds
    a random number drawn from a Gaussian distribution to the blaze angle
    before looking up the grating efficiency and selecting the order.

    Parameters
    ----------
    tophatwidth : float
        width of tophat function (in radian)
    '''
    def __init__(self, tophatwidth, **kwargs):
        self.tophatwidth = tophatwidth
        super().__init__(**kwargs)

    def probabilities(self, energies, pol, blaze):
        return super().probabilities(energies, pol, blaze + self.tophatwidth * (rand(len(blaze)) - 0.5))


@oneormoreelements
def wiggle_and_bar_tilt(e, dx=0, dy=0, dz=0, rx=0., ry=0., rz=0., max_bar_tilt=0.):
    '''Move and rotate elements around principal axes accounting for bar tilt.

    This function is made for `~marxs.optics.CATGrating`, and simulates a situation
    where the bar angles are not normal to the grating surface.
    We draw a bar tilt angle for each grating from a given distribution. Then,
    an element is rotated to compensate this bar tilt angle. This represents the a
    correction that happens when the grating is bonded into a frame.
    Additionally, the "normal" random misalignment is applied.

    Parameters
    ----------
    e : `marxs.simulator.Parallel` or list of those elements
        Elements where uncertainties will be set
    dx, dy, dz : float
        accuracy of positioning in x, y, z (in mm) - Gaussian sigma, not FWHM!
    rx, ry, rz : float
        accuracy of positioning. Rotation around x, y, z (in rad) - Gaussian
        sigma, not FWHM!
    max_bar_tilt : float
        Maximum value for bar tilt, drawn from a uniform distribution.
    '''
    tilts = np.random.uniform(low=-max_bar_tilt, high=max_bar_tilt, size=len(e.elements))
    tilt_offsets = np.zeros((len(e.elements), 3))
    tilt_offsets[:, 1] = tilts
    e.elem_uncertainty = genfacun(len(e.elements), [dx, dy, dz], [rx, ry, rz],
                                  rot_offset=-tilt_offsets)
    e.generate_elements()

    # Find the CAT grating membrane in the stack
    # e.g. L1 and L2 are sub-classes, we only want to apply this to main grating
    grats = e.elements_of_class(NonParallelCATGrating, subclass_ok=False)
    if len(grats) != len(tilts):
        raise SimulationSetupError(f'Number of elements of {e} does not match number of NonParallelCATGratings. \n' +
                             'This wiggle function requires one NonParallelCATGrating per element.')

    for elem, t in zip(grats, tilts):
        if not hasattr(elem, 'blaze_center'):
            raise ValueError(f'Object {elem} does not have a blaze_center attribute.')

        elem.blaze_center = t
