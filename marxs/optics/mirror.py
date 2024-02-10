# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from transforms3d.axangles import axangle2mat

from .base import FlatOpticalElement
from ..math.utils import h2e, e2h, distance_point_point, norm_vector
from ..math.polarization import parallel_transport


class PerfectLens(FlatOpticalElement):
    """This describes an infinitely large lens that focusses all rays exactly.

    This lens is just perfect, it has no astigmatism, no absorption, etc.  It
    is computationally cheap and if the details of the mirror are not important
    to the simulation, a perfect lens might provide an approximation for X-ray
    focussing. In many cases the ``PerfectLens`` will be combined with some
    blur function (e.g. random scatter by 1 arcsec) to achieve a simple
    approximation to some mirror specification.

    Parameters
    ----------
    focal_length : float
        The focal length of the lens in mm.
    d_center_optical_axis : float
        Distance between the optical axis and the center of this element in mm. The optical axis is
        located in -z direction, and the normal pos4d keywords should be used to give the size,
        location, and rotation of this element.
    reflectivity_interpolator : callable
        Callable that accepts energy and reflectivity angle and `grid` as input and returns the probability
        of single reflection. The calling signature is written to match a
        `scipy.interpolate.RectBivariateSpline`; ``grid`` is set to False to
        evaluate on a list of points.
        Default is to always return 1.
    """

    display = {'color': (0., 0.5, 0.),
               'opacity': 0.5,
               'shape': 'box'
    }

    loc_coos_name = ['mirror_x', 'mirror_y']

    def __init__(self, **kwargs):
        self.focallength = kwargs.pop('focallength')
        self.d_center_optax = kwargs.pop("d_center_optical_axis", 0)
        self.reflectivity_interpolator = kwargs.pop(
            "reflectivity_interpolator", lambda pos, ang, grid: np.ones_like(pos)
        )

        super().__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # A ray through the center is not broken.
        # So, find out where a central ray would go.
        p_opt_axis = (
            self.geometry["center"] - self.d_center_optax * self.geometry["e_z"]
        )
        focuspoints = h2e(p_opt_axis) + self.focallength * norm_vector(
            h2e(photons["dir"][intersect])
        )
        dir = norm_vector(e2h(focuspoints - h2e(interpos[intersect]), 0))
        pol = parallel_transport(
            photons["dir"].data[intersect, :],
            dir,
            photons["polarization"].data[intersect, :],
        )
        angle = np.arccos(
            np.abs(
                np.einsum(
                    "ij,ij->i", h2e(dir), norm_vector(h2e(photons["dir"][intersect]))
                )
            )
        )
        return {
            "dir": dir,
            "polarization": pol,
            "probability": self.reflectivity_interpolator(
                photons["energy"][intersect], angle / 4, grid=False
            )
            ** 2,
        }


class ThinLens(FlatOpticalElement):
    '''Focus rays with the thin lens approximation

    This implements the so called "thin lens" approximation, see
    https://en.wikipedia.org/wiki/Thin_lens for details.

    This class represents a lens that is (except for the thin lens
    approximation) perfect: There is no absorption and no wavelength dependence
    of the refraction.  There is no physical material that allows to
    manufacture an X-ray lens in the same shape as traditional glass lenses,
    but this class can be used as an "effective" model. It is computationally
    cheap and if the details of the mirror are not important to the simulation,
    a thin lens might provide an approximation for X-ray focussing.

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from marxs import source, optics
    >>> mysource = source.PointSource(coords=SkyCoord(30., 30., unit="deg"))
    >>> mypointing = source.FixedPointing(coords=SkyCoord(30., 30., unit='deg'))
    >>> myslit = optics.RectangleAperture(zoom=2)
    >>> lens = optics.ThinLens(focallength=10,zoom=40)

    >>> photons = mysource.generate_photons(11 * u.s)
    >>> photons = mypointing(photons)
    >>> photons = myslit(photons)
    >>> photons = lens(photons)

    >>> mdet = optics.FlatDetector(pixsize=0.01, position=np.array([-9.6, 0, 0]), zoom=1e5)
    >>> photons = mdet(photons)
    >>> fig = plt.plot(photons['det_x'], photons['det_y'], 's')

    '''

    display = {'color': (0., 0.5, 0.),
               'opacity': 0.5,
               }

    def __init__(self, **kwargs):
        self.focallength = kwargs.pop('focallength')
        super().__init__(**kwargs)

    def process_photon(self, dir, pos, energy, polarization):
        intersect, h_intersect, loc_inter = self.geometry.intersect(dir, pos)
        distance = distance_point_point(h_intersect,
                                        self.geometry['center'][np.newaxis, :])
        if distance == 0.:
            # No change of direction for rays through the origin.
            # Need to special case this, because rotation axis is not defined
            # in this case.
            new_ray_dir = h2e(dir)
        else:
            delta_angle = -distance / self.focallength
            e_rotation_axis = np.cross(dir[:3], (h_intersect - self.geometry['center'])[:3])
            # This is the first step that cannot be done on a stack of photons
            # Could have a for "i in photons", but might come up with better way
            rot = axangle2mat(e_rotation_axis, delta_angle)
            new_ray_dir = np.dot(rot, dir[:3])
            polarization = e2h(np.dot(rot, polarization[:3]), 0)
        return e2h(new_ray_dir, 0), h_intersect, energy, polarization, 1.
