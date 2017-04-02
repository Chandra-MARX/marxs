# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from transforms3d.axangles import axangle2mat

from .base import FlatOpticalElement
from ..math.utils import h2e, e2h, distance_point_point, norm_vector
from ..math.polarization import parallel_transport

class PerfectLens(FlatOpticalElement):
    '''This describes an infinitely large lens that focusses all rays exactly.

    This lens is just perfect, it has no astigmatism, no absorption, etc.
    It is computationally cheap and if the details of the mirror are not important
    to the simulation, a perfect lens might provide an approximation for
    X-ray focussing. In many cases the ``PerfectLens`` will be combined with some blur
    function (e.g. random scatter by 1 arsec) to achieve a simple approximation to some
    mirror specification.
    '''

    display = {'color': (0., 0.5, 0.),
               'opacity': 0.5,
    }

    loc_coos_name = ['mirror_x', 'mirror_y']

    def __init__(self, **kwargs):
        self.focallength = kwargs.pop('focallength')
        super(PerfectLens, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # A ray through the center is not broken.
        # So, find out where a central ray would go.
        focuspoints = h2e(self.geometry('center')) + self.focallength * norm_vector(h2e(photons['dir'][intersect]))
        dir = e2h(focuspoints - h2e(interpos[intersect]), 0)
        pol = parallel_transport(photons['dir'].data[intersect, :], dir,
                                 photons['polarization'].data[intersect, :])
        return {'dir': dir, 'polarization': pol}

class ThinLens(FlatOpticalElement):
    '''Focus rays with the thin lens approximation

    This implements the so called "thin lens" approximation, see
    https://en.wikipedia.org/wiki/Thin_lens for details.

    This class represents a lens that is (except for the thin lens approximation)
    perfect: There is no absorption and no wavelength dependence of the refraction.
    There is no physical material that allows to manufacture an X-ray lens in the same
    shape as traditional glass lenses, but this class can  be used as an "effective"
    model. It is computationally cheap and if the details of the mirror are not important
    to the simulation, a thin lens might provide an approximation for
    X-ray focussing.

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> from astropy.coordinates import SkyCoord
    >>> from marxs import source, optics
    >>> mysource = source.PointSource(coords=SkyCoord(30., 30., unit="deg"))
    >>> mypointing = source.FixedPointing(coords=SkyCoord(30., 30., unit='deg'))
    >>> myslit = optics.RectangleAperture(zoom=2)
    >>> lens = optics.ThinLens(focallength=10,zoom=40)

    >>> photons = mysource.generate_photons(11)
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
        super(ThinLens, self).__init__(**kwargs)

    def process_photon(self, dir, pos, energy, polarization):
        intersect, h_intersect, loc_inter = self.intersect(dir, pos)
        distance = distance_point_point(h_intersect,
                                        self.geometry('center')[np.newaxis, :])
        if distance == 0.:
            # No change of direction for rays through the origin.
            # Need to special case this, because rotation axis is not defined
            # in this case.
            new_ray_dir = h2e(dir)
        else:
            delta_angle = distance / self.focallength
            e_rotation_axis = np.cross(dir[:3], (h_intersect - self.geometry('center'))[:3])
            # This is the first step that cannot be done on a stack of photons
            # Could have a for "i in photons", but might come up with better way
            rot = axangle2mat(e_rotation_axis, delta_angle)
            new_ray_dir = np.dot(rot, dir[:3])
            polarization = e2h(np.dot(rot, polarization[:3]), 0)
        return e2h(new_ray_dir, 0), h_intersect, energy, polarization, 1.
