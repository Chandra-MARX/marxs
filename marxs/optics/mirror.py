import numpy as np

from transforms3d.axangles import axangle2mat

from .base import FlatOpticalElement
from ..math.pluecker import *
from ..math.utils import norm_vector

class PerfectLens(FlatOpticalElement):
    '''This describes an infinitely large lens that focusses all rays exactly.

    It is computationally cheap and if the details of the mirror are not important
    to the simulation, a thin lens might provide an approximation for
    X-ray focussing. In many cases the ``PerfectLens`` will be combined with some blur
    function (e.g. random scatter by 1 arsec) to achieve a simple approximation to some
    mirror specification.
    '''
    def __init__(self, **kwargs):
        self.focallength = kwargs.pop('focallength')
        super(PerfectLens, self).__init__(**kwargs)

    def process_photons(self, photons):
        # A ray through the center is not broken.
        # So, find out where a central ray would go.
        focuspoints = h2e(self.geometry['center']) + self.focallength * norm_vector(h2e(photons['dir']))
        photons['dir'] = e2h(focuspoints - h2e(photons['pos']), 0)
        return photons

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

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> from marxs import source, optics
    >>> mysource = source.PointSource((30., 30.))
    >>> mypointing = source.FixedPointing(coords=(30., 30.))
    >>> myslit = optics.RectangleAperture(zoom=2)
    >>> lens = optics.ThinLens(focallength=10,zoom=40)

    >>> photons = mysource.generate_photons(11)
    >>> photons = mypointing.process_photons(photons)
    >>> photons = myslit.process_photons(photons)
    >>> photons = lens.process_photons(photons)

    >>> mdet = optics.FlatDetector(pixsize=0.01, position=np.array([-9.6, 0, 0]), zoom=1e5)
    >>> photons = mdet.process_photons(photons)
    >>> fig = plt.plot(photons['det_x'], photons['det_y'], 's')
    '''
    def __init__(self, **kwargs):
        self.focallength = kwargs.pop('focallength')
        super(ThinLens, self).__init__(**kwargs)

    def process_photon(self, dir, pos, energy, polerization):
        intersect, h_intersect, loc_inter = self.intersect(dir, pos)
        distance = distance_point_point(h_intersect,
                                        self.geometry['center'][np.newaxis, :])
        if distance == 0.:
            # No change of direction for rays through the origin.
            # Need to special case this, because rotation axis is not defined
            # in this case.
            new_ray_dir = h2e(dir)
        else:
            delta_angle = distance / self.focallength
            e_rotation_axis = np.cross(dir[:3], (h_intersect - self.geometry['center'])[:3])
            # This is the first step that cannot be done on a stack of photons
            # Could have a for "i in photons", but might come up with better way
            rot = axangle2mat(e_rotation_axis, delta_angle)
            new_ray_dir = np.dot(rot, dir[:3])
        return e2h(new_ray_dir, 0), h_intersect, energy, polerization, 1.
