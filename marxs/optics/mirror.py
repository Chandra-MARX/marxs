import numpy as np

from transforms3d.axangles import axangle2mat

from .base import FlatOpticalElement
from ..math.pluecker import *


class ThinLens(FlatOpticalElement):
    '''
    '''
    def __init__(self, **kwargs):
        self.focallength = kwargs.pop('focallength')
        super(ThinLens, self).__init__(**kwargs)

    def process_photon(self, dir, pos, energy, polerization):
        h_intersect = self.intersect(dir, pos)
        distance = distance_point_point(h_intersect, self.geometry['center'][np.newaxis, :])
        if distance == 0.:
            # No change of direction for rays through the origin.
            # Need to special case this, because ration axis is not defined
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
