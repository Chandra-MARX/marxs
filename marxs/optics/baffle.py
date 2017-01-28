# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from .base import FlatOpticalElement
from ..math.pluecker import h2e
from ..visualization.utils import get_color, plane_with_hole


class Baffle(FlatOpticalElement):
    '''Plate with rectangular hole that allows photons through.

    The probability of photons that miss is set to 0.

    Parameters
    ----------
    photons: astropy Table
        table that includes information on all of the photons
    '''

    display = {'color': (1., 0.5, 0.4),
               'outer_factor': 3}


    def process_photons(self, photons, intersect, intercoos, interpoos):
        photons['pos'][intersect] = intercoos[intersect]
        photons['probability'][~intersect] = 0
        return photons

    def _plot_mayavi(self, viewer=None):

        r_out = self.display.get('outer_factor', 3)
        g = self.geometry
        outer = h2e(g['center']) + r_out * np.vstack([h2e( g['v_x']) + h2e(g['v_y']),
                                                      h2e(-g['v_x']) + h2e(g['v_y']),
                                                      h2e(-g['v_x']) - h2e(g['v_y']),
                                                      h2e( g['v_x']) - h2e(g['v_y'])
        ])
        inner =  h2e(g['center']) + np.vstack([h2e( g['v_x']) + h2e(g['v_y']),
                                               h2e(-g['v_x']) + h2e(g['v_y']),
                                               h2e(-g['v_x']) - h2e(g['v_y']),
                                               h2e( g['v_x']) - h2e(g['v_y'])
        ])
        xyz, triangles = plane_with_hole(outer, inner)

        from mayavi.mlab import triangular_mesh

        # turn into valid color tuple
        self.display['color'] = get_color(self.display)
        t = triangular_mesh(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles, color=self.display['color'])
        # No safety net here like for color converting to a tuple.
        # If the advanced properties are set you are on your own.
        for n in t.property.trait_names():
            if n in self.display:
                setattr(t.module_manager.children[0].actor.property, n, self.display(n))
        return t
