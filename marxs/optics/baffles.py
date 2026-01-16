# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from .base import FlatOpticalElement
from ..math.utils import h2e
from ..visualization.utils import plane_with_hole


class Baffle(FlatOpticalElement):
    """Plate with rectangular hole that allows photons through.

    The probability of photons that miss is set to 0, assuming that the
    baffles describes an opening in an otherwise closed structure.

    Parameters
    ----------
    photons: `astropy.table.Table`
        table that includes information on all of the photons
    """

    display = {'color': (1., 0.5, 0.4),
               'outer_factor': 3,
               'shape': 'plane with hole'}

    def process_photons(self, photons, intersect, interpos, intercoos):
        photons["pos"][intersect] = interpos[intersect]
        photons['probability'][~intersect] = 0
        return photons

    def triangulate_inner_outer(self):
        """Return a triangulation of the baffle hole embedded in a square.

        The size of the outer square is determined by the ``'outer_factor'`` element
        in ``self.display``.

        Returns
        -------
        xyz : np.array
            Numpy array of vertex positions in Euclidean space
        triangles : np.array
            Array of index numbers that define triangles
        """
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
        return plane_with_hole(outer, inner)
