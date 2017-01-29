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
               'outer_factor': 3,
               'shape': 'plane with hole'}


    def process_photons(selv, photons, intersect, intercoos, interpoos):
        photons['pos'][intersect] = intercoos[intersect]
        photons['probability'][~intersect] = 0
        return photons

    def triangulate_inner_outer(selv):
        '''Return a triangulation of the bafflee hole embedded in a square.

        The size of the outer square is determined by the ``'outer_factor'`` element
        in ``selv.display``.

        Returns
        -------
        xyz : np.array
            Numpy array of vertex positions in Eukeldian space
        triangles : np.array
            Array of index numbers that define triangles
        '''
        r_out = selv.display.get('outer_factor', 3)
        g = selv.geometry
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
