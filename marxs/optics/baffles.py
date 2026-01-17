# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from .base import FlatOpticalElement
from ..math.utils import h2e
from ..math import geometry
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

    default_geometry = geometry.RectangleHole
    display = {"color": (1.0, 0.5, 0.4)}

    def process_photons(self, photons, intersect, interpos, intercoos):
        photons["pos"][intersect] = interpos[intersect]
        photons['probability'][~intersect] = 0
        return photons


class CircularBaffle(Baffle):
    """Baffle plate with circular hole that allows photons through.

    The probability of photons that miss is set to 0, assuming that the
    baffles describes an opening in an otherwise closed structure.

    Parameters
    ----------
    photons: `astropy.table.Table`
        table that includes information on all of the photons
    """

    default_geometry = geometry.CircularHole
