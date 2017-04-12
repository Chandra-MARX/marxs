# Licensed under GPL version 3 - see LICENSE.rst
'''Geometry and operations of the Chandra Observatory.

This module provides the optical elements on the Chandra Observatory.

.. warning:: Currently incomplete.

   Missing:

    - a lot

   Incomplete:

   - Grating order efficiencies are the same for every order.

Some spacecraft properties are defined in module attributes, others are read from
reference files in the CALDB (or the files shipped with `classic MARX`_).
Properties that are very easy to define and that are very unlikely to ever change
(e.g. the mapping of ACIS chip number to chip name) are set in this module.
That makes is easier to see where the numbers come from and thus helps for users
who look at this module as an example of how to build up complex marxs setups.
'''
import numpy as np

from marxs.optics import MarxMirror
from .hess import HETG
from .chandra import Chandra
from .dither import LissajousDither
from .acis import ACIS
from .data import (NOMINAL_FOCALLENGTH, AIMPOINTS, PIXSIZE)
from .hrma_py import Aperture, HRMA
