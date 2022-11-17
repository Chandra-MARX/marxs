# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from marxs.optics.base import OpticalElement
from marxs.simulator import ParallelCalculated
from marxs.math.utils import h2e
from marxs.missions.mitsnl.catgrating import InterpolateEfficiencyTable as IET

# CAT gratings tabulated data
order_selector_Si = IET(
    get_pkg_data_filename('data/Si_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'))
order_selector_Si.coating = 'None'
order_selector_Pt = IET(
    get_pkg_data_filename('data/SiPt_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'))
order_selector_Pt.coating = 'Pt'
