# Licensed under GPL version 3 - see LICENSE.rst
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from marxs.missions.mitsnl.catgrating import InterpolateEfficiencyTable as IET

# CAT gratings tabulated data
order_selector_Si = IET(Table.read(
    get_pkg_data_filename('data/Si_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'),
                          format='ascii.ecsv'))
order_selector_Si.coating = 'None'
order_selector_Pt = IET(Table.read(
    get_pkg_data_filename('data/SiPt_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'),
                          format='ascii.ecsv'))
order_selector_Pt.coating = 'Pt'
