# Licensed under GPL version 3 - see LICENSE.rst
import os
import numpy as np
import astropy.units as u
from scipy.interpolate import RectBivariateSpline

from marxs.optics import GlobalEnergyFilter
from marxs.missions.mitsnl.catgrating import load_table2d
from marxs.missions.athena.spo import spogeom2pos4d

from .load_csv import load_number, load_table
from marxs.math.utils import xyz2zxy
from .utils import config

spogeom = load_table('spos', 'petallayout')
spogeom['r_mid'] = (spogeom['outer_radius'] + spogeom['inner_radius']) / 2

spo_pos4d = [np.dot(xyz2zxy, s) for s in spogeom2pos4d(spogeom)]

reflectivity = load_table2d(os.path.join(config['data']['caldb_inputdata'],
                                         'spos', 'coated_reflectivity.csv'))
reflectivity_interpolator = RectBivariateSpline(reflectivity[1].to(u.keV),
                                                reflectivity[2].to(u.rad),
                                                reflectivity[3][0])

geometricopening = load_number('spos', 'geometricthroughput', 'transmission') * load_number('spos', 'porespecifications', 'transmission')
geometricthroughput = GlobalEnergyFilter(filterfunc=lambda e: geometricopening,
                                         name='SPOgeometricthrougput')
