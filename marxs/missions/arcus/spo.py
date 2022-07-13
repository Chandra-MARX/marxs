# Licensed under GPL version 3 - see LICENSE.rst
import os
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import astropy.units as u
from scipy.interpolate import RectBivariateSpline

from marxs.optics import GlobalEnergyFilter
from marxs.missions.mitsnl.catgrating import load_table2d

from .load_csv import load_number, load_table
from marxs.math.utils import xyz2zxy
from .utils import config

spogeom = load_table('spos', 'petallayout')
spogeom['r_mid'] = (spogeom['outer_radius'] + spogeom['inner_radius']) / 2
spo_pos4d = []
# Convert angle to quantity here to make sure that unit is taken into account
for row, ang in zip(spogeom,
                    u.Quantity(spogeom['clocking_angle']).to(u.rad).value):
    spo_pos4d.append(compose([0,  # focallength,  # - spogeom[i]['d_from_12m']
                              row['r_mid'] * np.sin(ang),
                              row['r_mid'] * np.cos(ang)],
                             euler2mat(-ang, 0., 0.),
                             # In detail this should be (primary_length + gap + secondary_length) / 2
                             # but the gap is somewhat complicated and this is only used
                             # for display, we'll ignore that for now.
                             [row['primary_length'],
                              row['azwidth'] / 2.,
                              (row['outer_radius'] - row['inner_radius']) / 2.]))

spo_pos4d = [np.dot(xyz2zxy, s) for s in spo_pos4d]

reflectivity = load_table2d(os.path.join(config['data']['caldb_inputdata'],
                                         'spos', 'coated_reflectivity.csv'))
reflectivity_interpolator = RectBivariateSpline(reflectivity[1].to(u.keV),
                                                reflectivity[2].to(u.rad),
                                                reflectivity[3][0])

geometricopening = load_number('spos', 'geometricthroughput', 'transmission') * load_number('spos', 'porespecifications', 'transmission')
geometricthroughput = GlobalEnergyFilter(filterfunc=lambda e: geometricopening,
                                         name='SPOgeometricthrougput')
