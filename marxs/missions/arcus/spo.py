# Licensed under GPL version 3 - see LICENSE.rst
import os
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from scipy.interpolate import RectBivariateSpline

from marxs.optics import GlobalEnergyFilter
from marxs.missions.athena.spo import spogeom2pos4d
from marxs.missions.athena.spo import spogeom as athena_spogeom
from marxs.utils import tablerows_to_2d

from .load_csv import load_number, load_table
from marxs.math.utils import xyz2zxy
from .utils import config

spogeom = load_table('spos', 'petallayout')
spogeom['r_mid'] = (spogeom['outer_radius'] + spogeom['inner_radius']) / 2

spo_pos4d = [np.dot(xyz2zxy, s) for s in spogeom2pos4d(spogeom)]

reflectivity = tablerows_to_2d(Table.read(os.path.join(config['data']['caldb_inputdata'],
                                                       'spos', 'coated_reflectivity.csv'),
                                          format='ascii.ecsv'))
reflectivity_interpolator = RectBivariateSpline(reflectivity[0].to(u.keV),
                                                reflectivity[1].to(u.rad),
                                                reflectivity[2][0])

geometricopening = load_number('spos', 'geometricthroughput', 'transmission') * load_number('spos', 'porespecifications', 'transmission')
geometricthroughput = GlobalEnergyFilter(filterfunc=lambda e: geometricopening,
                                         name='SPOgeometricthrougput')


# Make an SPO layout that's larger than MIDEX Arcus, using exact Athena sizes for each SPO
athena_petal1 = athena_spogeom[athena_spogeom['Petal #'] == 1]
min_angle_distance = [np.abs(np.diff(athena_petal1[athena_petal1['Row #'] == i]['clocking_angle'])).min()
                       for i in range(1, 16)]
spos_large = []
for i in range(2, 10):
   row = athena_spogeom[athena_spogeom['Row #'] == i][:5]
   row['clocking_angle'] = np.arange(-2, 2.1) * min_angle_distance[i - 1] * athena_spogeom['clocking_angle'].unit
   spos_large.append(row)
   # Treat second XOU stack as independent row for now
   # In my table, they are in the second half of the table
   row = athena_spogeom[600:][athena_spogeom[600:]['Row #'] == i][:5]
   row['clocking_angle'] = np.arange(-2, 2.1) * min_angle_distance[i - 1] * athena_spogeom['clocking_angle'].unit
   spos_large.append(row)
spos_large = vstack(spos_large)
spos_large['MM #'] = np.arange(1, len(spos_large) + 1)
spos_large.remove_column('Petal #')
spos_large_pos4d = [np.dot(xyz2zxy, s) for s in spogeom2pos4d(spos_large)]


def rmid_spo(conf):
    '''Looking at SPO positions, find midpoint of all SPOs.

    To simplify this code it only considers the mid-points of the SPOs, not
    their rotation angle. Since the rectangles are rotated a bit, e.g.
    the aperture needs to be a bit bigger than the area covered by SPOs.
    Also, if elements are above or below the SPOs, the beam may converge
    or come from the side. `azscale` and `rscale` can be set to adjust
    the outer corners a bit.

    Parameters
    ----------
    conf : dict
        Configuration dictionary with entry 'spo_geom' and 'spo_pos4d'

    Returns
    -------
    zoom : float
        middle of the SPO array in y direction
    '''
    spopos = np.array(conf['spo_pos4d'])
    return 0.5 * (spopos[:, 1, 3].max() + spopos[:, 1, 3].min())


def zoom_to_cover_all_spos(conf, azscale=.8, rscale=1.1):
    '''Looking at SPO positions, find zoom of rectangle that covers it.

    To simplify this code it only considers the mid-points of the SPOs, not
    their rotation angle. Since the rectangles are rotated a bit, e.g.
    the aperture needs to be a bit bigger than the area covered by SPOs.
    Also, if elements are above or below the SPOs, the beam may converge
    or come from the side. `azscale` and `rscale` can be set to adjust
    the outer corners a bit.

    Parameters
    ----------
    conf : dict
        Configuration dictionary with entry 'spo_geom' and 'spo_pos4d'
    azscale : float
        The outer edge is set to the center of the outermost SPO, plus
        `azscale` times the dimension in azimuth (global x coordinate)
    rscale : float
        The outer edge is set to the center of the outermost SPO, plus
        `rscale` times the dimension in radius (global y coordinate)

    Returns
    -------
    zoom : np.array of shape (3,)
        Zoom array as input to make pos4d
    '''
    spopos = np.array(conf['spo_pos4d'])
    azmid = 0.5 * (spopos[:, 0, 3].max() + spopos[:, 0, 3].min())
    rmid = rmid_spo(conf)
    delta_az = conf['spo_geom']['azwidth']
    delta_r = conf['spo_geom']['outer_radius'] - conf['spo_geom']['inner_radius']
    return np.array([1,
                     spopos[:, 0, 3].max() - azmid + azscale * delta_az.max(),
                     spopos[:, 1, 3].max() - rmid + rscale * delta_r.max()])