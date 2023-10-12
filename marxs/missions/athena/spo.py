# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import astropy.units as u
from astropy.utils.data import get_pkg_data_path
from astropy.table import Table
from scipy.interpolate import interp1d

from marxs.optics import (PerfectLens, GlobalEnergyFilter,
                          RadialMirrorScatter)
from marxs.optics import aperture
from marxs.optics.aperture import CircleAperture
from marxs.simulator import Parallel, Sequence
from marxs.math.utils import e2h, h2e, norm_vector
from marxs.math.polarization import parallel_transport
from marxs.math.utils import xyz2zxy

__all__ = ['aperture',
           'spogeom', 'spo_pos4d', 'spogeom2pos4d',
           'PerfectLensSegment',
           'SPOChannelMirror', 'ScatterPerChannel',
           'geometricthroughput', 'willingaleloss',
           'spomounting',
           ]

spogeom = Table.read(get_pkg_data_path('data/xous.csv'), format='ascii.ecsv')
spogeom['r_mid'] = (spogeom['outer_radius'] + spogeom['inner_radius']) / 2

def spogeom2pos4d(spogeom):
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
                                   # 36 reflective plates in each stack
                                   36 * row['plate_height'] / 2.
                                   ]
                                  )
                         )
    return spo_pos4d

spo_pos4d = [np.dot(xyz2zxy, s) for s in spogeom2pos4d(spogeom)]


aperture = CircleAperture(position=[0, 0, 12200],
                          zoom=[1,
                                np.max(spogeom['outer_radius']) * 1.05,
                                np.max(spogeom['outer_radius']) * 1.05],
                          orientation=xyz2zxy[:3, :3])

class PerfectLensSegment(PerfectLens):
    '''A segment of a perfect lens

    This represents a single pair of XOUs or a single SPO.
    It acts as a perfect lens, where a ray through the center is not broken at all.

    Parameters
    ----------
    d_center_optical_axis : float
        Distance between the optical axis and the center of this element in mm. The optical axis is
        located in -z direction, and the normal pos4d keywords should be used to give the size,
        location, and rotation of this element.
    reflectivity_interpolator : callable
        Callable that accepts energy and reflectivity angle as input an returns the probability
        of single reflection. The calling signature is written to match a
        `scipy.interpolate.RectBivariateSpline`.
        Default is to always return 1.
    '''
    def __init__(self, **kwargs):
        self.d_center_optax = kwargs.pop('d_center_optical_axis')
        self.reflectivity_interpolator = kwargs.pop('reflectivity_interpolator',
                                                    lambda pos, ang, grid: np.ones_like(pos))
        super().__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # A ray through the center is not broken.
        # So, find out where a central ray would go.
        p_opt_axis = self.geometry['center'] - self.d_center_optax * self.geometry['e_z']
        focuspoints = h2e(p_opt_axis) + self.focallength * norm_vector(h2e(photons['dir'][intersect]))
        dir = norm_vector(e2h(focuspoints - h2e(interpos[intersect]), 0))
        pol = parallel_transport(photons['dir'].data[intersect, :], dir,
                                 photons['polarization'].data[intersect, :])
        angle = np.arccos(np.abs(np.einsum("ij,ij->i", h2e(dir),
                                         norm_vector(h2e(photons['dir'][intersect])))))
        return {'dir': dir, 'polarization': pol,
                'probability': self.reflectivity_interpolator(photons['energy'][intersect],
                                                         angle / 4,
                                                         grid=False)**2
                }


class SPOChannelMirror(Parallel):
    '''A collection of PerfectLensSegments into plates

    A number of SPOs are mounted together on a single plate and they are
    grouped into a Parallel structure here.

    Parameters
    ----------
    spo_pos4d : list of (4, 4) arrays
        pos4d arrays for each SPO or XOU pair
    spogeom : `astropy.table.Table`
        Table with columns "rmid" (radius from optical axis for each SPO / XOU pair) and
        "focal_length".
    reflectivity_interpolator : callable
        See `PerfectLensSegment`
    '''
    def __init__(self, **kwargs):
        kwargs['elem_pos'] = kwargs.pop('spo_pos4d')
        spogeom = kwargs.pop('spo_geom')
        kwargs['elem_class'] = PerfectLensSegment
        kwargs['elem_args'] = {'d_center_optical_axis': list(spogeom['r_mid']),
                               'focallength': list(spogeom['focal_length']),
                               'reflectivity_interpolator': kwargs.pop('reflectivity_interpolator')}
        kwargs['id_col'] = 'xou'
        super().__init__(**kwargs)


geometricthroughput = GlobalEnergyFilter(filterfunc=lambda e: 0.92609,
                                         name='SPOgeometricthrougput')
'''Open area of each plate (i.e. pore width/pore pitch).

This is set by having a 0.17 mm wide rib and a 2.3 mm rib pitch.
'''

tab = Table.read(get_pkg_data_path('data/lossterm.csv'), format='ascii.ecsv')
willingaleloss = GlobalEnergyFilter(
    filterfunc=interp1d(tab['energy'].to(u.keV, equivalencies=u.spectral()),
                        tab[tab.colnames[1]]),
    name='Loss_from_nonideal_spos')
'''Additional loss terms from pore-level ray-tracing (tabluted).

Includes, e.g. the fact the ribs are parallel, not radial etc.
'''


class ScatterPerChannel(RadialMirrorScatter):
    '''A scatter of infinite size that identifies photons by spo id

    This bypasses the intersection calculation and instead
    just selects photons for this scatter by spo id.

    Parameters
    ----------
    min_id : integer
    max_id : integer
        Photons with xou id between ``min_id`` and ``max_id`` are
        scattered.
    '''
    display = {'shape': 'None'}
    loc_coos_name = ['scat_y', 'scat_z']

    def __init__(self, **kwargs):
        self.min_id = kwargs.pop('min_id')
        self.max_id = kwargs.pop('max_id')
        super().__init__(**kwargs)

    def __call__(self, photons):
        # interpos is used to automatically set new position
        # (which we want unaltered, thus we pass pos) and local coords
        # but intercoos we want to set to a useful number for analysis later
        intersect, interpos, intercoos = self.geometry.intersect(photons['dir'].data,
                                                                 photons['pos'].data)
        # intersect is done based on xou to avoid problem of overlap
        # between channels
        intersect = ((photons['xou'] >= self.min_id) &
                     (photons['xou'] < self.max_id))
        return self.process_photons(photons, intersect, photons['pos'].data,
                                    intercoos)


def spomounting(photons):
    '''Remove photons that do not go through an SPO but hit the
    frame part of the petal.'''
    photons['probability'][photons['xou'] < 0] = 0.
    return photons


class SimpleSPOs(Sequence):

    def __init__(self, conf, **kwargs):
        mirror = []
        entrancepos = [0, 0, 12000]
        mirror.append(SPOChannelMirror(position=entrancepos,
                                       reflectivity_interpolator=conf['reflectivity_interpolator'],
                                       spo_pos4d=conf['spo_pos4d'],
                                       spo_geom=conf['spo_geom']))
        mirror.append(ScatterPerChannel(position=entrancepos,
                                        min_id=0,
                                        max_id=10000,
                                        inplanescatter=conf['inplanescatter'],
                                        perpplanescatter=conf['perpplanescatter'],
                                        orientation=xyz2zxy[:3, :3]))
        mirror.append(spomounting)
        mirror.append(geometricthroughput)
        mirror.append(willingaleloss)
        super().__init__(elements=mirror, **kwargs)
