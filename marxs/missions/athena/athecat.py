# Licensed under GPL version 3 - see LICENSE.rst
import copy
import os

import numpy as np
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import transforms3d
from scipy.interpolate import RectBivariateSpline

from marxs import optics
from marxs.missions.mitsnl.catgrating import CATL1L2Stack, l1_order_selector
from marxs.base import TagVersion
from marxs import design
from marxs.math.geometry import Cylinder
from marxs.math.utils import xyz2zxy
from marxs.missions.mitsnl.catgrating import InterpolateEfficiencyTable
from marxs.missions.athena import spo
from marxs.missions.mitsnl.catgrating import load_table2d
from marxs import simulator
from marxs import analysis


# Probably get rid of all of these
from marxs.missions.arcus.utils import config as arcus_config
from arcus.instrument.arcus import FiltersAndQE

tagversion = TagVersion(SATELLIT='Athena', GRATING='CAT')

# SPOs reflectivity tabluated data
reflectivity = load_table2d(os.path.join(arcus_config['data']['caldb_inputdata'],
                                         'spos', 'coated_reflectivity.csv'))
reflectivity_interpolator = RectBivariateSpline(reflectivity[1].to(u.keV),
                                                reflectivity[2].to(u.rad),
                                                reflectivity[3][0])


# CAT gratings tabluated data
order_selector_Si = InterpolateEfficiencyTable(
    get_pkg_data_filename('data/Si_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'))
order_selector_Si.coating = 'None'
order_selector_Pt = InterpolateEfficiencyTable(
    get_pkg_data_filename('data/SiPt_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'))
order_selector_Pt.coating = 'Pt'


conf = {
        'blazeang': np.deg2rad(1.6),
        'focallength': 12000.,
        'design_tilted_torus': 11.6e3,
        'alphafac': 2.2,
        'betafac': 4.4,
        # SPO scatter
        # factor 2.3545 converts from FWHM to sigma
        'perpplanescatter':  1.5 / 2.3545 * u.arcsec,
        # 2 * 0.68 converts HPD to sigma
        'inplanescatter': 7. / (2 * 0.68) * u.arcsec,
        'spo_pos4d': spo.spo_pos4d,
        'spo_geom': spo.spogeom,
        'reflectivity_interpolator': reflectivity_interpolator,

        # Due to the impelementation in Rowland circle array, this is reversed
        # from what I would expect.
        'grating_size': np.array([60., 60.]),
        'grating_frame': 2.,
        'det_kwargs': {'theta': [3.12, 3.18],
                       'd_element': 24.576 * 2 + 0.824 * 2 + 0.5,
                       'elem_class': optics.FlatDetector,
                       'elem_args': {'zoom': [1, 24.576, 12.288],
                                     'pixsize': 0.024,
                                     # orientation flips around CCDs so that det_x increases
                                     # with increasing x coordinate
                                    'orientation': np.array([[-1, 0, 0],
                                                             [0, -1, 0],
                                                             [0, 0, +1]])
                                     }},
        'gas_kwargs': {'parallel_spec': np.array([1., 0., 0., 0.]),
                       'normal_spec': np.array([0, 0, 0, 1]),
                       'opt_range': [8e3, 1.2e4],
                       'optimize_axis': 'z',
                       'radius': (np.min(spo.spogeom['inner_radius']),
                                  np.max(spo.spogeom['outer_radius'])),
                       'elem_class': CATL1L2Stack,
                       'elem_args': {'d': 2e-4,
                                     'order_selector': order_selector_Si,
                                     'l1_dims': {'bardepth': 5.7 * u.micrometer,
                                                 'period': 0.005 * u.mm,
                                                 'barwidth': 0.0005 * u.mm},
                                     'l2_dims': {'bardepth': 0.5 * u.mm,
                                                 'period': 0.966 * u.mm,
                                                 'barwidth': 0.05 * u.mm},
                                     # Set sigma to 0. Effectively disables this
                                     # factor.
                                     'qualityfactor': {'d': 200. * u.um,
                                                       'sigma': 0 * u.um},
                                     'l1_order_selector': l1_order_selector,
                                     },
                       },
    }


def add_rowland_to_conf(conf):
    conf['alpha'] = conf['alphafac'] * conf['blazeang']
    conf['beta'] = conf['betafac'] * conf['blazeang']
    R, r, pos4d = design.rowland.design_tilted_torus(conf['design_tilted_torus'],
                                                     conf['alpha'], conf['beta'])
    conf['rowland'] = design.RowlandTorus(R, r, pos4d=pos4d)
    conf['rowland'].pos4d = np.dot(xyz2zxy, conf['rowland'].pos4d)
    conf['blazemat'] = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]),
                                                         -conf['blazeang'])

add_rowland_to_conf(conf)


conf_chirp = copy.copy(conf)
conf_chirp['grating_size'] = np.array([80., 160.])
conf_chirp['chirp_energy'] = 0.6
# Use of a non-integer order makes no sense physically
# but is just a numerical tool to optimize at the blaze peak
conf_chirp['chirp_order'] = -5.4


class GAS(design.rowland.CircularMeshGrid):
    def __init__(self, conf):
        gg = conf['gas_kwargs']
        # Add in / update keywords that are calculated based on other conf
        # entries
        gg['rowland'] = conf['rowland']
        # Reverse grating size for d_element to account for rotation of elementa
        # roation is dome automatic by parallel_spec, but the algorithm that makes
        # the tiling does not know about that
        gg['d_element'] = conf['grating_size'] + 2 * conf['grating_frame']
        gg['elem_args']['zoom'] = [1.,
                                   conf['grating_size'][0] / 2.,
                                   conf['grating_size'][1] / 2.]
        gg['elem_args']['orientation'] = conf['blazemat']

        super().__init__(**gg)

        # Color gratings according to the sector they are in
        for e in self.elements:
            e.display = copy.deepcopy(e.display)
            # Angle from baseline in range 0..pi
            ang = np.arctan2(e.pos4d[1, 3], e.pos4d[2, 3]) % (np.pi)
            # pick one colors
            e.display['color'] = 'rgb'[int(ang / np.pi * 3)]


class RowlandDetArray(design.rowland.RowlandCircleArray):
    def __init__(self, conf):
        super().__init__(conf['rowland'], **conf['det_kwargs'])


# Place an additional detector on the Rowland circle.

detcirc = optics.CircularDetector(geometry=Cylinder.from_rowland(conf['rowland'],
                                                                 width=50,
                                                                 rotation=np.pi,
                                                                 kwargs={'phi_lim':[-np.pi/2, np.pi/2]}))
detcirc.loc_coos_name = ['detcirc_phi', 'detcirc_y']
detcirc.detpix_name = ['detcircpix_x', 'detcircpix_y']
detcirc.display['opacity'] = 0.1

flatdet = optics.FlatDetector(orientation=xyz2zxy[:3, :3], zoom=[1, 1e5, 1e5])
flatdet.loc_coos_name = ['detinf_x', 'detinf_y']
flatdet.detpix_name = ['detinfpix_x', 'detinfpix_y']
flatdet.display['shape'] = 'None'

zero = optics.FlatDetector(orientation=xyz2zxy[:3, :3], zoom=[1, 15, 15], pixsize=0.008)
# zero-order detector based on AXIS, need to adjust for Athena
zero.loc_coos_name = ['zero_x', 'zero_y']
zero.detpix_name = ['zero_x', 'zero_y']

class PerfectAthecat(simulator.Sequence):
    '''Default Definition of Athena with CAT gratings without any misalignments'''

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in its own
        function makes it easy to override for derived classes.
        '''
        proj2 = analysis.ProjectOntoPlane(orientation=xyz2zxy[:3, :3])
        proj2.loc_coos_name = ['projcirc_y', 'projcirc_z']
        return [RowlandDetArray(conf),
                analysis.ProjectOntoPlane(orientation=xyz2zxy[:3, :3]),
                simulator.Propagator(distance=-1000),
                detcirc,
                proj2,
                simulator.Propagator(distance=-1000),
                flatdet,
                simulator.Propagator(distance=-1000),
                zero]

    def post_process(self):
        self.KeepPos = simulator.KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=conf, **kwargs):
        elem = [spo.aperture,
                spo.SimpleSPOs(conf)]
        if 'gas_kwargs' in conf:
            elem.append(GAS(conf))
        elem.append(FiltersAndQE())
        elem.extend(self.add_detectors(conf))
        elem.append(tagversion)

        super().__init__(elements=elem,
                         postprocess_steps=self.post_process(),
                         **kwargs)
        if ('chirp_energy' in conf) and ('chirp_order' in conf):
            gratings = self.elements_of_class(optics.CATGrating, subclass_ok=False)
            opt = design.bendgratings.NumericalChirpFinder(detcirc,
                                       order=conf['chirp_order'],
                                       energy=conf['chirp_energy'],
                                       d=conf['gas_kwargs']['elem_args']['d'])
            design.bendgratings.chirp_gratings(gratings, opt, conf['gas_kwargs']['elem_args']['d'])


class AthecatForPlot(PerfectAthecat):
    def add_detectors(self, conf):
        return [RowlandDetArray(conf),
                zero]
