import copy

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from marxs import analysis, design, optics, simulator
from marxs.base import MarxsElement, TagVersion
from marxs.design import tolerancing as tol
from marxs.design.bendgratings import NumericalChirpFinder, chirp_gratings
from marxs.math.geometry import Cylinder
from marxs.missions.arcus.arcus import FiltersAndQE
from marxs.missions.lynx.lynx import LynxGAS, add_rowland_to_conf
from marxs.missions.lynx.mirror import MetaShellAperture, PerfectLensSegment
from marxs.missions.mitsnl.catgrating import (CATL1L2Stack,
                                              InterpolateEfficiencyTable,
                                              NonParallelCATGrating,
                                              catsupportbars)
from marxs.optics import FlatDetector, OrderSelector
from marxs.simulator import Propagator

tagversion = TagVersion(SATELLIT='AXIS')

# CAT gratings tabluated data
order_selector_Si = InterpolateEfficiencyTable(Table.read(
    get_pkg_data_filename('data/Si_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'),
    format='ascii.ecsv'))
order_selector_Si.coating = 'None'
order_selector_Pt = InterpolateEfficiencyTable(Table.read(
    get_pkg_data_filename('data/SiPt_efficiency_5_7.dat',
                          package='marxs.missions.mitsnl'),
    format='ascii.ecsv'))
order_selector_Pt.coating = 'Pt'


l1_order_selector = OrderSelector(orderlist=np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
                                  p=np.array([0.006, 0.0135, 0.022, 0.028, 0.861, 0.028, 0.022, 0.0135, 0.006]))
'''Simple order selector for diffraction on L1.

The numbers in the array are calculated for the 2018 Arcus gratings
assuming the L1 structure is independent from the grating membrane itself
(which is not true, but a valid first approximation.)
For AXIS, we are using 50% deeper gratings, yet the L1 bars are thinner.
Need better numbers for that.
'''


conf = {'metashellgeometry': Table.read(get_pkg_data_filename('AXIS_metashellgeom.dat'),
                                        format='ascii.ecsv'),
        # This gives a PSF HPD around 1 arcsec.
        # The nominal 1 arcsec includes alignment, but until I code that up, I just makes 
        # PSF a little wider here.
        # 4e-6 and 2e-6 work well otherwise.
        'inplanescatter': 5e-6 * u.rad,
        'perpplanescatter': 2.5e-6 * u.rad,
        'mirrorcoating': 'ir_reflectivity.dat',
        'mirrorgeometricscaling': 0.61,
        'blazeang': np.deg2rad(1.6),
        'focallength': 9000.,
        'design_tilted_torus': 8.6e3, 
        'alphafac': 2.2,
        'betafac': 4.4,
        'grating_size': np.array([30., 60.]),
        'grating_frame': 2.,
        'det_kwargs': {'y_range': [400, 600],
                       'd_element': [24.576 * 2 + 0.824 * 2 + 0.5,
                                     24.576 + 0.824 * 2 + 0.5],
                       'elem_class': optics.FlatDetector,
                       'elem_args': {'zoom': [1, 24.576, 12.288],
                                     'pixsize': 0.024,
                                     # orientation flips around CCDs so that det_x increases
                                     # with increasing x coordinate
                                    'orientation': np.array([[-1, 0, 0],
                                                             [0, -1, 0],
                                                             [0, 0, +1]])
                                     }},
        'gas_kwargs': {'parallel_spec': np.array([0., 1., 0., 0.]),
                       'normal_spec': np.array([0, 0, 0, 1]),
                       'guess_distance': 9e3,
                       'elem_class': CATL1L2Stack,
                       'elem_args': {'d': 2e-4,
                                     'order_selector': order_selector_Si,
                                     'l1_dims': {'bardepth': 5.7 * u.micrometer,
                                                 'period': 0.005 * u.mm,
                                                 'barwidth': 0.0007 * u.mm},
                                     'l2_dims': {'bardepth': 0.5 * u.mm,
                                                 'period': 0.966 * u.mm,
                                                 'barwidth': 0.07 * u.mm},
                                     # Set sigma to 0. Effectively disables this
                                     # factor.
                                     'qualityfactor': {'d': 200. * u.um,
                                                       'sigma': 1.5 * u.um},
                                     'l1_order_selector': l1_order_selector,
                                     },
                       },
    }


add_rowland_to_conf(conf)

conf_6060 = copy.copy(conf)
conf_6060['grating_size'] = np.array([60., 60.])

conf_chirp = copy.copy(conf)
conf_chirp['grating_size'] = np.array([80., 160.])
conf_chirp['chirp_energy'] = 0.6
# Use of a non-integer order makes no sense physically
# but is just a numerical tool to optimize at the blaze peak
conf_chirp['chirp_order'] = -5.4



class RowlandDetArray(design.rowland.RectangularGrid):
    def __init__(self, conf):
        y_offset = conf['rowland'].geometry['center'][1]
        super().__init__(rowland=conf['rowland'],
                         y_range=np.array(conf['det_kwargs']['y_range']) - y_offset,
                         d_element=conf['det_kwargs']['d_element'],
                         elem_args=conf['det_kwargs']['elem_args'],
                         elem_class=conf['det_kwargs']['elem_class'],
                         guess_distance=25.,
                         id_col='CCD_ID')


imaging = FlatDetector(zoom=[1, 15, 15], pixsize=0.008)
# zero-order detector based on https://axis.astro.umd.edu/images/proposal.pdf
imaging.loc_coos_name = ['imaging_x', 'imaging_y']
imaging.detpix_name = ['imaging_x', 'imaging_y']


# Place an additional detector on the Rowland circle.
detcirc = optics.CircularDetector(geometry=Cylinder.from_rowland(conf['rowland'],
                                                                 width=50,
                                                                 rotation=np.pi,
                                                                 kwargs={'phi_lim':[-np.pi/2, np.pi/2]}))
detcirc.loc_coos_name = ['detcirc_phi', 'detcirc_y']
detcirc.detpix_name = ['detcircpix_x', 'detcircpix_y']

flatdet = FlatDetector(zoom=[1, 1e5, 1e5])
flatdet.loc_coos_name = ['detinf_x', 'detinf_y']
flatdet.detpix_name = ['detinfpix_x', 'detinfpix_y']
flatdet.display['shape'] = 'None'


class MirrorGeometricScaling(MarxsElement):
    def __init__(self, conf):
        self.scale = conf['mirrorgeometricscaling']

    def __call__(self, photons):
        photons['probability'] = self.scale * photons['probability']
        return photons


class PerfectAXIS(simulator.Sequence):
    '''Default Definition of Lynx without any misalignments'''
    def add_mirror(self, conf):
        return [MetaShellAperture(conf), 
                PerfectLensSegment(conf, position=[conf['focallength'], 0, 0],
                zoom=[1, 1000, 1000]),
                MirrorGeometricScaling(conf),
                optics.RadialMirrorScatter(inplanescatter= conf['inplanescatter'],
                                           perpplanescatter=conf['perpplanescatter'],
                                           position=[conf['focallength'], 0, 0],
                                           zoom=[1, 1000, 1000])]

    def add_gas(self, conf):
        return [LynxGAS(conf),
                catsupportbars]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in its own
        function makes it easy to override for derived classes.
        '''
        proj2 = analysis.ProjectOntoPlane()
        proj2.loc_coos_name = ['projcirc_y', 'projcirc_z']
        return [RowlandDetArray(conf),
                imaging,
                analysis.ProjectOntoPlane(),
                Propagator(distance=-1000),
                detcirc,
                proj2,
                Propagator(distance=-1000),
                flatdet]

    def post_process(self):
        self.KeepPos = simulator.KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=conf, gratings=True, **kwargs):
        elem = self.add_mirror(conf)
        if gratings:
            elem.extend(self.add_gas(conf))
        elem.append(FiltersAndQE())
        elem.extend(self.add_detectors(conf))
        elem.append(tagversion)

        super().__init__(elements=elem,
                         postprocess_steps=self.post_process(),
                         **kwargs)
        if ('chirp_energy' in conf) and ('chirp_order' in conf):
            gratings = self.elements_of_class(NonParallelCATGrating, subclass_ok=False)
            opt = NumericalChirpFinder(detcirc,
                                       order=conf['chirp_order'],
                                       energy=conf['chirp_energy'],
                                       d=conf['gas_kwargs']['elem_args']['d'])
            chirp_gratings(gratings, opt, conf['gas_kwargs']['elem_args']['d'])


class AXISForPlot(PerfectAXIS):
    def add_detectors(self, conf):
        zero = FlatDetector(zoom=[1, 15, 15])
        zero.loc_coos_name = ['zero_x', 'zero_y']
        zero.detpix_name = ['zero_x', 'zero_y']
        return [RowlandDetArray(conf),
                imaging]


class AXIS(PerfectAXIS):
    def __init__(self, conf=conf, **kwargs):
        super().__init__(conf=conf, **kwargs)
        for row in conf['alignmentbudget']:
            elem = self.elements_of_class(row[0])
            if row[1] == 'global':
                tol.moveglobal(elem, *row[3])
            elif row[1] == 'individual':
                tol.wiggle(elem, *row[3])
            else:
                raise NotImplementedError('Alignment error {} not implemented'.format(row[1]))

