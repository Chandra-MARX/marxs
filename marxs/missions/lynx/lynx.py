# Licensed under GPL version 3 - see LICENSE.rst
''' Lynx telescope and instrument definition.

Since some aspects of the telescope and not well-defined yet, there are some rough edges.

Below is the running list of input data that could be improved with no or little code changes:

- QE (and entrance filters if any are present) for zero-order instruments missing
- Filter curves and QE for XGS detectors only defined to 2.1 keV. Just added arbitrary numbers
  after that to avoid overflow errors in the interpolation.
- Check mirror model. Does is use the energy dependence for different shells?
- Update L1 diffraction efficiencies. Note that transmission through L1 is currently handled separately.
- Update L2 dimensions.
'''

import copy
from astropy.table import Table
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import transforms3d
from scipy.interpolate import interp1d

from marxs import optics, simulator, design, analysis
from marxs.design.rowland import (design_tilted_torus, RowlandTorus,
                                  CircularMeshGrid)
from marxs.math.geometry import Cylinder
from marxs.optics import FlatDetector, CATGrating
from marxs.simulator import Propagator
from marxs.missions.mitsnl.catgrating import CATL1L2Stack, catsupportbars, l1_order_selector
from marxs.design import tolerancing as tol
from marxs.design.bendgratings import NumericalChirpFinder, chirp_gratings

from . import ralfgrating
from .mirror import MetaShell, MetaShellAperture



from marxs.base import TagVersion

tagversion = TagVersion(SATELLIT='LYNX')

filterdata = Table.read(get_pkg_data_filename('data/filtersqe.dat'),
                        format='ascii.ecsv')
filterqe = optics.GlobalEnergyFilter(filterfunc=interp1d(filterdata['energy'] / 1000,
                                                         filterdata['Total_throughput']))

conf = {'metashellgeometry': Table.read(get_pkg_data_filename('data/metashellgeom.dat'),
                                        format='ascii.ecsv'),
        'metashelleff': Table.read(get_pkg_data_filename('data/metashelleff.dat'),
                                    format='ascii.ecsv'),
        'inplanescatter': 2e-6 * u.rad,
        'perpplanescatter': 2e-6 * u.rad,
        'blazeang': np.deg2rad(1.6),
        'focallength': 10000.,
        'design_tilted_torus': 9.6e3, 
        'alphafac': 2.2,
        'betafac': 4.4,
        'grating_size': np.array([20., 50.]),
        'grating_frame': 2.,
        'det_kwargs': {'y_range': [400, 800],
                       'd_element': [16.884, 16.884],
                       'elem_class': optics.FlatDetector,
                       'elem_args': {'zoom': [1, 8.192, 8.192],
                                     'pixsize': 0.016}},
        'gas_kwargs': {'parallel_spec': np.array([0., 1., 0., 0.]),
                       'normal_spec': np.array([0, 0, 0, 1]),
                       'x_range': [7e3, 1e4],
                       'elem_class': CATL1L2Stack,
                       'elem_args': {'d': 2e-4,
                                     'order_selector': ralfgrating.order_selector_Si,
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
    R, r, pos4d = design_tilted_torus(conf['design_tilted_torus'], conf['alpha'], conf['beta'])
    conf['rowland'] = RowlandTorus(R, r, pos4d=pos4d)
    conf['blazemat'] = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]),
                                                         -conf['blazeang'])

add_rowland_to_conf(conf)

conf_5050 = copy.copy(conf)
conf_5050['grating_size'] = np.array([50., 50.])

conf_chirp = copy.copy(conf)
conf_chirp['grating_size'] = np.array([80., 160.])
conf_chirp['chirp_energy'] = 0.6
# Use of a non-integer order makes no sense physically
# but is just a numerical tool to optimize at the blaze peak
conf_chirp['chirp_order'] = -5.4


class LynxGAS(CircularMeshGrid):
    def __init__(self, conf):
        gg = conf['gas_kwargs']
        # Add in / update keywords that are calculated based on other conf
        # entries
        gg['rowland'] = conf['rowland']
        gg['d_element'] = conf['grating_size'] + 2 * conf['grating_frame']
        gg['radius'] = [np.min(conf['metashellgeometry']['r_inner']),
                        np.max(conf['metashellgeometry']['r_outer'])]
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


class RowlandDetArray(design.rowland.RectangularGrid):
    def __init__(self, conf):
        super(RowlandDetArray, self).__init__(conf['rowland'], **conf['det_kwargs'],
                                              guess_distance=25.)


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


class PerfectLynx(simulator.Sequence):
    '''Default Definition of Lynx without any misalignments'''
    def add_mirror(self, conf):
        return [MetaShellAperture(conf), MetaShell(conf)]

    def add_gas(self, conf):
        return [LynxGAS(conf),
                catsupportbars]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in its own
        function makes it easy to override for derived classes.
        '''
        microcal = FlatDetector(zoom=[1, 50, 50])
        microcal.loc_coos_name = ['microcal_x', 'microcal_y']
        microcal.detpix_name = ['microcalpix_x', 'microcalpix_y']
        proj2 = analysis.ProjectOntoPlane()
        proj2.loc_coos_name = ['projcirc_y', 'projcirc_z']
        return [RowlandDetArray(conf),
                analysis.ProjectOntoPlane(),
                Propagator(distance=-1000),
                detcirc,
                proj2,
                Propagator(distance=-1000),
                flatdet,
                Propagator(distance=-1000),
                microcal]

    def post_process(self):
        self.KeepPos = simulator.KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=conf, **kwargs):
        elem = self.add_mirror(conf)
        elem.extend(self.add_gas(conf))
        elem.append(filterqe)
        elem.extend(self.add_detectors(conf))
        elem.append(tagversion)

        super(PerfectLynx, self).__init__(elements=elem,
                                          postprocess_steps=self.post_process(),
                                          **kwargs)
        if ('chirp_energy' in conf) and ('chirp_order' in conf):
            gratings = self.elements_of_class(CATGrating, subclass_ok=False)
            opt = NumericalChirpFinder(detcirc,
                                       order=conf['chirp_order'],
                                       energy=conf['chirp_energy'],
                                       d=conf['gas_kwargs']['elem_args']['d'])
            chirp_gratings(gratings, opt, conf['gas_kwargs']['elem_args']['d'])


class LynxForPlot(PerfectLynx):
    def add_detectors(self, conf):
        microcal = FlatDetector(zoom=[1, 50, 50])
        microcal.loc_coos_name = ['microcal_x', 'microcal_y']
        microcal.detpix_name = ['microcalpix_x', 'microcalpix_y']
        return [RowlandDetArray(conf),
                microcal]


class Lynx(PerfectLynx):
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


def reformat_errorbudget(budget, globalfac=0.8):
    '''Reformat the error budget

    Last, global misalignment (that's not random) must be
    scaled in some way. Here, I use 0.8 sigma, which is the
    mean absolute deviation for a Gaussian.

    Parameters
    ----------
    budget : list
        See reference implementation for list format
    globalfac : ``None`` or float
        Factor to apply for global tolerances. A "global" tolerance is drawn
        only once per simulation. In contrast, for "individual" tolerances
        many draws are done and thus the resulting layout actually
        represents a distribution. For a "global" tolerance, the result hinges
        essentially on a single random draw. If this is set to ``None``,
        misalignments are drawn statistically. Instead, the tolerances can be
        scaled determinisitically, e.g. by "0.8 sigma" (the mean absolute
        deviation for a Gaussian distribution).
    '''
    for row in budget:
        tol = np.array(row[2], dtype=float)
        if row[1] == 'global':
            if globalfac is not None:
                tol *= globalfac
            else:
                tol *= np.random.randn(len(tol))

        row[3] = tol


align_requirement_in = [
    [LynxGAS, 'global', [1, 0.5, .5,
                         np.deg2rad(0.01), np.deg2rad(0.01), np.deg2rad(0.01)],
     None, 'CAT structure to mirrors'],
    [LynxGAS, 'individual', [.1, .25, .25,
                             np.deg2rad(0.1), np.deg2rad(0.2), np.deg2rad(0.1)],
     None, 'individual CAT gratings structure'],
    [RowlandDetArray, 'global', [0.1, 1., 3.,
                                 np.deg2rad(0.5), np.deg2rad(1), np.deg2rad(.01)],
     None, 'Camera to front assembly']]

align_requirement = copy.deepcopy(align_requirement_in)
reformat_errorbudget(align_requirement)

conf['alignmentbudget'] = align_requirement
