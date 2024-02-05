# Licensed under GPL version 3 - see LICENSE.rst
from copy import deepcopy
import numpy as np
import astropy.units as u
from astropy.table import QTable
from scipy.interpolate import interp1d
import transforms3d
import os

from astropy.table import Table
from scipy.interpolate import RectBivariateSpline

import marxs
from marxs.simulator import Sequence, KeepCol, ParallelCalculated
from marxs.optics import (GlobalEnergyFilter,
                          FlatDetector,
                          CircularDetector)
from marxs import optics
from marxs.design import rowland
import marxs.analysis
from marxs.math.utils import xyz2zxy
from marxs.utils import tablerows_to_2d
from marxs.design import tolerancing as tol
from marxs.missions.mitsnl.catgrating import catsupportbars, CATL1L2Stack, l1_order_selector
from marxs.missions.mitsnl.tolerances import wiggle_and_bar_tilt
from marxs.missions.athena.spo import SPOChannelMirror, ScatterPerChannel, spomounting
from marxs.missions.athena.spo import spogeom2pos4d
from marxs.missions.mitsnl.catgrating import InterpolateEfficiencyTable

from .ralfgrating import CATfromMechanical, CATWindow
from . import spo
from . import boom
from .load_csv import load_number, load_table
from .utils import tagversion, id_num_offset, config



__all__ = ['xyz2zxy',
           'Arcus', 'ArcusForPlot', 'ArcusForSIXTE']

defaultconf = rowland.double_rowland_from_channel_distance(750., 5915.51307, 12000. - 123.569239)
rowland.add_offset_double_rowland_channels(defaultconf)
for k, v in defaultconf.items():
    if isinstance(v, rowland.RowlandTorus):
        v.pos4d = xyz2zxy @ v.pos4d
for k, v in defaultconf['pos_opt_ax'].items():
    defaultconf['pos_opt_ax'][k] = xyz2zxy @ v

defaultconf['blazeang'] = 1.8
defaultconf['n_CCDs'] = 16
defaultconf['phi_det_start'] = 0.024  # was 0.037 at 600 mm channel spacing

# SPO scatter
# factor 2.3545 converts from FWHM to sigma
defaultconf['perpplanescatter'] = 1.5 / 2.3545 * u.arcsec
# 2 * 0.68 converts HPD to sigma
defaultconf['inplanescatter'] = 7. / (2 * 0.68) * u.arcsec

defaultconf['det_kwargs']= {
    'elem_class': FlatDetector,
    # orientation flips around CCDs so that det_x increases
    # with increasing x coordinate
    'elem_args': {'pixsize': 0.024, 'zoom': [1, 24.576, 12.288],
                  'orientation': np.array([[-1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, +1]])},
    'id_num_offset': 1,
}

# If we have the Arcus calibration database, load all the remaining
# configuration from there.
# If not (e.g. we are running in CI or readthedocs)
# do not load anything and use the defaults.
# Arcus specific tests will fail, but at least the model can be imported.
if 'data' in config:
    spogeom = load_table('spos', 'petallayout')
    spogeom['r_mid'] = (spogeom['outer_radius'] + spogeom['inner_radius']) / 2

    spo_pos4d = [np.dot(xyz2zxy, s) for s in spogeom2pos4d(spogeom)]

    defaultconf['spo_pos4d'] = spo_pos4d
    defaultconf['spo_geom'] = spogeom
    reflectivity = tablerows_to_2d(Table.read(os.path.join(config['data']['caldb_inputdata'],
                                                           'spos', 'coated_reflectivity.csv'),
                                              format='ascii.ecsv'))
    reflectivity_interpolator = RectBivariateSpline(reflectivity[0].to(u.keV),
                                                reflectivity[1].to(u.rad),
                                                reflectivity[2][0])

    defaultconf['reflectivity_interpolator'] = reflectivity_interpolator

    globalorderselector = InterpolateEfficiencyTable(
        Table.read(os.path.join(config['data']['caldb_inputdata'],
                 'gratings', 'efficiency.csv'), format='ascii.ecsv'))
    '''Global instance of an order selector to use in all CAT gratings.

    As long as the efficiency table is the same for all CAT gratings, it makes
    sense to define that globally. If every grating had its own independent
    order selector, we would have to read the selector file a few hundred times.
    '''


    defaultconf['gratinggrid'] = {'d_element': [32., 32.5],
                              'elem_class': CATL1L2Stack,
                              'elem_args': {'zoom': [1., 28./2, 28.5/2],
                                            'order_selector': globalorderselector,
                                            'l1_dims': {'bardepth': 0.0057 * u.mm,
                                                        'period': 0.005 * u.mm,
                                                        'barwidth': 0.0009 * u.mm},
                                            'l2_dims': {'bardepth': 0.5 * u.mm,
                                                        'period': 0.9622504 * u.mm,
                                                        'barwidth': 0.0962250 * u.mm},
                                            'qualityfactor': {'d': 200. * u.um,
                                                              'sigma': 1.75 * u.um},
                                            'l1_order_selector': l1_order_selector,
                              },
                              'parallel_spec': np.array([1., 0., 0., 0.]),
                   }


    ccdfwhm = QTable.read(os.path.join(config['data']['caldb_inputdata'],
                                       'detectors',
                                       'ccd_2021', 'arcus_ccd_rmf_20210211.txt'),
                         format='ascii.no_header', names=['energy', 'FWHM'])
    # Units currently not set in table
    ccdfwhm['energy'] *= u.keV
    ccdfwhm['FWHM'] *= u.eV
    defaultconf['ccdredist'] = ccdfwhm


# Other arguments for the detector - are they needed for something?
#defaultconf['detector']['d_element'] = [
#    defaultconf['det_kwargs']['elem_args']['zoom'][1] * 2 + 0.824 * 2 + 0.5,
#    defaultconf['det_kwargs']['elem_args']['zoom'][2] * 2 + 0.824 * 2 + 0.5,
#]


channels = ['1', '1m', '2', '2m']
# set of all channels. Used as default for a number of functions below.


def reformat_randall_errorbudget(budget, globalfac=0.8):
    '''Reformat the numbers from LSF-CAT-Alignment-v3.xls

    Randall gives 3 sigma errors, while I need 1 sigma a input here.
    Also, units need to be converted: mu -> mm, arcsec -> rad
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
        scaled deterministically, e.g. by "0.8 sigma" (the mean absolute
        deviation for a Gaussian distribution).
    '''
    for row in budget:
        tol = np.array(row[2], dtype=float)
        tol[:3] = tol[:3] / 1000.  # mu to mm
        tol[3:] = np.deg2rad(tol[3:] / 3600.)  # arcsec to rad
        tol = tol / 3   # Randall gives 3 sigma values
        if row[1] == 'global':
            if globalfac is not None:
                tol *= globalfac
            else:
                tol *= np.random.randn(len(tol))

        row[3] = tol


class Aperture(optics.MultiAperture):
    def __init__(self, conf, channels=channels, **kwargs):
        # Set a little above entrance pos (the mirror) for display purposes.
        aper_z = 12200
        channels = deepcopy(channels)

        apers = []

        # Currently, channel 1 and 1m are so close, they need to
        # share an aperture
        #for pair in [('1', '1m', +1), ('2', '2m', -1)]:
        #    if (pair[0] in channels) and (pair[1] in channels):
        #        pos = [0, pair[2] * spo.rmid_spo(conf), aper_z]
        #        aperzoom = spo.zoom_to_cover_all_spos(conf, .8, 1.5)
        #        aperzoom[1] *=2
        #        rect = optics.RectangleAperture(position=pos,
        #                                        zoom=aperzoom,
        #                                        orientation=xyz2zxy[:3, :3])
        #        rect.display['outer_factor'] = 2.
        #        apers.append(rect)
        #        channels.remove(pair[0])
        #        channels.remove(pair[1])

        # Now, add individual apertures if they are not double
        # Thus, needs to be geometrically bigger for off-axis sources.
        aperzoom = spo.zoom_to_cover_all_spos(conf, 1.1, 0.8)
        for chan in channels:
            pos = conf['pos_opt_ax'][chan][:3].copy()
            pos[2] += aper_z
            if '1' in chan:
                pos[1] += spo.rmid_spo(conf)
            elif '2' in chan:
                pos[1] -= spo.rmid_spo(conf)
            else:
                raise ValueError('No rules for channel {}'.format(chan))
            pos[2] = aper_z

            rect = optics.RectangleAperture(position=pos,
                                            zoom=aperzoom,
                                            orientation=xyz2zxy[:3, :3])
            rect.display['outer_factor'] = 1.3
            apers.append(rect)

        super(Aperture, self).__init__(elements=apers, **kwargs)


class SimpleSPOs(Sequence):

    def __init__(self, conf, channels=channels,
                 **kwargs):
        rot180 = transforms3d.euler.euler2mat(np.pi, 0, 0, 'szyx')
        # Make lens a little larger than aperture, otherwise an non on-axis ray
        # (from pointing jitter or an off-axis source) might miss the mirror.
        mirror = []
        for chan in channels:
            entrancepos = conf['pos_opt_ax'][chan][:3].copy()
            entrancepos[2] += 12000
            rot = np.eye(3) if '1' in chan else rot180
            mirror.append(SPOChannelMirror(position=entrancepos,
                                           orientation=rot,
                                           reflectivity_interpolator=conf['reflectivity_interpolator'],
                                           spo_pos4d=conf['spo_pos4d'],
                                           spo_geom=conf['spo_geom'],
                                           id_num_offset=id_num_offset[chan]))
            mirror.append(ScatterPerChannel(position=entrancepos,
                                            min_id=id_num_offset[chan],
                                            max_id=id_num_offset[chan] + 1000,
                                            inplanescatter=conf['inplanescatter'],
                                            perpplanescatter=conf['perpplanescatter'],
                                            orientation=xyz2zxy[:3, :3]))
        mirror.append(spomounting)
        self.geometricopening = load_number('spos', 'geometricthroughput', 'transmission') * \
            load_number('spos', 'porespecifications', 'transmission')
        mirror.append(GlobalEnergyFilter(filterfunc=lambda e: self.geometricopening,
                                         name='SPOgeometricthrougput'))

        tab = load_table('spos', 'lossterm')
        en = tab['energy'].to(u.keV, equivalencies=u.spectral())
        mirror.append(GlobalEnergyFilter(filterfunc=interp1d(en, tab[tab.colnames[1]]),
                                         name='Loss_from_nonideal_spos'))

        super(SimpleSPOs, self).__init__(elements=mirror, **kwargs)


class CATGratings(Sequence):
    def __init__(self, conf, channels=channels, **kwargs):

        elements = []

        for chan in channels:
            shift_matrix = np.eye(4)
            shift_matrix[:, 3] = conf['pos_opt_ax'][chan][:]

            elements.append(CATfromMechanical(pos4d=shift_matrix,
                                              channel=chan, conf=conf,
                                              id_num_offset=id_num_offset[chan],
                                              ))
        elements.append(catsupportbars)
        super(CATGratings, self).__init__(elements=elements, **kwargs)


class FiltersAndQE(Sequence):
    '''
    Parameters
    ----------
    conf, channels : any
        Parameters are accepted for consistency with other elements in
        Arcus, but currently, they are not needed here because all relevant
        settings are hardcoded.
    kwargs_interp1d : dict
        keywords for the filter function, for example to allow values outside
        the range available in the configuration files:
        `{'bounds_error'=False}`.
    '''

    filterlist = [('filters', 'sifilter'),
                  ('filters', 'opticalblocking'),
                  ('filters', 'uvblocking'),
                  ('detectors', 'contam'),
                  ('detectors', 'qe')]

    def get_filter(self, directory, name):
        tab = load_table(directory, name)
        en = tab['energy'].to(u.keV, equivalencies=u.spectral())
        return GlobalEnergyFilter(filterfunc=interp1d(en, tab[tab.colnames[1]],
                                                      **self.kwargs_interp1d),
                                  name=name)

    def __init__(self, conf=None, channels=None, kwargs_interp1d={}, **kwargs):
        self.kwargs_interp1d = kwargs_interp1d
        elems = [self.get_filter(*n) for n in self.filterlist]
        super(FiltersAndQE, self).__init__(elements=elems, **kwargs)


class DetMany(rowland.RectangularGrid):

    y_range=[-400, 400]

    def __init__(self, conf, **kwargs):
        super(DetMany, self).__init__(rowland=conf['rowland_detector'],
                                      elem_class=self.elem_class,
                                      elem_args=self.elem_args,
                                      d_element=self.d_element,
                                      y_range=self.y_range,
                                      # convention is to start counting at 1
                                      id_num_offset=1,
                                      id_col='CCD',
                                      guess_distance=-25.)


class Det16(DetMany):
    '''Place only hand-selected 16 CCDs'''
    def __init__(self, conf, theta=[3.1255, 3.1853, 3.2416, 3.301], **kwargs):
        self.theta = theta
        super(Det16, self).__init__(conf, **kwargs)
        assert len(self.elements) == 16
        # but make real detectors orange
        disp = deepcopy(self.elements[0].display)
        disp['color'] = 'orange'
        for e in self.elements:
            e.display = disp


class DetCamera(ParallelCalculated):
    '''CCD detectors in camera layout

    8 CCDs in each strip, with gaps between them in 3-2-3 groups
    to match the shadows from the filter support.
    One strip is offset by a few mm so that the chip gaps fall
    into slightly different positions.
    '''
    d_fsupport = 4.
    '''Distance between CCDs under a filter support bar in mm'''

    d_ccd = 2.15
    '''Minimal distance between CCDs in mm'''

    offset = [-5., 5.]
    '''offset of one strip vs the other to avoid matching chip gaps in mm'''

    id_col = 'CCD'

    def __init__(self, conf):
        r = conf['rowland_detector'].r
        phi_m = np.arcsin(conf['d'] / r) + np.pi
        ccd = conf['det_kwargs']['elem_args']['zoom'][1]
        p0 = conf['phi_det_start']
        gaps = np.array([0, self.d_ccd, self.d_fsupport, self.d_ccd,
                         self.d_ccd, self.d_fsupport,
                         self.d_ccd, self.d_ccd])
        theta1 = (p0 + ccd / r) + np.arange(8) * 2 * ccd / r + gaps.cumsum() / r
        theta = np.hstack([phi_m - theta1 - self.offset[0] / r,
                           phi_m + self.offset[1] / r + theta1])
        # Sort so that CCD number increases from -x to +x
        theta.sort()
        theta = theta[::-1]
        pos = conf['rowland_detector'].parametric(theta, 0)

        super().__init__(pos_spec=pos,
                         normal_spec=conf['rowland_detector'].normal,
                         parallel_spec=conf['rowland_detector'].pos4d @ np.array([0, 1, 0, 0]),
                         **conf['det_kwargs'],
                         )
        assert len(self.elements) == conf['n_CCDs']
        # but make real detectors orange
        disp = deepcopy(self.elements[0].display)
        disp['color'] = 'orange'
        for e in self.elements:
            e.display = disp


class FocalPlaneDet(marxs.optics.FlatDetector):
    loc_coos_name = ['detfp_x', 'detfp_y']
    detpix_name = ['detfppix_x', 'detfppix_y']

    def __init__(self, **kwargs):
        if ('zoom' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['zoom'] = [.2, 10000, 10000]
        if ('orientation' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['orientation'] = xyz2zxy[:3, :3]
        super(FocalPlaneDet, self).__init__(**kwargs)


class PerfectArcus(Sequence):
    '''Default Definition of Arcus without any misalignments'''
    aper_class = Aperture
    spo_class = SimpleSPOs
    #gratings_class = RegularGrid
    gratings_class = CATGratings
    filter_and_qe_class = FiltersAndQE
    '''Set any of these classes to None to not have them included.
    (e.g. SIXTE does filters and QE itself).
    '''

    list_of_classes = ['aper_class', 'spo_class', 'gratings_class',
                       'filter_and_qe_class']

    def add_boom(self, conf):
        '''Add four sided boom. Only the top two bays contribute any
        absorption, so we can save time by not modelling the remaining bays.'''

        return []
        #return [boom.FourSidedBoom(orientation=xyz2zxy[:3, :3],
        #                           position=[0, 0, 596.],
        #                           boom_dimensions={'start_bay': 6})]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__
        because all detectors need different parameters. Placing this
        specific code in it's own function makes it easy to override
        for derived classes.

        '''
        # rotate such that phi=0 is at the bottom
        rot = transforms3d.axangles.axangle2mat(np.array([0, 1, 0]), np.pi)
        circdet = CircularDetector(orientation=xyz2zxy[:3, :3] @ rot,
                                   zoom=[defaultconf['rowland_detector'].r,
                                         defaultconf['rowland_detector'].r,
                                         20],
                                   position=[0, 0, np.sqrt(defaultconf['rowland_detector'].r**2 - conf['d']**2)],
                                   )
        circdet.display['opacity'] = 0.1
        circdet.detpix_name = ['circpix_x', 'circpix_y']
        circdet.loc_coos_name = ['circ_phi', 'circ_y']
        circproj = marxs.analysis.ProjectOntoPlane(orientation=xyz2zxy[:3, :3])
        circproj.loc_coos_name = ['projcirc_x', 'projcirc_y']
        reset = marxs.simulator.simulator.Propagator(distance=-100.)
        twostrips = DetCamera(conf)
        proj = marxs.analysis.ProjectOntoPlane(orientation=xyz2zxy[:3, :3])
        detfp = FocalPlaneDet()
        return [circdet, circproj, reset, twostrips, proj, detfp]

    def post_process(self):
        self.KeepPos = KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=defaultconf, channels=channels,
                 **kwargs):
        elem = []
        for c in self.list_of_classes:
            cl = getattr(self, c)
            if cl is not None:
                elem.append(cl(conf, channels))
        elem.extend(self.add_boom(conf))
        elem.extend(self.add_detectors(conf))
        elem.append(tagversion)
        super(PerfectArcus, self).__init__(elements=elem,
                                           postprocess_steps=self.post_process(),
                                           **kwargs)


class Arcus(PerfectArcus):
    def __init__(self, conf=defaultconf, channels=channels,
                 **kwargs):
        super(Arcus, self).__init__(conf=conf, channels=channels, **kwargs)
        for row in conf['alignmentbudget']:
            elem = self.elements_of_class(row[0])
            if row[1] == 'global':
                tol.moveglobal(elem, *row[3])
            elif row[1] == 'individual':
                tol.wiggle(elem, *row[3])
            elif row[1] == 'individual_with_tilt':
                wiggle_and_bar_tilt(elem, *row[3])
            else:
                raise NotImplementedError('Alignment error {} not implemented'.format(row[1]))


class ArcusForPlot(PerfectArcus):
    '''Arcus with setting that are good for 3D plots

    In particular, there is a full boom and no large catch-all focal plane.
    '''
    def add_boom(self, conf):
        return [boom.FourSidedBoom(orientation=xyz2zxy[:3, :3],
                                   position=[0, 0, 546.])]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__
        because all detectors need different parameters. Placing this
        specific code in it's own function makes it easy to override
        for derived classes.

        '''
        return [DetCamera(conf)]


class ArcusForSIXTE(Arcus):
    filter_and_qe_class = None

    def add_detectors(self, conf):
        return [FocalPlaneDet()]

# Note: We are inventing a new format here.
# - individual -> wiggle
# - global -> moveglobal
# When revising, might be better to put functions directly in there to
# avoid confusion.

align_requirement_smith = [
    [SPOChannelMirror, 'individual', [20, 100, 500, 300, 300, 10],
     None, 'individual SPO in petal'],
    # The following term is for the alignment of each channel
    # (SPO + CAT petal) relative to the front assembly, i.e. SPO and
    # CAT petal are moved together.
    # To make this work, MARXS needs relative position uncertainties
    # which, at present, are not implemented.
    # However, these numbers are smaller than "Camera to front assembly"
    # and for simulations of just one channel would have the same effect.
    # Thus, neglecting them for now is a only a small contribution.
    #[spo.SPOChannelMirror, 'global', [2000., 800, 800, 180, 180, 180],
    # None, 'SPO petal to front assembly'],
    [CATfromMechanical, 'global', [1000, 1000, 1000, 300, 300, 600],
     None, 'CAT petal to SPO petal'],
    [CATfromMechanical, 'individual', [1000, 1000, 200, 300., 180, 300],
     None, 'CAT windows to CAT petal'],
    [CATWindow, 'individual', [1000, 1000, 200, 300, 180, 300],
     None, 'individual CAT to window'],
    [DetCamera, 'global', [5000, 2000, 1000, 180, 180, 180],
     None, 'Camera to front assembly']]
'''This is taken from LSF-CAT-Alignment-v17.xls from R. Smith'''

align_requirement = deepcopy(align_requirement_smith)
reformat_randall_errorbudget(align_requirement)

defaultconf['alignmentbudget'] = align_requirement
