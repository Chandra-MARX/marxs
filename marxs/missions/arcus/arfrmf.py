# Licensed under GPL version 3 - see LICENSE.rst
import os
from os.path import join as pjoin
import string

import numpy as np
import astropy.units as u
from astropy.table import Table, QTable
from scipy.interpolate import interp1d
from sherpa.models import NormGauss1D

from marxs.missions.arcus.utils import config
from marxs.missions.arcus.arcus import defaultconf, DetCamera, FiltersAndQE
from marxs.missions.arcus.load_csv import load_table
from marxs.missions.arcus.utils import tagversion
from marxs.reduction import ogip


def filename_from_meta(filetype='fits', **kwargs):
    '''Generate default filename from ARF or RMF header values

    Parameters
    ----------
    filetype : string
        file ending (the file is usually saved as fits, but the ending
        might be "arf", "rmf", or any other string)
    kwargs: string/int
        Keywords used to construct filename
    '''
    chan = kwargs.pop('ARCCHAN')
    if chan == '1111':
        chan = 'all'  # special case for readability
    filename = f'chan_{chan}'
    return filename + ogip.filename_from_meta(filetype=filetype, **kwargs)


class OnCCD:
    '''Class that calculates if a wavelength falls on a CCD or not

    Some of the Arcus parameters are taken from the dictionary, others (such as
    the position of the CCDs are taken from loading in the Arcus simulator
    and inspecting its objects.

    Parameters
    ----------
    conf : dict
        Dictionary with Arcus design parameters
    '''
    def __init__(self, conf):
        self.conf = conf
        camera = DetCamera(conf)
        self.corners = np.zeros((len(camera.elements), 2, 2))

        for i, ccd in enumerate(camera.elements):
            self.corners[i, 0, :] = (ccd.geometry['center'] -
                                     ccd.geometry['v_y'])[[0, 2]]
            self.corners[i, 1, :] = (ccd.geometry['center'] +
                                     ccd.geometry['v_y'])[[0, 2]]
        facets = load_table('gratings', 'facets')
        self.effective_grating_period = np.cos(conf['blazeang'] * u.degree) * np.mean(facets['period']) * facets['period'].unit
        # without the blaze angle would look like this
        # effective_grating_period = np.mean(facets['period']) * \
        #    facets['period'].unit

    def __call__(self, channel_mid, order, opt_ax):
        '''Mirror and grating efficiency

        This function interpolates the mirror and grating efficiency on
        a given energy grid.

        Parameters
        ----------
        channel_mid : `~astropy.units.quantity.Quantity`
            Representative energy for each channel. Grating efficiency etc. are
            interpolated on this energy, so bins should be narrow enough to
            resolve any features.
        order : ind
            Diffraction order
        opt_ax : string
            Name of the optical axes
        conf : dict
            Arcus configuration

        Returns
        -------
         transmission : np.array
            Transmission in the range 0..1 for each bin.
         '''
        opt_ax_x = self.conf['pos_opt_ax'][opt_ax][0]
        # np.sign accounts for the fact that some orders are
        # reversed in direction
        alpha = np.arctan2(np.sign(opt_ax_x) *
                           (self.corners[:, :, 0] - opt_ax_x),
                           self.conf['f'] - self.corners[:, :, 1]) * u.rad
        theta = np.arcsin(order * u.dimensionless_unscaled *
                          channel_mid.to(u.Angstrom,
                                         equivalencies=u.spectral()) /
                          self.effective_grating_period)
        onanyccd = np.zeros(len(theta), dtype=bool)

        for row in alpha:
            ind = (theta > min(row)) & (theta < max(row))
            onanyccd[ind] = True
        return onanyccd


class FiltersQE(FiltersAndQE):
    def __call__(self, en_mid):
        # Make it look like a photon table so that I can call MARXS element
        tab = Table(data=[en_mid, np.ones(len(en_mid))],
                    names=['energy', 'probability'])
        tab = super().__call__(tab)
        return tab['probability']


class MirrGrat:
    def __init__(self, aefforder,
                 kind='quadratic', fill_value=0., bounds_error=False):
        # Bug in astropy <=4.2:
        # need to wrap into extra QTable or sort will fail
        aefforder = QTable(aefforder, copy=True)
        aefforder.sort("wave")
        self.interp = {}
        self.unit = {}
        for o in aefforder.colnames[1:]:
            self.interp[o] = interp1d(aefforder['wave'], aefforder[o],
                                      kind=kind, fill_value=fill_value,
                                      bounds_error=bounds_error)
            # interp1d does not conserve unit, so need to keep separately
            self.unit[o] = aefforder[o].unit

    def __call__(self, channel_mid, order):
        '''Mirror and grating efficiency

        This function interpolates the effective areas in the caldb.

        Parameters
        ----------
        channel_mid : `~astropy.units.quantity.Quantity`
            Representative energy for each channel. Grating efficiency etc. are
            interpolated on this energy, so bins should be narrow enough to
            resolve any features.
        order : ind
            Diffraction order

        Returns
        -------
        transmission : np.array
            Transmission in the range 0..1 for each bin.
        '''
        transmission = self.interp[str(order)](channel_mid.to(u.Angstrom,
                                               equivalencies=u.spectral()))

        # Just to prevent unphysical extrapolation
        return np.clip(transmission, 0, None) * self.unit[str(order)]


def mkarf(channel_edges, order,
          mirr_grat=None, trans_filters_qe=None,
          conf=defaultconf,
          channels=list(defaultconf['pos_opt_ax'].keys())):
    '''Make an ARF for 0th order or a grating spectrum

    Parameters
    ----------
    channel_edges : `~astropy.units.quantity.Quantity`
        Edges of the channels in energy or wavelength. The ARF that is
        generated from  n edges will have n-1 bins.
    order : int
        Diffraction order
    mirr_grat : callable or None
        Function that interpolates mirror and grating efficiency. If None,
        it is read from ``mirr_grat.tab`` in the caldb.
    trans_filters_qe : callable or None
        Function that interpolates filter curves and CCD QE. If None,
        it is initialized from `FiltersQE` which uses Arcus defaults and
        allows extrapolation.
    conf : dict
        Arcus configuration
    channels : list
        List of channel names for an ARF. ARFs can be made for a single order
        or be combined for a set of orders.

    Returns
    -------
    transmission : np.array
        Transmission in the range 0..1 for each bin.
    '''
    energy_edges = channel_edges.to(u.keV, equivalencies=u.spectral())
    energy_edges = np.sort(energy_edges)
    en_mid = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    onccd = OnCCD(conf)

    if mirr_grat is None:
        aefforder = QTable.read(os.path.join(config['data']['caldb_inputdata'], 'aeff',
                                     'mirr_grat.tab'), format='ascii.ecsv')
        mirr_grat = MirrGrat(aefforder)

    if trans_filters_qe is None:
        trans_filters_qe = FiltersQE(kwargs_interp1d={'bounds_error': False,
                                       'fill_value': 0.})
    specresp_filtqe = trans_filters_qe(en_mid)
    specresp_sim = mirr_grat(en_mid, order)
    all_onccd = np.zeros(len(specresp_filtqe))
    for chan in channels:
        if chan not in defaultconf['pos_opt_ax'].keys():
            raise ValueError(f'channel {chan} not defined in Arcus')
        specresp_onccd = onccd(en_mid, order, chan)
        all_onccd += specresp_onccd

    arf = ogip.ARF(data=[energy_edges[:-1], energy_edges[1:],
                    specresp_sim * specresp_filtqe * all_onccd,
                    specresp_sim, specresp_filtqe, all_onccd,
                    energy_edges[1:].to(u.Angstrom,
                                        equivalencies=u.spectral()),
                    energy_edges[:-1].to(u.Angstrom,
                                         equivalencies=u.spectral()),
                    ],
              names=['ENERG_LO', 'ENERG_HI', 'SPECRESP', 'MIRROR_GRAT',
                     'FILTERS_QE', 'ONCCD',
                     # CIAO adds these to grating ARFs, so we do, too:
                     'BIN_LO', 'BIN_HI',
                     ])
    arf.meta['ARCCHAN'] = ''.join(np.isin(list(conf['pos_opt_ax'].keys()),
                                          channels).astype(int).astype(str))
    tagversion(arf, ORDER=order)
    return arf


## Can I replace this with ogiprmffrom_Gauss_sigma?
# I wrote it for that purpose, but need to compare output to make sure it works that same way.
def mkrmf0(bin_edges, *, ccdfwhm: QTable, threshold=1e-6):
    '''Make RMF for 0 order

    The RMF is a single Gaussian.

    Parameters
    ----------
    channel_edges : `~astropy.units.quantity.Quantity`
        Edges of the channels in energy or wavelength. The RMF that is
        generated from n edges will have n-1 bins.
    threshold : float
        To reduce the file size, RMF components below the threshold value
        are cut-off
    ccdfwhm : `astropy.table.Table`
        Table with CCD properties. Two columns are expected with
        'energy' (in keV) and 'FWHM' (in eV). Note that the units are
        hardcoded and different between columns.
    '''
    ebounds = ogip.RMF.ebounds_from_edges(bin_edges)
    matrix = Table(names=['ENERG_LO', 'ENERG_HI', 'N_GRP',
                          'F_CHAN', 'N_CHAN', 'MATRIX'],
                   data=[np.zeros(len(ebounds), dtype=d) for d in [
                       np.float32, np.float32, np.int16,
                       np.object, np.object, np.object]],
                   units=[u.keV, u.keV, None, None, None, None])

    matrix['ENERG_LO'] = ebounds['E_MIN']
    matrix['ENERG_HI'] = ebounds['E_MAX']
    matrix.meta['DETCHANS'] = len(bin_edges) - 1
    matrix.meta['LO_THRES'] = threshold
    matrix.meta['TLMIN{}'.format(matrix.colnames.index('F_CHAN') + 1)] = 0
    ebounds.meta['TLMIN{}'.format(ebounds.colnames.index('CHANNEL') + 1)] = 0

    for tab in [ebounds, matrix]:
        tab = tagversion(tab, ORDER=0)

    for r, miden in enumerate(0.5 * (ebounds['E_MIN'] + ebounds['E_MAX'])):
        func = NormGauss1D('name')
        func.pos = miden
        func.FWHM = np.interp(miden, ccdfwhm['energy'],
                              ccdfwhm['FWHM'].to(u.keV))  # eV to keV
        fullmatrix = func(ebounds['E_MIN'], ebounds['E_MAX'])
        out = ogip.RMF.arr_to_rmf_matrix_row(fullmatrix, 0,
                                             threshold=threshold)
        for i, col in enumerate(['N_GRP', 'F_CHAN', 'N_CHAN', 'MATRIX']):
            matrix[col][r] = out[i]
    return ogip.RMF(matrix, ebounds, CHANTYPE='PHA', HDUCLAS3='REDIST')


def write_readme(osip, outpath, outroot='', ARCCHAN='all'):
        '''Write README file to directory with ARFs

        Parameters
        ----------
        osip : `marxs.reduction.osip.OSIPBase` object
            OSIP objet that generates data in this directory.
        outpath : string
            Location where the output ARFs are deposited
        outroot : string
            prefix for output filename
        '''
        # Get a table so that I can tag it and get all the meta information
        tag = Table()
        tagversion(tag)
        # Format the meta information into a string
        tagstring = ''
        for k, v in tag.meta.items():
            if isinstance(v, str):
                tagstring += f'{k}: {v}\n'
            else:
                tagstring += f'{k}: {v[0]}   // {v[1]}\n'

        with open(pjoin(os.path.dirname(__file__),
                        'data', "osip_template.md")) as t:
            template = string.Template(t.read())

        output = template.substitute(tagversion=tagstring,
                                     description=f'{osip.__class__}: {osip.osip_description}',
                                     offsetorders=osip.offset_orders,
                                     filenamemain=filename_from_meta(filetype='arf', ARCCHAN=ARCCHAN, ORDER=-4, CCDORDER=-4, TRUEORD=-4),
                                     filenameupper=filename_from_meta(filetype='arf', ARCCHAN=ARCCHAN, ORDER=-4, CCDORDER=-4, TRUEORD=-5),
                                     filenamelower=filename_from_meta(filetype='arf', ARCCHAN=ARCCHAN, ORDER=-4, CCDORDER=-4, TRUEORD=-3))
        with open(pjoin(outpath, "README.md"), "w") as f:
            f.write(output)
