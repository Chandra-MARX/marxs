# Licensed under GPL version 3 - see LICENSE.rst
import re
import datetime

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy.table import Column, Table
import astropy.units as u

# We want Sherpa to be an optional dependency, so the imports happen in
# individual functions, not here.
# from sherpa.models import NormGauss1D, Scale1D
# from sherpa.astro.models import Lorentz1D
# from sherpa import stats, optmethods
# from sherpa.fit import Fit
# from sherpa.data import Data1DInt

__all__ = [
    'sherpa2caldb',
    'empty_lsfparmtable', 'lsfparmtable_add_Chandra_header',
    'CALDB_interp', 'sherpa_from_spline',
    'make_rmf',
    'fit_LSF', 'plot_LSFfit',
    ]

from .ogip import RMF

COPY_LSFPARM_KEYWORDS = [
    # OGIP mantatory
    'TELESCOP', 'INSTRUME', 'FILTER',
    # Chandra
    'DETNAM', 'RAND_TG', 'GRATING', 'GRATTYPE', 'TG_M', 'ORDER', 'SHELL',
]

modelnames = re.compile('(GAUSS|LORENTZ)[0-9]+_PARMS')
gaussn = re.compile('GAUSS(?P<n>[0-9]+)_PARMS')
lorentzn = re.compile('LORENTZ(?P<n>[0-9]+)_PARMS')

# convert parameters from CALDB convention to Sherpa
# For NormGauss1d Sherpa uses FWHM, while CALDB uses sigma
gconv = np.array([1, 2 * np.sqrt(2 * np.log(2)), 1])
lconv = np.array([1, 1, 1])


def flatten_sherpa_model(model):
    '''Flatten a Sherpa model into a list to make easy to iterate over

    Parameters
    ----------
    model : `sherpa.models.model.BinaryOpModel`
        Sherpa model consisting of different parts

    Returns
    -------
    modellist : list of `sherpa.models.model.Model`
        Flat list of individual models
    '''
    return [m for m in model if not hasattr(m, 'parts')]


def sherpa2caldb(shmodel):
    '''Get parameters from Sherpa model into tabular CALDB format

    Parameters
    ----------
    shmodel : `sherpa.models.model.BinaryOpModel`
        This model should consists of Gauss and Lorentz components.
        The component model parameters will be extracted into the order
        and naming expected for LSFPARM files in the CALDB.

    Returns
    -------
    d : dict
        Dictionary with model parameters in CALDB format.
    '''
    from sherpa.astro.models import Lorentz1D
    from sherpa.models import NormGauss1D

    d = {}
    for comp in flatten_sherpa_model(shmodel):
        if isinstance(comp, NormGauss1D):
            # convert to np.array to make sure we can divide by gconv
            # but then convert back to list to make sure it's JSON serializable
            modelpars = list(np.array([comp.ampl.val, comp.fwhm.val,
                                  comp.pos.val]) / gconv)
        elif isinstance(comp, Lorentz1D):
            modelpars = list(np.array([comp.ampl.val, comp.fwhm.val,
                                  comp.pos.val]) / lconv)
        else:
            raise Exception(f'Component {comp} not valid for LSFPARM files')
        d[comp.name] = modelpars
    return d


def empty_lsfparmtable(widths, waves, model, extname, order):
    '''
    Parameters
    ----------
    width : np.array
        extraction width
    waves
    model : Sherpa model instance
        This instance is inspected for the name of the model components to
        set up the correct columns
    '''
    from sherpa.astro.models import Lorentz1D
    from sherpa.models import NormGauss1D

    tab = Table()
    tab.add_column(Column(data=[len(widths)], name='NUM_WIDTHS',
                          dtype=np.int32))
    tab.add_column(Column(data=[widths], name='WIDTHS', dtype=np.float32,
                          shape=(len(widths)), unit=u.degree))
    tab.add_column(Column(data=[len(waves)], name='NUM_LAMBDAS',
                          dtype=np.int32))
    tab.add_column(Column(data=[waves], name='LAMBDAS',
                          dtype=np.float32, shape=(len(waves)),
                          unit=u.Angstrom))
    # From the ICD: The columns TG LAM LO and TG LAM HI give that lower and
    # upper wavelength of
    # the box that was used to extract the LSF data.
    # I think these columns are informational only, but I need to check that.
    dwave = np.max(np.diff(waves)) + 1.2
    tab.add_column(Column(data=[waves - dwave], name='TG_LAM_LO',
                          dtype=np.float32, shape=(len(waves)),
                          unit=u.Angstrom))
    tab.add_column(Column(data=[waves + dwave], name='TG_LAM_HI',
                          dtype=np.float32, shape=(len(waves)),
                          unit=u.Angstrom))
    tab.add_column(Column(name='EE_FRACS', length=1, dtype=np.float32,
                          shape=(len(widths), len(waves))))
    # See which columns are needed to describe the models
    modelcomps = flatten_sherpa_model(model)
    compnames = [m.name for m in modelcomps]
    # Check names are unique and follow DPID naming scheme
    if not len(set(compnames)) == len(compnames):
        raise ValueError("Names of model components are not unique (using default names?)")
    # Check models are right type and follow DPID naming rules:
    for m in modelcomps:
        if not isinstance(m, (NormGauss1D, Lorentz1D)):
            raise ValueError('Model can only contain Sherpa NormGauss1D and Lorentz1D components.')
    # Check model and name match
    # Check count starts at 1 (not 0), consequtive numbers
    for m in compnames:
        tab.add_column(Column(name=m + '_PARMS', length=1, dtype=np.float32,
                              shape=(len(widths), len(waves), 3)))

    # Add header keywords.
    # TODO: Not done yet, just adding what I can to make mkrmf read it
    # Not sure for all keywords what they mean or should be
    tab.meta['EXTNAME'] = extname
    tab.meta['CCLS0001'] = 'CPF'
    tab.meta['CDTP0001'] = 'DATA'
    tab.meta['CCNM0001'] = 'LSFPARM'
    tab.meta['RAND_TG'] = '0.0'   # make this a parameter
    tab.meta['LONGSTRN'] = 'OGIP 1.0'
    tab.meta['CREATOR'] = 'lsfparm.py'  # update
    tab.meta['DATE'] = str(datetime.datetime.now()).replace(' ', 'T')[:19],
    tab.meta['CONTENT'] = 'CDB_' + tab.meta['EXTNAME'] + '_LSFPARM'
    tab.meta['HDUCLASS'] = 'ASC',
    tab.meta['HDUCLAS1'] = 'PARAMETERS'
    tab.meta['HDUCLAS2'] = 'PSF'
    tab.meta['HDUCLAS3'] = 'LSF'
    tab.meta['TG_M'] = order  # not sure what goes here. Check existing files!
    tab.meta['ORDER'] = order
    tab.meta['DETNAM'] = 'UNKNOWN'
    tab.meta['INSTRUME'] = 'UNKNOWN'
    tab.meta['CVSD0001'] = '1999-07-22T00:00:00'  # Chandra as an example
    tab.meta['CVST0001'] = '00:00:00'
    tab.meta['TELESCOP'] = 'UNKOWN'
    tab.meta['FILTER'] = 'UNKNOWN'
    return tab


def lsfparmtable_add_Chandra_header(tab, evt):
    tab.meta['CDES0001'] = tab.meta['EXTNAME'] + \
        ' line spread function, input for mkgrmf'
    tab.meta['DETNAM'] = evt.meta['DetectorType']
    tab.meta['INSTRUME'] = evt.meta['DetectorType'].split('-')[0]
    tab.meta['CVSD0001'] = '1999-07-22T00:00:00'
    tab.meta['CVST0001'] = '00:00:00'
    tab.meta['TELESCOP'] = 'CHANDRA'
    tab.meta['FILTER'] = 'NONE'
    tab.meta['GRATING'] = evt.meta['GratingType']
    tab.meta['GRATTYPE'] = tab.meta['EXTNAME'] if evt.meta['GratingType']=='HETG' else ''  # Check value for LETG

    tab.meta['SHELL'] = {'MEG': '1100', 'HEG': '0011', 'LEG': '1111'}[tab.meta['EXTNAME']]  # put right name here for LEG
    tab.meta['CBD10001'] = 'ORDER({})'.format(tab.meta['ORDER'])
    tab.meta['CBD20001'] = 'RAND_TG({})'.format(tab.meta['RAND_TG'])
    tab.meta['CBD30001'] = 'SHELL({})'.format(tab.meta['SHELL'])
    tab.meta['CBO10001'] = 'GRATING({})'.format(tab.meta['GRATING'])
    tab.meta['CBO20001'] = 'GRATTYPE({})'.format(tab.meta['GRATTYPE'])  # check LEG if present at all
    tab.meta['CBO30001'] = 'TG_M({})'.format(tab.meta['TG_M'])
    # No idea what those mean...
    #('FDLT0001', 2.0),
    #('CAL_QUAL', 0),

    return tab


def CALDB_interp(row, method='linear', bounds_error=False,
                 fill_value=None):
    '''Get interpolating functions from values in LSFPARM

    Parameters
    ----------
    row : `astropy.table.Row`
        Table row from CALDB LSFPARM file
    method : str
        Interpolation method, passed to `scipy.interpolate.RegularGridInterpolator`
    bounds_error : bool
        Passed to `scipy.interpolate.RegularGridInterpolator`
    fill_value : float
        Passed to `scipy.interpolate.RegularGridInterpolator`
    '''
    interpolators = {}

    for col in row.colnames:
        if col == 'EE_FRACS':
            interpolators[col] = RGI((row['WIDTHS'], row['LAMBDAS']),
                                     row[col],
                                     method=method,
                                     bounds_error=bounds_error,
                                     fill_value=fill_value)
        elif modelnames.match(col):
            interpolators[col] = [RGI((row['WIDTHS'], row['LAMBDAS']),
                                      row[col][:, :, i].squeeze(),
                                      method=method,
                                      bounds_error=bounds_error,
                                      fill_value=fill_value)
                                  for i in [0, 1, 2]]
    return interpolators


def sherpa_from_spline(interpolators, width, wave):
    '''Make a Sherpa model function from CALDB LSFPARM parameters

    Parameters
    ----------
    interpolators : dict
        Dictionary with interpolating functions for each column in the
        LSFPARM table
    width : `astropy.units.Quantity`
        Extraction width
    wave : `astropy.units.Quantity`
        Wavelength

    Returns
    -------
    model : `sherpa.models.model.Model`
        Sherpa model with parameters set to the LSFPARM values
    '''
    from sherpa.models import NormGauss1D, Scale1D
    from sherpa.astro.models import Lorentz1D

    width_deg = width.to(u.deg).value
    wave_ang = wave.to(u.Angstrom).value

    model = []
    for col in interpolators:
        if col == 'EE_FRACS':
            eef = Scale1D(name='EE_FRACS')
            eef.c0 = interpolators[col]((width_deg, wave_ang))
            # model is underdetermined if this the ampl of all functions
            # is left free
            eef.c0.frozen = True
        elif gaussn.match(col) or lorentzn.match(col):
            if gaussn.match(col):
                new = NormGauss1D(name=col)
                conv = gconv
            else:
                new = Lorentz1D(name=col)
                conv = lconv
            new.ampl, new.fwhm, new.pos = \
                np.stack([interpolators[col][i]((width_deg, wave_ang))
                          for i in [0, 1, 2]]).flatten() * conv
            model.append(new)

    sumampl = np.sum([m.ampl.val for m in model
                      if isinstance(m, NormGauss1D) or
                      isinstance(m, Lorentz1D)])
    for m in model:
        if isinstance(m, NormGauss1D) or isinstance(m, Lorentz1D):
            m.ampl.val = m.ampl.val / sumampl
    # Start value is 0, unless we explicitly set the start value.
    # So, split models and pass [0]
    # as start value to avoid a numerical 0.0 in the model expression.
    return eef * sum(model[1:], model[0])


def make_rmf(lsfparmrow, wave_edges, width, threshold=1e-6, kw_interp={}):
    '''Make an RMF from a CALDB LSFPARM input

    Generate an RMF in PHA channels. This routine generates RMFs with
    TLMIN=0, which is valid OGIP.

    Parameters
    ----------
    lsfparmrow : `astropy.table.Row`
        Table row from CALDB LSFPARM file
    wave_edges : `astropy.units.Quantity`
        Wavelength edges for the RMF
    width : `astropy.units.Quantity`
        Extraction width
    threshold : float
        Threshold for RMF matrix
    kw_interp : dict
        Keyword arguments passed to `CALDB_interp`

    Returns
    -------
    rmf : `marxs.reduction.ogip.RMF`
        RMF object
    '''
    ebounds = RMF.ebounds_from_edges(wave_edges)
    matrix = Table(names=['ENERG_LO', 'ENERG_HI', 'N_GRP',
                          'F_CHAN', 'N_CHAN', 'MATRIX'],
                   data=[np.zeros(len(ebounds), dtype=d) for d in [
                       np.float32, np.float32, np.int16,
                       object, object, object]],
                   units=[u.keV, u.keV, None, None, None, None])

    matrix['ENERG_LO'] = ebounds['E_MIN']
    matrix['ENERG_HI'] = ebounds['E_MAX']
    matrix.meta['DETCHANS'] = len(wave_edges) - 1
    matrix.meta['LO_THRES'] = threshold
    matrix.meta['TLMIN{}'.format(matrix.colnames.index('F_CHAN') + 1)] = 0
    ebounds.meta['TLMIN{}'.format(ebounds.colnames.index('CHANNEL') + 1)] = 0
    for tab in [ebounds, matrix]:
        for k in COPY_LSFPARM_KEYWORDS:
            if k in lsfparmrow.meta:
                tab.meta[k] = lsfparmrow.meta[k]

    ang_lo = matrix['ENERG_HI'].to(u.Angstrom,
                                   equivalencies=u.spectral())
    ang_hi = matrix['ENERG_LO'].to(u.Angstrom,
                                   equivalencies=u.spectral())
    midwave = 0.5 * (ang_lo + ang_hi)

    splines = CALDB_interp(lsfparmrow, **kw_interp)
    for r, wav in enumerate(midwave):
        func = sherpa_from_spline(splines, width, wav)
        # We want sherpa in increasing wavelength, so reverse order
        fullmatrix = func(ang_lo[::-1], ang_hi[::-1])
        # but then we want the RMF in increasing energy, so we reverse again
        out = RMF.arr_to_rmf_matrix_row(fullmatrix[::-1], 0,
                                        threshold=threshold)
        for i, col in enumerate(['N_GRP', 'F_CHAN', 'N_CHAN', 'MATRIX']):
            matrix[col][r] = out[i]
    return RMF(matrix, ebounds, CHANTYPE='PHA', HDUCLAS3='REDIST')


def fit_LSF(evt, model, wavebin=0.001, d_wave=0.03, colname='tg_mlam',
                stat=None, method=None):
    '''Fit an LSF model to a simulated set of photons

    Photons are binned into a histogram and the model is fit to that.
    If a "probability" column is given in the event list,
    the histogram it will contain non-integer data, which does not
    follow a Poisson distribution.
    Unfortunately, the statistical error in each bin is not well defined,
    but some number is needed for the fit. So, the square root of the
    number of photons in each bin (with a floor of 1) is used.
    This is not ideal, but it gives satisfactory results for typical cases.

    evt : `astropy.table.Table`
        Table with photons.
        This table must already be filtered to contain only photons
        that are part of the LSF (e.g. filter on order and wavelength
        before calling this function).
    model : `sherpa.models.model.Model`
        LSF model to be fit.
    wavebin : float
        Wavelength bin size for histogram
    d_wave : float
        Half-width of wavelength range to fit
    colname : str
        Name of column in evt that contains the wavelength. The default
        name is taken from the Chandra terminaology.
    stat : `sherpa.stats.Stat` or None
        If None, use `sherpa.stats.Cash`
    method : `sherpa.optmethods.OptMethod` or None
        If None, use `sherpa.optmethods.LevMar`

    Returns
    -------
    sdata : `sherpa.data.Data1D`
        Data object with the histogram
    res : `sherpa.fit.FitResults`
        Results of the fit
    '''
    from sherpa.fit import Fit
    from sherpa.data import Data1DInt
    from sherpa import stats, optmethods
    from sherpa.models import NormGauss1D, Scale1D
    from sherpa.astro.models import Lorentz1D

    if stat is None:
        stat = stats.Cash()
    if method is None:
        method = optmethods.LevMar()
    probability = evt['probability'] if 'probability' in evt.colnames else None

    hist, edges = np.histogram(evt[colname].value, weights=probability,
                               bins=evt[colname].value.mean() + np.arange(- d_wave, d_wave, wavebin))
    # Avoid error of 0 in the uncertainty,
    # setting a minimum to 1.
    # Better to run longer simulations to get more photons, but need to
    # avoid numerical error if it happens.
    sdata = Data1DInt('counts_histogram', edges[:-1], edges[1:], hist,
                      staterror=np.sqrt(hist.clip(1, np.inf)))
    modfit = Fit(sdata, model, stat=stat, method=method)
    res = modfit.fit()

    # Normalize functional models to 1 in norm and put all normalization
    # in Scale1D (the EEF)
    sumampl = np.sum([m.ampl.val for m in model
                      if isinstance(m, NormGauss1D) or
                      isinstance(m, Lorentz1D)])
    for m in model:
        if isinstance(m, NormGauss1D) or isinstance(m, Lorentz1D):
            m.ampl.val = m.ampl.val / sumampl
        if isinstance(m, Scale1D):
            m.c0 = m.c0.val * sumampl

    return sdata, res


def plot_LSFfit(sdata, model, axes):
    '''Plot an LSF fit for visual inspection

    Parameters
    ----------
    sdata : `sherpa.data.Data1D`
        Data object with the histogram
    model : `sherpa.models.model.Model`
        LSF model from the fit to be plotted
    axes : list of `matplotlib.axes.Axes`
        The first axis is used linar plot, highligting the fit to the
        line center and the second axis for a log plot, highlighting the
        wings.
    '''
    for ax in axes:
        ax.plot(sdata.x, sdata.y, 'k', label='Data')
        ax.plot(sdata.x, model(sdata.xlo, sdata.xhi), linewidth=2, label='model')
        eef = model.parts[0]
        for m in flatten_sherpa_model(model.parts[1]):
            ax.plot(sdata.x, (eef * m)(sdata.xlo, sdata.xhi), linewidth=2,
                    label=f'EEF * {m.name}')
        ax.set_xlabel('wavelength [Ang]')
        ax.set_ylabel('counts / bin')
    axes[1].set_yscale('log')
    axes[1].set_ylim([1, np.max(sdata.y)])
