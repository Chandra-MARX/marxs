# Licensed under GPL version 3 - see LICENSE.rst
import re
import datetime

import numpy as np
from scipy import interpolate
from astropy.table import Column, Table
import astropy.units as u
from sherpa.models import NormGauss1D, Scale1D
from sherpa.astro.models import Lorentz1D

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


# Is this used anywhere? Usually, do it from spline.
# Can always convert CALDB to spline and then use that.
# So I think this is just a second implementation of the same thing
# which should be be here because now I have to debug etc. twice.
def caldb2sherpa(caldb, row, iwidth, iwave):
    '''Convert CALDB entries to Sherpa model

    Convert entries in a CALDB lsfparm files to a Sherpa model.  The
    lsfparm files can contain several rows, for different off-axis
    angles and radii and in each row there will be entries for a
    number of wavelength points and extraction width.

    This function expects as input the index numbers for row,
    wavelength as index for the wavelength array etc. In practical
    applications, the CALDB file will be queried for a specific
    position, wavelength etc., but for development purposes it is
    useful to go into the raw array, e.g. to read some unrelated CALDB
    file (say for a different detector) to use as a starting point to
    fit the lsfparm parameters or to plot different settings for
    comparison.

    caldb : `astropy.table.Table`
        CALDB lsfparm table

    '''
    model = []
    for col in caldb.colnames:
        if col == 'EE_FRACS':
            eef = Scale1D(name='EE_FRACS')
            eef.c0 = caldb['EE_FRACS'][row][iwidth, iwave]
            # model is underdetermined if this the ampl of all functions
            # is left free
            eef.c0.frozen = True
        elif gaussn.match(col):
            newg = NormGauss1D(name=col)
            newg.ampl, newg.fwhm, newg.pos = caldb[col][row][iwidth, iwave, :] * gconv
            model.append(newg)
        elif lorentzn.match(col):
            newg = Lorentz1D(name=col)
            newg.ampl, newg.fwhm, newg.pos = caldb[col][row][iwidth, iwave, :] * lconv
            model.append(newg)

    sumampl = np.sum([m.ampl.val for m in model if isinstance(m, NormGauss1D) or isinstance(m, Lorentz1D)])
    for m in model:
        if isinstance(m, NormGauss1D) or isinstance(m, Lorentz1D):
            m.ampl.val = m.ampl.val / sumampl
    # Start value is 0, unless we explicitly set the start value. So, split models and pass [0]
    # as start value to avoid a numerical 0.0 in the model expression.
    return eef * sum(model[1:], model[0])


def flatten_sherpa_model(model):
    if hasattr(model, 'parts'):
        modellist = []
        for p in model.parts:
            modellist.extend(flatten_sherpa_model(p))
        return modellist
    else:
        return [model]


def sherpa2caldb(shmodel):
    '''
    shmodel : Sherpa model instance
    '''
    d = {}
    for comp in flatten_sherpa_model(shmodel):
        if isinstance(comp, NormGauss1D):
            modelpars = np.array([comp.ampl.val, comp.fwhm.val,
                                  comp.pos.val]) / gconv
        elif isinstance(comp, Lorentz1D):
            modelpars = np.array([comp.ampl.val, comp.fwhm.val,
                                  comp.pos.val]) / lconv
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
    model : Sherpa moden instance
        This instance is inspected for the name of the model components to
        set up the correct columns
    '''
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


class interp1d_2dsignature(interpolate.interp1d):
    '''1D interp with the same signature as `scipy.interpolate.interp2d`

    For use with LSFPARM files that have only one WIDTH.

    '''
    def __init__(self, x, y, z, **kwargs):
        super().__init__(y, z, **kwargs)

    def __call__(self, x, y):
        return super().__call__(y)


def CALDB_interp(row, kind='cubic', **kwargs):
    '''Get interpolating functions from values in LSFPARM

    Parameters
    ----------
    row : `astropy.table.Row`
        Table row from CALDB LSFPARM file
    **kwargs
        All other keywords arguments are passed to
        `scipy.interpolate.interp1d` or `scipy.interpolate.interp2d`.
    '''
    interpolators = {}

    if row['NUM_WIDTHS'] > 1:
        interp = interpolate.interp2d
    else:
        interp = interp1d_2dsignature

    for col in row.colnames:
        if col == 'EE_FRACS':
            interpolators[col] = interp(row['WIDTHS'],
                                        row['LAMBDAS'], row[col],
                                        kind=kind,
                                        **kwargs)
        elif modelnames.match(col):
            interpolators[col] = [interp(row['WIDTHS'],
                                         row['LAMBDAS'],
                                         row[col][:, :, i].squeeze(),
                                         kind=kind,
                                         **kwargs)
                                  for i in [0, 1, 2]]
    return interpolators


def sherpa_from_spline(splines, width, wave):
    '''
    To-Do: Convert to correct unit before calling .value
    To-Do: np.abs is really just there if an interpolation
      goes below 0. Setting to 0 or linearly interpolating
      would be better, but I simply want a fast solution right now.
    '''
    model = []
    for col in splines:
        if col == 'EE_FRACS':
            eef = Scale1D(name='EE_FRACS')
            eef.c0 = splines[col](width.value, wave.value)
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
                np.abs(np.stack([splines[col][i](width.value, wave.value)
                          for i in [0, 1, 2]]).flatten() * conv)
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
    '''
    makes PHA channels (because that's what the lsfparm files are made for)
    I made a choice here: writing with TLMIN=0, which is valid OGIP
    '''
    ebounds = RMF.ebounds_from_edges(wave_edges)
    matrix = Table(names=['ENERG_LO', 'ENERG_HI', 'N_GRP',
                          'F_CHAN', 'N_CHAN', 'MATRIX'],
                   data=[np.zeros(len(ebounds), dtype=d) for d in [
                       np.float32, np.float32, np.int16,
                       np.object, np.object, np.object]],
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
