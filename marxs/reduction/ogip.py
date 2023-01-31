# Licensed under GPL version 3 - see LICENSE.rst
'''
Try to be lenient in reading, i.e. don't check mandatory keywords.
Will read files that have the correct columns, even if HDUVER or similar
keywords are missing. Need to be stricter?
'''
from warnings import warn
import numpy as np
from astropy.table import QTable, Column
from astropy.io import fits
import astropy.units as u


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
    #  convert to string. Will happen anyway when writing to fits
    for k in ['ORDER', 'CCDORDER', 'TRUEORD']:
        #  np.integer is not a subclass of int, so need to test both
        if (k in kwargs) and isinstance(kwargs[k], (int, np.integer)):
            kwargs[k] = f'{kwargs[k]:+}'

    if 'CCDORDER' in kwargs:
        filename = f'_ccdord_{kwargs["CCDORDER"]}_true_{kwargs["TRUEORD"]}'
    else:
        filename = f'_{kwargs["ORDER"]}'

    filename += f'.{filetype}'
    return filename


class OGIPFormatError(Exception):
    pass


class ColOrKeyColumn(Column):
    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if np.all(self.data == self.data[0]):
            self._parent_table().meta[self.name] = self.data[0]
            self._parent_table().remove_column(self.name)
        else:
            if self.name not in self._parent_table().colnames:
                self._parent_table().columns[self.name] = self
                if self.name in self._parent_table().meta:
                    del self._parent_table().meta[self.name]


class ColOrKeyTable(QTable):
    Column = ColOrKeyColumn

    def __getitem__(self, item):
        if isinstance(item, str) and item in self.meta:
            if item in self.colnames:
                warn(f'{item} is both in meta dict and a column.')
            return self.meta[item]
        else:
            return super().__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, str) and (np.ndim(value) == 0 or
                                      np.all(np.asanyarray(value) == value[0])):
            if np.ndim(value) == 0:
                self.meta[item] = value
            else:
                self.meta[item] = value[0]
            if item in self.colnames:
                self.remove_column(item)
        else:
            super().__setitem__(item, value)


def _check_mandatory_keywords(tab, extra_keywords=[]):
    keywords = ['TELESCOP', 'INSTRUME', 'FILTER'] + extra_keywords
    ind = np.isin(keywords, list(tab.meta))
    if not np.all(ind):
        raise OGIPFormatError('Mandatory keywords missing from meta: {}'.format(np.array(keywords)[~ind]))


def _check_col_and_type(tab, req):
    for col, dtype, ndim in req:
        if not ((col in tab.colnames) or (col in tab.meta)):
            raise OGIPFormatError(f'Required column {col} missing.')
        cancast = [np.can_cast(tab[col].dtype, d, 'same_kind') for d in dtype]
        if not any(cancast):
            raise OGIPFormatError(f'dtype for {col} must be {dtype}.')
        if ((ndim is None) or
            ((ndim == 1) and (col in tab.meta)) or
            (tab[col].ndim == ndim)):
            pass
        else:
            raise OGIPFormatError(f'column {col} must have {ndim} dimensions.')


class ARF(ColOrKeyTable):
    required_cols = [('ENERG_LO', (np.float64,), 1),
                     ('ENERG_HI', (np.float64,), 1),
                     ('SPECRESP', (np.float64,), 1)]

    def write(self, *args, **kwargs):
        self.meta.update({'EXTNAME': 'SPECRESP',
                          'HDUCLASS': 'OGIP',
                          'HDUVERS':  '1.1.0',
                          'HDUCLAS1': 'RESPONSE',
                          'HDUCLAS2': 'SPECRESP',
                          })
        _check_mandatory_keywords(self)
        super().write(*args, **kwargs, format='fits')

    @classmethod
    def read(cls, *args, **kwargs):
        self = super().read(*args, format='fits', **kwargs)
        _check_col_and_type(self, self.required_cols)
        return self


class RMF:
    '''Read and represent OGIP RMF data

    Sherpa has a similar class (in fact with more properties)
    I just want to make sure I understand everything that goes into it,
    so I make that myself here.

    Parameters
    ----------
    CHANTYPE : 'PHA', 'PI' or None
        If not None, set the CHANTYPE keyword to PHA (should be used if
        possible) or PI. `None` indicates that the CHANTYPE is set already.
    HDUCLAS3 : 'REDIST', 'DETECTOR', 'FULL' or None
        Set HDUCLAS3 to the value given. If none, do not set.
        Allowed values for HDUCLAS3 are:
           - 'REDIST' for a matrix whose elements represent probabilities
             associated with the photon redistribution process only
           - 'DETECTOR' for a matrix whose elements have been multipled by
             all energy-dependent effects associated with detector
             (eg detector efficiency, window transmission etc).
           - 'FULL' for a matrix whose elements have been multipled by all
             energy-dependent effects associated with detector, optics,
             collimator, filters etc.

    Parameters
    ----------
    rmffile : filename
    '''
    # Note: Some dimesions are set to None, which means that checking
    # must be done separately.
    # If dtype is an object (a list for a variable length array), we don't
    # check what's inside that object.
    matrix_required_cols = [('ENERG_LO', (np.float64,), 1),
                            ('ENERG_HI', (np.float64,), 1),
                            ('N_GRP', (np.int32,), 1),
                            ('F_CHAN', (np.integer, object), None),
                            ('N_CHAN', (np.integer, object), None),
                            ('MATRIX', (np.float64, object), None)]
    ebounds_required_cols = [('CHANNEL', (np.integer,), 1),
                             ('E_MIN', (np.float64,), 1),
                             ('E_MAX', (np.float64,), 1)]

    def __init__(self, matrix, ebounds, CHANTYPE=None, HDUCLAS3=None):
        self.matrix = matrix
        self.ebounds = ebounds
        for m in [self.matrix, self.ebounds]:
            m.meta['DETCHANS'] = len(m)
            if CHANTYPE is not None:
                m.meta['CHANTYPE'] = CHANTYPE
            if HDUCLAS3 is not None:
                m.meta['HDUCLAS3'] = HDUCLAS3
        _check_col_and_type(self.matrix, self.matrix_required_cols)
        _check_col_and_type(self.ebounds, self.ebounds_required_cols)
        _check_mandatory_keywords(self.matrix, extra_keywords=['CHANTYPE'])
        _check_mandatory_keywords(self.ebounds, extra_keywords=['CHANTYPE'])

    @classmethod
    def read(cls, rmffile):
        matrix = ColOrKeyTable.read(rmffile, format='fits', hdu='MATRIX')
        ebounds = ColOrKeyTable.read(rmffile, format='fits', hdu='EBOUNDS')
        _check_col_and_type(matrix, cls.matrix_required_cols)
        _check_col_and_type(ebounds, cls.ebounds_required_cols)
        for tab in [matrix, ebounds]:
            _check_mandatory_keywords(tab, ['CHANTYPE', 'DETCHANS'])
        return cls(matrix, ebounds)

    @staticmethod
    def ebounds_from_edges(channel_edges):
        '''
        Parameters
        ----------
        waveedges : `~astropy.units.quantity.Quantity`
            Edges of channels
        '''
        energy_edges = channel_edges.to(u.keV,
                                        equivalencies=u.spectral()).value
        sort_energies = np.argsort(energy_edges)
        ebounds = ColOrKeyTable({'CHANNEL': np.arange(len(channel_edges) - 1),
                                 'E_MIN': energy_edges[sort_energies][:-1],
                                 'E_MAX': energy_edges[sort_energies][1:]})
        for col in ['E_MIN', 'E_MAX']:
           ebounds[col].unit = u.keV
        return ebounds

    def write(self, rmffile, overwrite=False, TLMIN=None):
        '''
        Paramters
        ---------
        TLMIN : int or `None`
            If `None` then the TLMIN# keywords shoud already be set, if not,
            if is a convenient way to set them on writing.
       '''
        _check_col_and_type(self.matrix, self.matrix_required_cols)
        _check_col_and_type(self.ebounds, self.ebounds_required_cols)
        matrix_tlmin = 'TLMIN{}'.format(self.matrix.colnames.index('F_CHAN') + 1)
        eb_tlmin = 'TLMIN{}'.format(self.ebounds.colnames.index('CHANNEL') + 1)
        if TLMIN is not None:
            self.matrix.meta[matrix_tlmin] = TLMIN
            self.ebounds.meta[eb_tlmin] = TLMIN
        _check_mandatory_keywords(self.matrix, ['CHANTYPE', 'DETCHANS',
                                                matrix_tlmin])
        _check_mandatory_keywords(self.ebounds, ['CHANTYPE', 'DETCHANS',
                                                 eb_tlmin])
        self.matrix.meta['EXTNAME'] = 'MATRIX'
        self.matrix.meta['HDUCLASS'] = 'OGIP'
        self.matrix.meta['HDUVERS'] = '1.1.0'
        self.matrix.meta['HDUCLAS1'] = 'RESPONSE'
        self.matrix.meta['HDUCLAS2'] = 'RSP_MATRIX'
         # futher required keywords: CHANTYPE (PI or PHA), DETCHANS, TLMIN# (# is column number of F_CHAN)
        # Recommended keywords: NUMGRP, NUMELT


        # EBOUNDS
        self.ebounds.meta['EXTNAME'] = 'EBOUNDS'
        self.ebounds.meta['HDUCLASS'] = 'OGIP'
        self.ebounds.meta['HDUCLAS1'] = 'RESPONSE'
        self.ebounds.meta['HDUCLAS2'] = 'EBOUNDS'
        self.ebounds.meta['HDUVERS'] = '1.2.0'

        # Here I'm assuming that if MATRIX is an object (a list) then
        # F_CHAN and N_CHAN are also. That's the case when this class
        # is used ot make them, but might not be true in general if
        # read in from a file.
        if self.matrix['MATRIX'].dtype == np.object:
            self.variable_length_to_fixed_length()

        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.table_to_hdu(self.matrix),
                                fits.table_to_hdu(self.ebounds)])

        hdulist.writeto(rmffile, overwrite=overwrite, checksum=True)

    def row(self, energy):
        rowind = (energy > self.matrix['ENERG_LO']) & \
            (energy < self.matrix['ENERG_HI'])
        rows = rowind.nonzero()[0][0]
        return self.matrix[rows]

    @property
    def en_mid(self):
        return 0.5 * (self.ebounds['E_MIN'] + self.ebounds['E_MAX'])

    def full_rmf(self, energy):
        '''Return rmf over the full range of channels

        Parameters
        ----------
        energy : `astropy.quantity.Quantity`

        Returns
        -------
        e_mid : array
            mid-points of energy bins in keV
        rmf : array
            rmf value in each bin
        '''
        rmf = np.zeros(len(self.en_mid))
        row = self.row(energy)
        tlmin = self.matrix.meta['TLMIN{}'.format(self.matrix.colnames.index('F_CHAN') + 1)]
        for i in range(row['N_GRP']):
            # Python is 0 indexed, FITS is 1 indexed
            chans = slice(row['F_CHAN'][i] - tlmin,
                          row['F_CHAN'][i] + row['N_CHAN'][i] - tlmin)
            matindex = np.cumsum(np.concatenate(([0], row['N_CHAN'])))
            mat = slice(matindex[i], matindex[i + 1])
            rmf[chans] = row['MATRIX'][mat]
        return self.ebounds['E_MIN'], self.ebounds['E_MAX'], rmf

    def rmf(self, energy):
        '''Return rmf over channels where values is non-zero

        Parameters
        ----------
        energy : `astropy.quantity.Quantity`

        Returns
        -------
        e_min, e_max : array
            min and max of energy bins in keV
        rmf : array
            rmf value in each bin
        '''
        row = self.row(energy)
        ind = np.concatenate([np.arange(row['F_CHAN'][i] - 1,
                                        row['F_CHAN'][i] + row['N_CHAN'][i]-1)
                              for i in range(row['N_GRP'])])
        return self.ebounds['E_MIN'][ind], self.ebounds['E_MAX'][ind], \
            np.asanyarray(row['MATRIX'])[ind]

    def rmf_angtrom(self, energy):
        en_lo, en_hi, rmf = self.rmf(energy)
        return en_hi.to(u.Angstrom, equivalencies=u.spectral()), en_lo.to(u.Angstrom, equivalencies=u.spectral()), rmf

    @staticmethod
    def arr_to_rmf_matrix_row(arr, TLMIN_F_CHAN, threshold=1e-6):
        '''Split an array into channel groups for an RMF

        The format of an RMF is set by the [OGIP cal/gen/92-002]_.
        For efficiency, not the entire RMF matrix is stored in the
        rmf files, but only the non-zero components. So, for each
        row (each row contains the matrix for one energy channel)
        the elements that are zero or below a threshold are removed
        from the matrix. This leaves one or more "channel subsets"
        and extra index arrays are needed to specify the position
        and lengths of these channel subsets in the full array.

        [OGIP cal/gen/92-002]_ discusses how arrays can be stored
        as fixed-length or as variable-length arrays in a fits file.
        Since this function operates on a single row only, it return
        lists. Formatting in a fixed-length array, if needed, has
        to be done separately.

        References
        ----------
        .. [OGIP cal/gen/92-002] https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html

        Parameters
        ----------
        arr : `np.array`
            RMF respsonse matrix over all channels of the isntrument
        threshold : float
            Array values below the threshold will be removed.

        Returns
        -------
        N_GRP : int
            Number of channel subsets
        F_CHAN : list
            First channel in each subset (1 indexed)
        N_CHAN : list
            Number of channels in each subset
        MATRIX : list
            Matrix elements
        '''
        ind = arr > threshold
        # Add False at beginning to end, so that np.diff can be
        # used to detect changes to and from True even if the first
        # and / or last value in the row is True
        nfchan = np.diff(np.concatenate(([False], ind, [False]))).nonzero()[0]
        start_end = nfchan.reshape((-1, 2))
        nchan = start_end[:, 1] - start_end[:, 0]
        #
        fchan = start_end[:, 0] + TLMIN_F_CHAN
        assert nchan.shape == fchan.shape
        return len(fchan), list(fchan), list(nchan), list(arr[ind])

    def variable_length_to_fixed_length(self):
        '''Convert variable length arrays to fixed length arrays

        Astropy cannot write variable length arrays
        (https://github.com/astropy/astropy/issues/11323 and linked isues).
        Until that is fixed, all arrays need to be converted to fixed length
        for writing.

        Note: In many cases, this will only moderately increase the file size
        but speed up operations significantly, so it might be a good idea
        anyway.
        '''
        m = self.matrix
        # int16 is recommended in OGIP, but too small for e.g. Arcus
        # So, determine smallest inttype that works
        maxvalf = np.concatenate(m['F_CHAN']).max()
        maxvaln = np.concatenate(m['N_CHAN']).max()
        maxval = max(maxvalf, maxvaln)
        for inttype in [np.int16, np.int32, np.int64]:
            if inttype(maxval) == maxval:
                break
        fchan = np.zeros((len(m), np.max(m['N_GRP'])), dtype=inttype)
        nchan = np.zeros_like(fchan)
        matrix = np.zeros((len(m), max([sum(r) for r in m['N_CHAN']])),
                          dtype=np.float32)
        for i, r in enumerate(m):
            fchan[i, :len(r['F_CHAN'])] = r['F_CHAN']
            nchan[i, :len(r['N_CHAN'])] = r['N_CHAN']
            matrix[i, :len(r['MATRIX'])] = r['MATRIX']
        # if max(N_GRP) == 1 then we can squueze out the first dimension
        # and write a number per row instead of an array.
        m['F_CHAN'] = fchan.squeeze()
        m['N_CHAN'] = nchan.squeeze()
        m['MATRIX'] = matrix
