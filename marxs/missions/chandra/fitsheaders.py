'''

See `ASC FITS File Designers' Guide ASC-FITS-2.1.0 <cxc.harvard.edu/contrib/arots/fits/ascfits.ps>`.
'''
from astropy import time
import astropy.units as u

from ... import __version__
from .data import TLMINMAX, PIXSIZE, NOMINAL_FOCALLENGTH, ODET

def update_header(header, h):
    for elem in h:
        if elem[0] not in header:
            header[elem[0]] = elem[1]

def complete_CC(header, content, hduclass):
    '''Configuration Control Component'''
    h = [
        ('ORIGIN', 'ASC'),
        ("HDUDOC", "ASC-FITS-2.0: Rots, McDowell: ASC FITS File Designers Guide"),
        ("CONTENT", content),
        ("HDUNAME", header['EXTNAME']),
        ]
    for a, b in zip(['HDUCLASS', 'HDUCLAS1', 'HDUCLAS2', 'HDUCLAS3'], hduclass):
        h += [(a, b)]
    update_header(header, h)


def complete_T(header):
    '''Timing component'''
    now = time.Time.now()
    now.format = 'fits'
    nowexp = now + header['EXPOSURE'][0] * u.s
    h = [
        ('DATE', (now.value[:23], 'Date and time of file creation {0}'.format(now.value[23:]))),
        ('DATE-OBS', (now.value[:23], 'TT with clock correction if CLOCKAPP')),
        ('DATE-END', (nowexp.value[:23], 'TT with clock correction if CLOCKAPP')),
        ("TIMESYS", ("TT", "AXAF time will be Terrestrial Time")),
        ("MJDREF", (50814, "MJD of clock start")),
        ("TIMEZERO", (0, "Clock Correction")),
        ("TIMEUNIT", 's'),
        ("BTIMNULL", (0., "Basic Time offset (s)")),
        ("BTIMRATE", (2.5625000912249E-01, "Basic Time clock rate (s / VCDUcount)")),
        ("BTIMDRFT", (2.1806598193841E-17, "Basic Time clock drift (s / VCDUcount^2)")),
        ("BTIMCORR", (0.0000000000000E+00, "Correction applied to Basic Time rate (s)")),
        ("TIMEREF", ("LOCAL", "Time is local for data")),
        ("TASSIGN", ("SATELLITE", "Source of time assignment")),
        ("CLOCKAPP", (True, "Clock correction applied")),
        ("TIERRELA", (1e-9, "Short term clock stability")),
        ("TIERABSO", (1e-4, "Absolute precision of clock correction")),
        ("TIMVERSN", ("ASC-FITS-2.1", "AXAF Fits design document")),
        ("TIMEPIXR", (0., "Time stamp refers to start of bin")),
        ("TIMEDEL", (3.241, "Time resolution of data in seconds")),
        ]
    update_header(header, h)
    # The following keywords depend on the DATE-OBS found in the header, which may differ from
    # the number calculated above, if it was set to a specific date previously.
    tstart = time.Time(header['DATE-OBS'][0], format='fits')
    tstart.format = 'cxcsec'
    header["TSTART"] = (tstart.value, "As in the TIME column: raw space craft clock;")
    header['TSTOP'] = ((tstart + header['EXPOSURE'][0] * u.s).value, "  add TIMEZERO and MJDREF for absolute TT")
    header['OBS-MJD'] = tstart.mjd


def complete_O(header):
    '''Observation info component'''
    h = [("MISSION", ( "AXAF", "Mission is AXAF")),
         ("TELESCOP", ("CHANDRA", "TELESCOPE is Chandra")),
         ("GRATING", ("NONE", "Grating")),
         ('DATACLASS', ('SIMULATED', 'see http://marxs.rtfd.org')),
         ('ONTIME', (header['EXPOSURE'][0], 'Sum of GTIs')),
         ('DTCOR', (1., 'Dead Time Correction')),
         ('LIVETIME', (header['EXPOSURE'][0], 'Ontime multiplied by DTCOR')),
         ('OBSERVER', ('MARXS', 'This is a simulation.')),
         ('FOC_LEN', (NOMINAL_FOCALLENGTH, 'Assumed focal length')),
         ]


DMKEYWORDS = [('MTYPE1', 'chip'), ('MFORM1', 'chipx,chipy'),
              ('MTYPE2', 'tdet'), ('MFORM2', 'tdetx,tdety'),
              ('MTYPE3', 'det'), ('MFORM3', 'detx,dety'),
              ('MTYPE4', 'sky'), ('MFORM4', 'x,y'),
              ('MFORM5', 'RA,DEC'), ('MTYPE5', 'EQPOS')]
'''CIAO data model (DM) keywords that group columns together'''

def add_evt_column_header(header, data):
    '''Add CIAO keywords to header of an eventfile.'''
    # Clean out column related keywords that may not be valid any longer.
    for k in header.iterkeys():
        if k[:5] in ['TCTYP', 'TCRVL', 'TCDLT', 'TCRPX', 'TLMIN', 'TLMAX']:
            del header[k]
    instr = header['INSTRUME'][0]
    if instr not in TLMINMAX.keys():
        raise KeyError('TLMIN and TLMAX not specified for detector {0}'.format(instr))
    colnamesup = [c.upper() for c in data.colnames]
    if len(set(data.colnames)) != len(set(colnamesup)):
        raise KeyError('Fits files are case insensitive. Column names in data must be unique if converted to upper case.')
    tl = TLMINMAX[instr]
    odet = ODET[instr]
    for i, k in enumerate(data.colnames):
        if k.upper() in tl:
            header['TLMIN{0}'.format(i+1)] = tl[k.upper()][0]
            header['TLMAX{0}'.format(i+1)] = tl[k.upper()][1]
    for k in DMKEYWORDS:
        header[k[0]] = k[1]

    # Turn X,Y into a WCS that e.g. ds9 can interpret
    indx = colnamesup.index('X') + 1
    header['TCTYP{0}'.format(indx)] = 'RA---TAN'
    header['TCRVL{0}'.format(indx)] = header['RA_PNT']
    header['TCDLT{0}'.format(indx)] = -PIXSIZE[instr]  # - because RA increases to left
    header['TCRPX{0}'.format(indx)] = odet[0]
    indy = colnamesup.index('Y') + 1
    header['TCTYP{0}'.format(indy)] = 'DEC---TAN'
    header['TCRVL{0}'.format(indy)] = header['DEC_PNT']
    header['TCDLT{0}'.format(indy)] = PIXSIZE[instr]
    header['TCRPX{0}'.format(indy)] = odet[1]
    header['RADECSYS'] = ('ICRS', 'WCS system')


def complete_header(header, data=None, content=['UNKNOWN'], hduclass='UNKNOWN'):
    '''Complete fits header for Chandra fits files.

    This method add the common keywords that are required for all CXC fit files.
    With few exception, the methods will not overwrite existing keywords, but will
    only fill in the value of required keywords if they have not been set before.
    There are certain keywords that cannot be generated without more information
    (e.g. the name of the detector used). These keywords must be set outside of these
    routines.
    The exception where keywords are overwritten is timing information that is contained in
    redundant keywords. This method read "EXPOSURE" and "DATE-OBS" from the header and
    calculates "TSTART", "TSTOP" and "OBS-MJD" from those values.

    Parameters
    ----------
    header : a dictionary-like object
        In most cases, this will be ``photons.meta``.
    data : `astropy.table.Table` or ``None``
        For event tables the header includes keywords that depend on the order of the columns
        in the data (e.g. column 5 and 6 define the WCS). Pass in the full table as data for
        those cases.
    content : string
        Content keyword as specified by ASC
    hduclass : list of stings
        The list can contain 1-4 elements, depending on the data product.
        See appendix A1 in the
        `ASC FITS File Designers' Guide ASC-FITS-2.1.0 <cxc.harvard.edu/contrib/arots/fits/ascfits.ps>`.
    References
    ----------
    `ASC FITS File Designers' Guide ASC-FITS-2.1.0 <cxc.harvard.edu/contrib/arots/fits/ascfits.ps>`.
    '''
    if content.upper().startswith('EVT'):
        add_evt_column_header(header, data)
    complete_CC(header, content=content, hduclass=hduclass)
    complete_T(header)
    complete_O(header)

def sort_columns(photons):
    '''Clean up the order of column names

    Parameters
    ----------
    photons : `astropy.table.Table`
        Event list.

    Returns
    -------
    photons : `astropy.table.Table`
        Event list with columns sorted such that pairs of columns that describe coordinates
        appear together (e.g. "detx" and "dety"). All other columns are sorted alphabetically.
    '''
    colnames = photons.colnames
    # find those pairs that end on x and y like "tdetx, tdety"
    end_x = set([c[:-1] for c in colnames if c[-1] == 'x'])
    end_y = set([c[:-1] for c in colnames if c[-1] == 'y'])
    endxy = end_x.intersection(end_y)
    # columns that are not x/y pairs. Careful: Could be e.g. "energy" without "energx"
    othercol = set(colnames) - set([c+'x' for c in endxy]) - set([c+'y' for c in endxy])
    # Since we are using sets, the order is now random, but CIAO expects certain columns in order
    # When comparing the names, we have to take care of upper case / lower case.
    # That does not matter when we write it to a fits file, but as long as its an astropy Table
    # column names are case sensitive.
    endxy = list(endxy)
    endxy_l = [c.lower() for c in endxy]
    expected_order = ['chip', 'tdet', 'det', '']
    endxy_ordered = []
    for c in expected_order:
        if c in endxy_l:
            index = endxy_l.index(c)
            endxy_ordered.append(endxy.pop(index))
            endxy_l.pop(index)   # Keep endxy_l to have the same order as endxy

    endxy_ordered.extend(endxy)  # Add additional cols not part of the required order.

    ordered_cols = []
    for c in endxy_ordered:
        ordered_cols.extend([c + 'x', c + 'y'])

    # Other pairs we want ordered if present
    pairs = [('ra', 'dec'), ('RA', 'DEC'), ('ra', 'de'), ('RA', 'DE')]
    for p in pairs:
        if p[0] in othercol and p[1] in othercol:
            othercol -= set(p)
            ordered_cols.extend(p)

    # Now put the rest in repeatable order
    othercol = list(othercol)
    othercol.sort()
    ordered_cols.extend(othercol)
    return photons(ordered_cols)
