'''Geometry and operations of the Chandra Observatory.

This module provides the optical elements on the Chandra Observatory.

.. warning:: Currently incomplete.

   Missing:

    - a lot

   Incomplete:

   - Grating order efficiencies are the same for every order.

Some spacecraft properties are defined in module attributes, others are read from
reference files in the CALDB (or the files shipped with `classic marx`).
Properties that are very easy to define and that are very unlikely to ever change
(e.g. the mapping of ACIS chip number to chip name) are set in this module.
That makes is easier to see where the numbers come from and thus helps for users
who look at this module as an example of how to build up complex marxs setups.
'''
import os
from math import cos, sin
from ConfigParser import ConfigParser
import numpy as np

from astropy.table import Table
from transforms3d.utils import normalized_vector as norm_vec
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat
from transforms3d.axangles import axangle2mat

from ... import optics
from ...optics import MarxMirror as HDMA
from ...optics import FlatDetector, FlatGrating, uniform_efficiency_factory
from ...source import FixedPointing
from ...simulator import Sequence, Parallel
from ...math.pluecker import h2e
from .fitsheaders import complete_header
from .data import NOMINAL_FOCALLENGTH, AIMPOINTS, TDET, ODET, PIXSIZE

ACIS_name = ['I0', 'I1', 'I2', 'I3', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5']
'''names of the 10 ACIS chips'''


def chip2tdet(chip, tdet, id_num):
    '''Convert CHIP coordinates to TDET coordiantes.

    See eqn (5) in `the Chandra coordiante memo <http://cxc.harvard.edu/contrib/jcm/ncoords.ps>`_.

    Parameters
    ----------
    chip : (N, 2) np.array
        chip coordiantes
    tdet : dict
        dictionary with definitions for the coordiante conversion.
        See `ACISTDET` for an example.
    id_num : integer
        chip ID number (e.g. ``1`` for ACIS-I1)
    '''
    scale = tdet['scale'][id_num]
    handedness = tdet['handedness'][id_num]
    origin_tdet = tdet['origin'][id_num]
    theta = tdet['theta'][id_num]
    rotation = np.array([[cos(theta), sin(theta)],
                         [-sin(theta), cos(theta)]])
    return scale * handedness * np.dot(rotation, (chip - 0.5).T).T + (origin_tdet + 0.5)


class ACISChip(FlatDetector):

    def __init__(self, **kwargs):
        self.TDET = TDET['ACIS']
        self.ODET = ODET['ACIS']
        self.pixsize_in_rad = np.deg2rad(PIXSIZE['ACIS'])
        super(ACISChip, self).__init__(**kwargs)

    @property
    def chip_name(self):
        return 'AXIS-{0}'.format(ACIS_name[self.id_num])

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # CHIP and TDET are based on pixel coordinates
        # +1 because Chandra pixel convention is 1 based
        chip = intercoos[intersect, :] / self.pixsize + self.centerpix + 1
        tdet = chip2tdet(chip, self.TDET, self.id_num)
        # DET is based on spacecraft coordiantes, but in observations they need to be derived
        # from pixel coordiantes because that's all we have.
        # Here, we already know the spacecraft coordiantes (STF), so we can start from there.
        # I just hope it is consistent and I did not screw up the offsets at some point.
        fc = h2e(interpos[intersect, :])
        mn = fc  # Systems differ only in x-direction
        mn[:, 0] -= NOMINAL_FOCALLENGTH
        x = mn[:, 1] / mn[:, 0] / self.pixsize_in_rad
        y = mn[:, 2] / mn[:, 0] / self.pixsize_in_rad
        detx = self.ODET[0] - x
        dety = self.ODET[1] + y
        theta = np.deg2rad(photons.meta['ROLL_PNT'][0])
        skyx = self.ODET[0] - x * cos(theta) - y * sin(theta)
        skyy = self.ODET[1] - x * sin(theta) + y * cos(theta)

        photons.meta['ACSYS1'] = ('CHIP:AXAF-ACIS-1.0', 'reference for chip coord system')
        photons.meta['ACSYS2'] = ('TDET:{0}'.format(self.TDET['version']), 'reference for tiled detector coord system')
        photons.meta['ACSYS3'] = ('DET:ASC-FP-1.1', 'reference for focal plane coord system')
        photons.meta['ACSYS4'] = ('SKY:ASC-FP-1.1', 'reference for sky coord system')
        return {'chipx': chip[:, 0], 'chipy': chip[:, 1],
                'tdetx': tdet[:, 0], 'tdety': tdet[:, 1],
                'detx': detx, 'dety': dety,
                'x': skyx, 'y': skyy,}

class ACIS(Parallel):
    '''
    Missing:
    - This currently only implements the ideal **detection**, no PHA, no
read-out streaks,
      no pile-up etc.
    - contamination
    '''

    OLSI = np.array([0.684, 0.750, 236.552])
    '''Origin of the LSI in SST coordinates.

    Numbers are taken for the Chandra coordinate memo I at
    http://cxc.harvard.edu/contrib/jcm/ncoords.ps
    '''

    id_col = 'CCD_ID'

    def __init__(self, chips, **kwargs):

        # This stuff is Chandra specific, but applies to HRC, too.
        # Move to a more general class, once the HRC gets implemented.
        self.aimpoint = kwargs.pop('aimpoint')
        self.detoffset = np.array([kwargs.pop(['DetOffsetX'], 0),
                                    0,
                                    kwargs.pop(['DetOffsetY'], 0)])
        # Now the ACIS specific case
        kwargs['elem_pos'] = None
        kwargs['elem_class'] = ACISChip
        kwargs['elem_args'] = {'pixsize': 0.023985 }

        super(ACIS, self).__init__(**kwargs)
        self.chips = chips
        self.elements = [self.elements[i] for i in chips]

    def get_corners(self):
        '''Get the coordinates of the ACIS pixel corners.

        Currently, this method reads the datafile included with MARX, but alternatively if could also
        get this information from CALDB.
        It is very unlikely that the coordinates will ever change, so either location will work.

        Returns
        -------
        out : list
            List of dictionaries with entries 'LL', 'LR', 'UR', and 'UL'. The values of the entries are
            3d coordinates of the chip corner in LSI coordinates.
            There is one dictionary per ACIS chip.
        '''
        conf = ConfigParser()
        # get the basename of the path
        basedir = os.path.dirname(os.path.dirname(os.path.dirname(optics.__file__)))
        conf.read(os.path.join(basedir, 'setup.cfg'))
        marxsrc = conf.get('MARX', 'srcdir')
        t = Table.read(os.path.join(marxsrc, 'marx', 'data', 'pixlib', 'pix_corner_lsi.par'), format='ascii')
        out = []
        for chip in ACIS_name:
            coos = {}
            for corner in ['LL', 'LR', 'UR', 'UL']:
                ind = t['col1'] == 'ACIS-{0}-{1}'.format(chip, corner)
                coos[corner] = np.array([np.float(v) for v in t[ind]['col4'][0][1:-1].split()])
            out.append(coos)
        return out

    def calculate_elempos(self):
        # This stuff is true for HRC, too. Move to more general class, once HRC is implemened.
        corners = self.get_corners()
        pos4d = []
        for i, n in enumerate(ACIS_name):
            # LSI (local science) instrument coordinates
            # notation from coordiante memo: e_x, e_y, e_z is not unit vector!
            p0 = corners[i]['LL']
            e_y = corners[i]['LR'] - p0
            e_z = corners[i]['UL'] - p0
            e_x = np.cross(e_y, e_z)
            center = p0 + 0.5 * e_y + 0.5 * e_z
            # This contains the rotation and the zoom.
            # Note: If I find out that I screwed up the direction of the rotation
            #       I have to treat the zoom separately, because without the .T
            #       it would not work.
            rotlsi = np.vstack([norm_vec(e_x), e_y / 2, e_z / 2]).T
            A = np.eye(4)
            A[:3, :3] = rotlsi
            A[:3, 3] = center
            # Now apply the origin of the LSI system
            A[:3, 3] += self.OLSI
            # see http://cxc.cfa.harvard.edu/proposer/POG/html/chap6.html#tth_chAp6
            A[:3, 3] += self.aimpoint
            # for the coordiante convention - Offset is opposite to direction of coordiante system
            A[:3, 3] -= self.detoffset
            pos4d.append(A)
        return pos4d

    def process_photons(self, photons, *args, **kwargs):
        photons = super(ACIS, self).process_photons(photons, *args, **kwargs)
        photons.meta['SIM_X'] = self.aimpoint[0] - self.detoffset[0]
        photons.meta['SIM_Y'] = self.aimpoint[1] - self.detoffset[1]
        photons.meta['SIM_Z'] = self.aimpoint[2] - self.detoffset[2]
        photons.meta['DETNAM'] = 'ACIS-' + ''.join([str(i) for i in self.chips])
        photons.meta['INSTRUME'] = ('ACIS', 'Instrument')
        return photons

class LissajousDither(FixedPointing):
    '''Lissajous dither pattern with adjustable parameters.

    Parameters
    ----------
    DitherAmp : np.array
        (pitch, yaw, roll) dither amplitude in arcsec (same unit for roll!)
    DitherPeriod : np.array
        (pitch, yaw, roll) dither Period in sec
    DitherPhase : np.array
        (pitch, yaw, roll) dither phase at ``time = 0``
    '''
    def __init__(self, **kwargs):
        self.DitherAmp = kwargs.pop('DitherAmp', np.array([8., 8., 0.]))
        self.DitherPeriod = kwargs.pop('DitherPeriod', np.array([1000., 707., 1e5]))
        self.DitherPhase = kwargs.pop('DitherPhase', np.zeros(3))
        super(LissajousDither, self).__init__(**kwargs)

    def dither(self, time):
        '''Calculate the dither offset relative to pointing direction.

        Parameters
        ----------
        time : np.array
            Time when the dither motion should be calculated

        Returns
        -------
        delta : np.array of shape (N, 3)
            dither motion offset in pitch, yaw, roll for N times in rad
        '''
        return np.deg2rad(self.DitherAmp / 3600.) * np.sin(2. * np.pi * time[:, np.newaxis] / self.DitherPeriod + self.DitherPhase)  / np.array([np.cos(np.deg2rad(self.dec)), 1., 1.])

    def pointing(self, time):
        '''Calculate the pointing direction for a set of times

        There are the steps to convert the dither (which are offsets) to the total
        pointing direction with the roll properly taken care of.
        (See marxasp.c in the marx code for more details.)

        1.  Convert ra/dec offsets to absolute pointing.
        2.  Roll resulting vector about nominal pointing
        3.  Convert result to ra/dec.

        Parameters
        ----------
        time : np.array
            Array of times

        Results
        -------
        pointing : (n, 3) np.array
            Ra, Dec, roll values in radian for the pointing direction at time t.
        '''
        nominal = np.deg2rad(np.array([self.ra, self.dec, self.roll]))
        dither = self.dither(time)
        # roll in astronomical system is defined opposite of the usual mathematical angle
        # because the ra in the coordinate system increases in the other direction.
        roll = - nominal[2] + dither[:, 2]
        # Express directions as x,y,z vectors
        phi = nominal[0]
        theta = np.pi/2. - nominal[1]
        e_nominal = np.array([np.sin(phi) * np.sin(theta),
                              np.cos(phi) * np.sin(theta),
                              np.cos(theta)])
        phi = nominal[0] + dither[:, 0]
        theta = np.pi/2.- (nominal[1] + dither[:, 1])
        e_dither = np.vstack([np.sin(phi) * np.sin(theta),
                              np.cos(phi) * np.sin(theta),
                              np.cos(theta)]).T
        pointing_dir = np.zeros_like(dither)

        # common case for Chandra
        if np.allclose(roll, roll[0]):
            mat = axangle2mat(e_nominal, roll[0], is_normalized=True)
            constant_roll = True

        for i in range(len(time)):
            if not constant_roll:
                mat = axangle2mat(e_nominal, roll[i], is_normalized=True)
            pointing_dir[i, :] = np.dot(mat, e_dither[i, :])

        # convert x,y,z pointing back to ra, dec, roll
        pointing = np.vstack([np.arctan2(pointing_dir[:, 0], pointing_dir[:, 1]) % (2.*np.pi),
                              np.pi / 2. - np.arccos(pointing_dir[:, 2]),
                              roll]).T
        return pointing

    def photons_dir(self, ra, dec, time):
        '''Calculate direction on photons in homogeneous coordinates.

        Parameters
        ----------
        ra : np.array
            RA for each photon in rad
        dec : np.array
            DEC or each photon in rad
        time : np.array
            Time for each photons in sec

        Returns
        -------
        photons_dir : np.array of shape (n, 4)
            Homogeneous direction vector for each photon
        '''
        # Minus sign here because photons start at +inf and move towards origin
        pointing = self.pointing(time)

        photons_dir = np.zeros((len(ra), 4))
        photons_dir[:, 0] = - np.cos(dec) * np.cos(ra)
        photons_dir[:, 1] = - np.cos(dec) * np.sin(ra)
        photons_dir[:, 2] = - np.sin(dec)
        for i in range(len(ra)):
            mat3d = euler2mat(np.deg2rad(pointing[i, 0]),
                              np.deg2rad(-pointing[i, 1]),
                              np.deg2rad(-pointing[i, 2]), 'rzyx')

            photons_dir[i, :3] = np.dot(mat3d.T, photons_dir[i, :3])

        return photons_dir


    def write_asol(self, photons, asolfile, timestep=0.256):
        time = np.arange(0, photons.meta['EXPOSURE'][0], timestep)
        pointing = np.rad2deg(self.pointing(time))
        asol = Table([time, pointing[:, 0], pointing[:, 1], pointing[:, 2]],
                     names=['time', 'ra', 'dec', 'roll'],
                     )
        asol['time'].unit = 's'
        # The following columns represent measured offsets in Chandra
        # They are not part of this simulation. Simply set them to 0
        for col in [ 'ra_err', 'dec_err', 'roll_err',
                     'dy', 'dz', 'dtheta', 'dy_err', 'dz_err', 'dtheta_err',
                      'roll_bias', 'pitch_bias', 'yaw_bias', 'roll_bias_err', 'pitch_bias_err', 'yaw_bias_err']:
            asol[col] = np.zeros_like(time)
            if 'bias' in col:
                asol[col].unit = 'deg / s'
            elif ('dy' in col) or ('dz' in col):
                asol[col].unit = 'mm'
            else:
                asol[col].unit = 'deg'
        asol['q_att'] = [mat2quat(euler2mat(np.deg2rad(p[0]),
                                   np.deg2rad(-p[1]),
                                   np.deg2rad(-p[2]), 'rzyx'))
                         for p in pointing]
        # Copy info like the exposure time from the photons list meta to asol,
        # but not column specific keywords like TTYPEn, TCTYPn, MTYPEn, MFORMn, etc:
        for k in photons.meta:
            if (('TYP' not in k) and ('FORM' not in k) and
                ('TCR' not in k) and ('TCDEL' not in k) and (k not in asol.meta)):
                asol.meta[k] = photons.meta[k]
        asol.meta['EXTNAME'] = 'ASPECT'
        complete_header(asol.meta, None, 'ACASOL', ['OGIP', 'TEMPORALDATA', 'ASPECT'])
        # In MARXS t=0 is the start of the observation, but for Chandra we need to make that
        # consistent with the value of the TSTART keyword.
        asol['time'] += asol.meta['TSTART'][0]
        asol.write(asolfile, format='fits')  # , checksum=True) - works only with fits interface




class Chandra(Sequence):
    '''

    Uses CALDB environment variable for now. Might want to change that to a
    parameter, so that it can be set in a pythonic way.
    '''
    def process_photons(photons):
        photons.meta['MISSION'] = ('AXAF', 'Mission')
        photons.meta['TELESCOP'] = ('CHANDRA', 'Telescope')
        return super(Chandra, self).process_photons(photons)
