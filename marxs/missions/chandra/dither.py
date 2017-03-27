# Licensed under GPL version 3 - see LICENSE.rst

import numpy as np
from astropy.units import Quantity
from astropy.table import Table
import astropy.units as u

from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat
from transforms3d.axangles import axangle2mat

from ...source import FixedPointing
from .fitsheaders import complete_header

class LissajousDither(FixedPointing):
    '''Lissajous dither pattern with adjustable parameters.

    Parameters
    ----------
    DitherAmp : `astropy.units.Quantity`
        (pitch, yaw, roll) dither amplitude
    DitherPeriod : `astropy.units.Quantity`
        (pitch, yaw, roll) dither Period in sec
    DitherPhase : `astropy.units.Quantity`
        (pitch, yaw, roll) dither phase at ``time = 0`` in radian
    '''
    def __init__(self, **kwargs):
        self.DitherAmp = kwargs.pop('DitherAmp', np.array([8., 8., 0.]) * u.arcsec)
        self.DitherPeriod = kwargs.pop('DitherPeriod', np.array([1000., 707., 1e5]) * u.s)
        self.DitherPhase = kwargs.pop('DitherPhase', np.zeros(3) * u.radian)
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
        return self.DitherAmp * np.sin(2. * np.pi * u.radian * time[:, np.newaxis] * u.s / self.DitherPeriod + self.DitherPhase)  / np.array([np.cos(self.coords.icrs.dec), 1., 1.])


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

        Returns
        -------
        pointing : (n, 3) np.array
            Ra, Dec, roll values in radian for the pointing direction at time t.
        '''
        nominal = Quantity([self.coords.ra, self.coords.dec,
                            self.roll])
        dither = self.dither(time)
        # roll in astronomical system is defined opposite of the usual mathematical angle
        # because the ra in the coordinate system increases in the other direction.
        roll = nominal[2] + dither[:, 2]
        # Express directions as x,y,z vectors
        phi = nominal[0]
        theta = np.pi/2. * u.rad - nominal[1]
        e_nominal = np.array([np.sin(phi) * np.sin(theta),
                              np.cos(phi) * np.sin(theta),
                              np.cos(theta)])
        phi = nominal[0] + dither[:, 0]
        theta = np.pi/2. * u.rad - (nominal[1] + dither[:, 1])
        e_dither = np.vstack([np.sin(phi) * np.sin(theta),
                              np.cos(phi) * np.sin(theta),
                              np.cos(theta)]).T
        pointing_dir = np.zeros_like(dither.value)

        # common case for Chandra
        if np.allclose(roll.to(u.rad).value, roll[0].to(u.rad).value):
            mat = axangle2mat(e_nominal, -roll[0].to(u.rad).value, is_normalized=True)
            constant_roll = True

        for i in range(len(time)):
            if not constant_roll:
                mat = axangle2mat(e_nominal, -roll[i].to(u.rad).value, is_normalized=True)
            pointing_dir[i, :] = np.dot(mat, e_dither[i, :].value)

        # convert x,y,z pointing back to ra, dec, roll
        pointing = np.vstack([np.arctan2(pointing_dir[:, 0], pointing_dir[:, 1]) % (2.*np.pi),
                              np.pi / 2. - np.arccos(pointing_dir[:, 2]),
                              roll.to(u.rad).value]).T
        return pointing

    def photons_dir(self, coos, time):
        '''Calculate direction on photons in homogeneous coordinates.

        Parameters
        ----------
        coos : `astropy.coordiantes.SkyCoord`
            Origin of each photon on the sky
        time : np.array
            Time for each photons in sec

        Returns
        -------
        photons_dir : np.array of shape (n, 4)
            Homogeneous direction vector for each photon
        '''
        ra = coos.ra.rad
        dec = coos.dec.rad
        # Minus sign here because photons start at +inf and move towards origin
        pointing = self.pointing(time)
        photons_dir = np.zeros((len(ra), 4))
        photons_dir[:, 0] = - np.cos(dec) * np.cos(ra)
        photons_dir[:, 1] = - np.cos(dec) * np.sin(ra)
        photons_dir[:, 2] = - np.sin(dec)
        for i in range(len(ra)):
            mat3d = euler2mat(pointing[i, 0],
                              - pointing[i, 1],
                              - pointing[i, 2], 'rzyx')

            photons_dir[i, :3] = np.dot(mat3d.T, photons_dir[i, :3])

        return photons_dir


    def write_asol(self, photons, asolfile, timestep=0.256):
        '''Write an aspect solution (asol) file

        Chandra analysis scripts often require an aspect solution, which is essientiall
        a list of pointing directions in the dither pattern vs. time.
        This method write such a list to a file.

        Parameters
        ----------
        photons :  `astropy.table.Table` or `astropy.table.Row`
            Table with photon properties. Some meta data from the header of this table
            is required (e.g. the length of the observation).
        asolfile : string
            Path and file name where the asol file is saved.
        timestamp : float
            Time step between entries in the asol file in seconds.
        '''
        time = np.arange(0, photons.meta['EXPOSURE'][0], timestep)
        pointing = self.pointing(time).to(u.deg).value
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



    # def pointing(self, time):
    #     '''Calculate the pointing direction for a set of times

    #     There are the steps to convert the dither (which are offsets) to the total
    #     pointing direction with the roll properly taken care of.
    #     (See marxasp.c in the marx code for more details.)

    #     1.  Convert ra/dec offsets to absolute pointing.
    #     2.  Roll resulting vector about nominal pointing
    #     3.  Convert result to ra/dec.

    #     Parameters
    #     ----------
    #     time : np.array
    #         Array of times

    #     Results
    #     -------
    #     pointing : (n, 3) np.array
    #         Ra, Dec, roll values in radian for the pointing direction at time t.
    #     '''
    #     dither = self.dither(time).to(u.radian).value
    #     phi = dither[:, 0]
    #     theta = np.pi/2. - dither[:, 1]
    #     e_dither = np.vstack([np.sin(phi) * np.sin(theta),
    #                           np.cos(phi) * np.sin(theta),
    #                           np.cos(theta)]).T
    #     pointing_dir = np.zeros_like(dither)

    #     # common case for Chandra
    #     roll = self.roll.to(u.radian).value
    #     if np.allclose(dither[:, 2], dither[0, 2]):
    #         mat = axangle2mat([1, 0, 0], -dither[0, 2] - roll, is_normalized=True)
    #         constant_roll = True

    #     for i in range(len(time)):
    #         if not constant_roll:
    #             mat = axangle2mat([1, 0, 0], -dither[i, 2] - roll, is_normalized=True)
    #         pointing_dir[i, :] = np.dot(mat, e_dither[i, :])

    #     # convert x,y,z pointing back to ra, dec, roll
    #     pointing = np.vstack([np.arctan2(pointing_dir[:, 0], pointing_dir[:, 1]) % (2.*np.pi),
    #                           np.pi / 2. - np.arccos(pointing_dir[:, 2]),
    #                           dither[:, 2]]).T
    #     return pointing

    # def photons_dir(self, coos, time):
    #     '''Calculate direction on photons in homogeneous coordinates.

    #     Parameters
    #     ----------
    #     coos : `astropy.coordiantes.SkyCoord`
    #         Origin of each photon on the sky
    #     time : np.array
    #         Time for each photons in sec

    #     Returns
    #     -------
    #     photons_dir : np.array of shape (n, 4)
    #         Homogeneous direction vector for each photon
    #     '''
    #     photons_dir = super(LissajousDither, self).photons_dir(coos, time)
    #     # Minus sign here because photons start at +inf and move towards origin
    #     pointing = self.dither(time).to(u.rad).value
    #     for i in range(len(time)):
    #         mat3d = euler2mat(pointing[i, 0],
    #                           - pointing[i, 1],
    #                           - pointing[i, 2], 'rzyx')

    #         photons_dir[i, :3] = np.dot(mat3d.T, photons_dir[i, :3])

    #     return photons_dir


    # def write_asol(self, photons, asolfile, timestep=0.256):
    #     time = np.arange(0, photons.meta['EXPOSURE'][0], timestep)
    #     pointing = np.rad2deg(self.pointing(time))
    #     asol = Table([time, pointing[:, 0], pointing[:, 1], pointing[:, 2]],
    #                  names=['time', 'ra', 'dec', 'roll'],
    #                  )
    #     asol['time'].unit = 's'
    #     # The following columns represent measured offsets in Chandra
    #     # They are not part of this simulation. Simply set them to 0
    #     for col in [ 'ra_err', 'dec_err', 'roll_err',
    #                  'dy', 'dz', 'dtheta', 'dy_err', 'dz_err', 'dtheta_err',
    #                   'roll_bias', 'pitch_bias', 'yaw_bias', 'roll_bias_err', 'pitch_bias_err', 'yaw_bias_err']:
    #         asol[col] = np.zeros_like(time)
    #         if 'bias' in col:
    #             asol[col].unit = 'deg / s'
    #         elif ('dy' in col) or ('dz' in col):
    #             asol[col].unit = 'mm'
    #         else:
    #             asol[col].unit = 'deg'
    #     asol['q_att'] = [mat2quat(euler2mat(np.deg2rad(p[0]),
    #                                np.deg2rad(-p[1]),
    #                                np.deg2rad(-p[2]), 'rzyx'))
    #                      for p in pointing]
    #     # Copy info like the exposure time from the photons list meta to asol,
    #     # but not column specific keywords like TTYPEn, TCTYPn, MTYPEn, MFORMn, etc:
    #     for k in photons.meta:
    #         if (('TYP' not in k) and ('FORM' not in k) and
    #             ('TCR' not in k) and ('TCDEL' not in k) and (k not in asol.meta)):
    #             asol.meta[k] = photons.meta[k]
    #     asol.meta['EXTNAME'] = 'ASPECT'
    #     complete_header(asol.meta, None, 'ACASOL', ['OGIP', 'TEMPORALDATA', 'ASPECT'])
    #     # In MARXS t=0 is the start of the observation, but for Chandra we need to make that
    #     # consistent with the value of the TSTART keyword.
    #     asol['time'] += asol.meta['TSTART'][0]
    #     asol.write(asolfile, format='fits')  # , checksum=True) - works only with fits interface
