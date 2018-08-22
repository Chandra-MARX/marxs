# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Table
from transforms3d.utils import normalized_vector as norm_vec

from ...optics import FlatDetector
from ...simulator import Parallel
from ...math.utils import h2e

from .data import (NOMINAL_FOCALLENGTH, AIMPOINTS, TDET, ODET, PIXSIZE,
                   PIX_CORNER_LSI_PAR, chip2tdet)



ACIS_name = ['I0', 'I1', 'I2', 'I3', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5']
'''names of the 10 ACIS chips'''


class ACISChip(FlatDetector):
    '''A class that defines one chip in the ACIS instrument.'''
    def __init__(self, **kwargs):
        self.TDET = TDET['ACIS']
        self.ODET = ODET['ACIS']
        self.pixsize_in_rad = np.deg2rad(PIXSIZE['ACIS'])
        kwargs['ignore_pixel_warning'] = True
        super(ACISChip, self).__init__(**kwargs)

    @property
    def chip_name(self):
        return 'ACIS-{0}'.format(ACIS_name[self.id_num])

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # CHIP and TDET are based on pixel coordinates
        # +1 because Chandra pixel convention is 1 based
        chip = intercoos[intersect, :] / self.pixsize + self.centerpix + 1
        tdet = chip2tdet(chip, self.TDET, self.id_num)
        # DET is based on spacecraft coordiantes, but in observations they need to be derived
        # from pixel coordinates because that's all we have.
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
        skyx = self.ODET[0] - x * np.cos(theta) + y * np.sin(theta)
        skyy = self.ODET[1] + x * np.sin(theta) + y * np.cos(theta)

        photons.meta['ACSYS1'] = ('CHIP:AXAF-ACIS-1.0', 'reference for chip coord system')
        photons.meta['ACSYS2'] = ('TDET:{0}'.format(self.TDET['version']), 'reference for tiled detector coord system')
        photons.meta['ACSYS3'] = ('DET:ASC-FP-1.1', 'reference for focal plane coord system')
        photons.meta['ACSYS4'] = ('SKY:ASC-FP-1.1', 'reference for sky coord system')
        return {'chipx': chip[:, 0], 'chipy': chip[:, 1],
                'tdetx': tdet[:, 0], 'tdety': tdet[:, 1],
                'detx': detx, 'dety': dety,
                'x': skyx, 'y': skyy,}

class ACIS(Parallel):
    '''The ACIS instrument

    Missing:

    - This currently only implements the ideal **detection**, no PHA, no read-out streaks,
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
        kwargs['elem_pos'] = self.calculate_elempos()
        kwargs['elem_class'] = ACISChip
        # Use 0.024 because that's more consistent with 1024 pix
        kwargs['elem_args'] = {'pixsize': 0.024 } # {'pixsize': 0.023985 }

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
        t = Table.read(PIX_CORNER_LSI_PAR, format='ascii')
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
