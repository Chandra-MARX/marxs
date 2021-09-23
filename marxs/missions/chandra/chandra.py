# Licensed under GPL version 3 - see LICENSE.rst

import numpy as np

from .fitsheaders import complete_header
from ...simulator import Sequence


class Chandra(Sequence):
    '''Main class representing the Chandra X-ray observatory


    '''
    def process_photons(self, photons):
        photons.meta['MISSION'] = ('AXAF', 'Mission')
        photons.meta['TELESCOP'] = ('CHANDRA', 'Telescope')
        return super().process_photons(photons)

    def write_evt(self, photons, filename):
        '''Write a Chandra event level 1 file.

        As opposed to directly saving the photon list, this adds some Chandra
        specific meta data.

        Parameters
        ----------
        photons :  `astropy.table.Table` or `astropy.table.Row`
            Table with photon properties. Some meta data from the header of this table
            is required (e.g. the length of the observation).
        filename : string
            Path and file name where the file is saved.
        '''
        photons.meta['EXTNAME'] = 'EVENTS'
        # rename RA, DEC columns - otherwise CIAO tasks will be confused.
        photons.rename_column('ra', 'marxs_ra')
        photons.rename_column('dec', 'marxs_dec')
        complete_header(photons.meta, photons, 'EVT1', ['OGIP', 'EVENTS', 'ALL'])
        photons.write(filename, format='fits')
        # add_GTIs(filename)
