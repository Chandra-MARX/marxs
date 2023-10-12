# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import astropy.units as u
import pytest

from marxs.missions.arcus.arcus import (PerfectArcus,
                                        defaultconf as conf)
from marxs.missions.arcus.defaults import DefaultSource, DefaultPointing
from marxs.missions.arcus.utils import id_num_offset, config
from ..analyze_grid import zeropos


class DirectLight(PerfectArcus):
    '''Default Definition of Arcus without any misalignments'''
    gratings_class = None

pointing = DefaultPointing()
instrument = DirectLight()
source = DefaultSource(energy=1. * u.keV)
photons = source.generate_photons(40_000 * u.s)
photons = pointing(photons)
photons = instrument(photons)
n_apertures = len(list(id_num_offset.values()))
photons['aper_id'] = np.digitize(photons['xou'],
                                     bins=list(id_num_offset.values()))
chan_name = list(id_num_offset.keys())

@pytest.mark.parametrize('channel', ['1', '1m', '2', '2m'])
@pytest.mark.parametrize('col',['proj_x', 'proj_y', 'circ_phi', 'circ_y'])
def test_zeropos(channel, col):
    ind = photons['aper_id'] == chan_name.index(channel) + 1
    assert zeropos(col, channel, conf) == pytest.approx(np.mean(photons[col][ind]), rel=1e-2)