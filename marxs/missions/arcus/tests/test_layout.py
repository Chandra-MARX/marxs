# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import pytest
import astropy.units as u

from marxs.missions.arcus.arcus import (PerfectArcus,
                                        defaultconf as conf)
from marxs.missions.arcus.defaults import DefaultSource, DefaultPointing
from marxs.missions.arcus.utils import id_num_offset
from ..analyze_grid import zeropos

@pytest.mark.xfail(reason="Arcus configuration is changing, so this might fail.")
def test_layout_to_EdHertz():
    '''Compare derived torus tilt angle to number calculated by
    Ed Hertz and Casey.'''
    assert np.isclose(np.rad2deg(np.sin(conf['rowland_central'].pos4d[2, 1])), 2.89214, rtol=1e-3)

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
    '''Test the zeropos function'''
    ind = photons['aper_id'] == chan_name.index(channel) + 1
    assert zeropos(col, channel, conf) == pytest.approx(np.mean(photons[col][ind]), rel=1e-2)

def test_arcus_zeropos():
    '''Apply the zeropos function to test the Arcus zero order positions'''
    assert np.abs(zeropos('proj_y', '1', conf) - zeropos('proj_y', '1m', conf)) > 4
    assert np.abs(zeropos('proj_y', '2', conf) - zeropos('proj_y', '1', conf)) > 4
    assert np.abs(zeropos('proj_y', '2', conf) - zeropos('proj_y', '2m', conf)) > 4
