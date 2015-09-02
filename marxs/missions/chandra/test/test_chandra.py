import os
import numpy as np

from astropy.table import Table

from ... import chandra
from ....source import ConstantPointSource, FixedPointing
#from .....optics import MarxMirror


CWD = os.getcwd()
TEST_DIR = os.path.dirname(__file__)


def setup_function(function):
    os.chdir(TEST_DIR)

def teardown_function(function):
    os.chdir(CWD)

def test_ditherpattern():
    '''check that the dither pattern generated with consistent with MARX.'''
    mypointing = chandra.LissajousDither(coords=(212.5, -33.), roll=15.)
    time = np.arange(1000)
    coords = np.rad2deg(mypointing.pointing(time))
    masol = Table.read('sim_asol.fits')
    assert np.allclose(coords[:, 0], np.interp(time, masol['time']-masol['time'][0], masol['ra']))
    assert np.allclose(coords[:, 1], np.interp(time, masol['time']-masol['time'][0], masol['dec']))
    assert np.allclose(coords[:, 2], np.interp(time, masol['time']-masol['time'][0], masol['roll']))

def test_stationary_pointing():
    '''Constant pointing can also be realized through a Lissajous with amplitude=0.'''
    mysource = ConstantPointSource((30., 30.), energy=1., flux=1.)
    fixedpointing = FixedPointing(coords=(30., 30.), roll=15.)
    lisspointing = chandra.LissajousDither(coords=(30.,30.), roll=15., DitherAmp=np.zeros(3))

    photons = mysource.generate_photons(2)
    fixedphotons = fixedpointing(photons.copy())
    lissphotons = lisspointing(photons.copy())

    assert np.allclose(fixedphotons['dir'], lissphotons['dir'])
