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

def test_detector_coordsystems():
    '''Compare detector coordinates for known photon direction with CIAO dmcoords.'''
    mysource = ConstantPointSource((30., 30.), energy=1., flux=1.)
    mypointing = chandra.LissajousDither(coords=(30.,30.), roll=15.)
    # marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0,0]))
    acis = chandra.ACIS(chips=[0,1,2,3], aimpoint=chandra.AIMPOINTS['ACIS-I'])

    photons = mysource.generate_photons(2)
    photons = mypointing(photons)
    # We want reproducible direction, so don't use mirror, but set direction by hand
    # photons = marxm(photons)
    # photons = photons[photons['probability'] > 0]
    photons['pos'] = np.array([[100., 0, 0, 1],[100, 0, 0, 1]])

    photons = acis(photons)
    # We need the asol to run CIAO.
    # I've done that already and the expected numbers are hard-coded in the assert statement.
    # Here are commands I used:
    # mypointing.write_asol(photons, 'asol.fits')
