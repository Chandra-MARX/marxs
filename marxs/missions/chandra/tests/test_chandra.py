import os
import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord

from ... import chandra
from ....source import PointSource, FixedPointing
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
    mysource = PointSource(coords=SkyCoord(30., 30., unit="deg"), energy=1., flux=1.)
    fixedpointing = FixedPointing(coords=(30., 30.), roll=15.)
    lisspointing = chandra.LissajousDither(coords=(30.,30.), roll=15., DitherAmp=np.zeros(3))

    photons = mysource.generate_photons(2)
    fixedphotons = fixedpointing(photons.copy())
    lissphotons = lisspointing(photons.copy())

    assert np.allclose(fixedphotons['dir'], lissphotons['dir'])

def test_detector_coordsystems():
    '''Compare detector coordinates for known photon direction with CIAO dmcoords.

    Currently, I don't aim to make this very precise - within a pixel or so is fine.
    To make this more precise, there are several places to go:
    The most obvious one is the size of the pixel - currently the number of pixels times
    the pixel size does now match the length of the chip precisely.
    '''
    mysource = PointSource(coords=SkyCoord(30., 30., unit="deg"), energy=1., flux=1.)
    mypointing = chandra.LissajousDither(coords=(30.,30.), roll=15.)
    # marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0,0]))
    acis = chandra.ACIS(chips=[0,1,2,3], aimpoint=chandra.AIMPOINTS['ACIS-I'])

    photons = mysource.generate_photons(5)
    photons = mypointing(photons)
    # We want reproducible direction, so don't use mirror, but set direction by hand
    # photons = marxm(photons)
    # photons = photons[photons['probability'] > 0]
    photons['pos'] = np.array([[100., 0, 0, 1],[100, 10, 0, 1], [100, 0, 10, 1],
                               [100, -10, 0, 1], [100, 0, -10, 1]])

    photons.meta['MISSION'] = ('AXAF', 'Mission')
    photons.meta['TELESCOP'] = ('CHANDRA', 'Telescope')

    photons = acis(photons)
    # We need the asol to run CIAO.
    # I've done that already and the expected numbers are hard-coded in the assert statement.
    # Here are commands I used:
    # mypointing.write_asol(photons, 'asol.fits')
    # chandra.Chandra.write_evt('photons.fits')

    atol = 1.
    # This is the starting point
    assert np.allclose(photons['detx'], [4096.5, 4513.2, 4096.5, 3679.8, 4096.5], atol=atol)
    assert np.allclose(photons['dety'], [4096.5, 4096.5, 3679.8, 4096.5, 4513.2], atol=atol)
    # I took these numbers and types them into dmcoords.
    # Thus, if these numbers change when increasing the accuracy of the pixel size or the
    # position of the chip edges, ... then all the numbers below need to be changed, too.
    assert np.all(photons['CCD_ID'] == [3, 3, 1, 2, 3])
    assert np.allclose(photons['chipx'], [984.34, 984.34, 355.7, 40.91, 566.56], atol=atol)
    assert np.allclose(photons['chipy'], [994.8, 577.02, 994.23, 658.95, 993.75], atol=atol)
    assert np.allclose(photons['x'], [4096.5, 4499., 3988.5, 3693.4, 4204.34], atol=atol)
    assert np.allclose(photons['y'], [4096.5, 3988.65, 3693.4, 4204.5, 4499.], atol=atol)
    assert np.allclose(photons['tdetx'], [4137.2, 4555, 4137.77, 3720, 4138.25], atol=atol)


def test_ACIS_pixel_number():
    '''The pixel size and chip size given in the docs are not consistent.

    Make sure we end up with 1024*1024 chips.
    '''
    acis = chandra.ACIS(chips=[0, 1, 2, 3], aimpoint=chandra.AIMPOINTS['ACIS-I'])
    for i in range(3):
        assert acis.elements[i].npix == [1024, 1024]

    acis = chandra.ACIS(chips=[4, 5, 6, 7, 8, 9], aimpoint=chandra.AIMPOINTS['ACIS-I'])
    for i in range(5):
        assert acis.elements[i].npix == [1024, 1024]
