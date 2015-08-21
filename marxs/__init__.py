
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

'''Convert from photon energy in keV to wavelength in mm'''
energy2wave = 1.2398419292004202e-06
