"""
This is an Astropy affiliated package.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("marxs")
except PackageNotFoundError:
    # package is not installed
    pass


'''Convert from photon energy in keV to wavelength in mm'''
energy2wave = 1.2398419292004202e-06
