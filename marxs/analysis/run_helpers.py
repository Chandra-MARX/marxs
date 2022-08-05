# Licensed under GPL version 3 - see LICENSE.rst
'''This module contains helper functions to make running simulations easier.

Sometimes, those functions save only a few lines of code, but that can still
improve the readibility of e.g. loops and cells in notebooks.
'''
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from marxs.source import PointSource, FixedPointing


__all__ = ['default_coord',
           'run_monoenergetic_simulation',
           ]

default_coord = SkyCoord(30., 30., unit='deg')

def monoenergetic_astrosimulation(instrument, energy, n_photons=2e4,
                                  reference_transform=np.eye(4)):
    '''Run simple simulation at fixed energy and fixed pointing.

    Parameters
    ----------
    instrument : `marxs.simulator.simulator.Sequence`
        Definition of an instrument with an aperture, but no pointing.
        This function will add a simple fixed pointing and a point source.
    energy : quantity
        Energy for the simulation
    n_photon : int
        number of photons for the simulation
    reference_transform : np.array of shape (4, 4)
        For instruments that do not use the canonocal x-axis as the optical axis.
        See `marxs.source.pointing.FixedPointing`


    Returns
    -------
    photons : `astropy.table.Table`
        photon list
    '''
    mysource = PointSource(coords=default_coord,
                           energy=energy,
                           flux=1. / u.s / u.cm**2)
    fixedpointing = FixedPointing(coords=default_coord, reference_transform=reference_transform)
    photons = mysource.generate_photons(n_photons * u.s)
    photons = fixedpointing(photons)
    photons = instrument(photons)
    return photons
