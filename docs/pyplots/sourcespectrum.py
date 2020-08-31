import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
import astropy.units as u
from marxs.source import Source
en = np.arange(0.5, 7., .5) * u.keV
fluxperbin = en.value**(-2) / u.s / u.cm**2 / u.keV

# Input as astropy QTable
tablespectrum = QTable([en, fluxperbin], names=['energy', 'fluxdensity'])
s = Source(energy=tablespectrum, name='table')


fig = plt.figure()

photons = s.generate_photons(1e3 * u.s)
plt.hist(photons['energy'], histtype='step', label=s.name, bins=20)

leg = plt.legend()
lab = plt.xlabel('Energy [keV]')
lab = plt.ylabel('Counts / bin')
