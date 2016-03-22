import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from marxs.source.source import Source
en = np.arange(0.5, 7., .5)
flux = en**(-2)

# Input as dictionary
dictspectrum = {'energy': en, 'flux': flux}
s1 = Source(energy=dictspectrum, name='dict')

# Input as astropy Table
tablespectrum = Table([en, flux], names=['energy', 'flux'])
s2 = Source(energy=tablespectrum, name='table')

# Input as numpy arrays
numpyspectrum = np.vstack([en, flux])
s3 = Source(energy=numpyspectrum, name='2d array')

fig = plt.figure()
for s in [s1, s2, s3]:
    photons = s.generate_photons(1e3)
    plt.hist(photons['energy'], histtype='step', label=s.name, bins=20)

leg = plt.legend()
lab = plt.xlabel('Energy [keV]')
lab = plt.ylabel('Counts / bin')
