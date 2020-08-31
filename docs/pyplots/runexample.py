import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import QTable

from marxs.source import poisson_process
from marxs.missions import chandra
from marxs.source import PointSource

ngc1313_X1 = SkyCoord("3h18m19.99s -66d29m10.97s")

energies = np.arange(.3, 8., .01) * u.keV
fluxdensity = 6.3e-4 * energies.value**(-1.9) / u.s / u.cm**2 / u.keV
fluxperbin = fluxdensity[1:] * np.diff(energies)
flux = poisson_process(fluxperbin.sum())
energytab = QTable({'energy': energies, 'fluxdensity': fluxdensity})

aperture = chandra.Aperture()

src1 = PointSource(coords=ngc1313_X1, energy=energytab,
                   flux=flux, geomarea=aperture.area)

pointing = chandra.LissajousDither(coords=ngc1313_X1)
hrma = chandra.HRMA()
acis = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])

photons = src1.generate_photons(5 * u.ks)
photons = pointing(photons)
photons = aperture(photons)
photons = hrma(photons)
photons = acis(photons)

line = plt.plot(photons['tdetx'], photons['tdety'], '.')
