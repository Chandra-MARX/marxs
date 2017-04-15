import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord

from marxs.source import poisson_process
from marxs.missions import chandra
from marxs.source import PointSource

ngc1313_X1 = SkyCoord("3h18m19.99s -66d29m10.97s")

energies = np.arange(.3, 8., .01)
spectrum1 = 6.3e-4 * energies**(-1.9)
flux1 = (spectrum1[1:] * np.diff(energies)).sum()
flux1 = poisson_process(flux1)

aperture = chandra.Aperture()

src1 = PointSource(coords=ngc1313_X1, energy={'energy': energies, 'flux': spectrum1},
                   flux=flux1, geomarea=aperture.area)

pointing = chandra.LissajousDither(coords=ngc1313_X1)
hrma = chandra.HRMA()
acis = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])

photons = src1.generate_photons(5e3)  # 5 ks exposure time
photons = pointing(photons)
photons = aperture(photons)
photons = hrma(photons)
photons = acis(photons)

line = plt.plot(photons['tdetx'], photons['tdety'], '.')
