import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import table

from marxs.source import poisson_process
from marxs import optics
from marxs.source import PointSource
from marxs.missions import chandra

ngc1313_X1 = SkyCoord("3h18m19.99s -66d29m10.97s")
ngc1313_X2 = SkyCoord("3h18m22.34s -66d36m03.68s")

energies = np.arange(.3, 8., .01)
spectrum1 = 6.3e-4 * energies**(-1.9)
spectrum2 = 3.9e-4 * energies**(-2.4)

flux1 = poisson_process((spectrum1[1:] * np.diff(energies)).sum())
flux2 = poisson_process((spectrum2[1:] * np.diff(energies)).sum())

parfile = os.path.join(os.path.dirname(optics.__file__), 'hrma.par')
HRMA = optics.MarxMirror(parfile)

src1 = PointSource(coords=ngc1313_X1, energy={'energy': energies, 'flux': spectrum1},
                   flux=flux1, geomarea=HRMA.area)
src2 = PointSource(coords=ngc1313_X2, energy={'energy': energies, 'flux': spectrum2},
                   flux=flux2, geomarea=HRMA.area)

pointing = chandra.LissajousDither(coords=ngc1313_X1)
hetg = chandra.HETG()
acis = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])

p1 = src1.generate_photons(1e4)  # 10 ks exposure time
p1 = pointing(p1)
p1 = HRMA(p1)
p1 = hetg(p1)
p1 = acis(p1)

p2 = src2.generate_photons(1e4)  # 10 ks exposure time
p2 = pointing(p2)
p2 = HRMA(p2)
p2 = hetg(p2)
p2 = acis(p2)

p = table.vstack([p1, p2])

fig = plt.figure()
plt.plot(p1['tdetx'], p1['tdety'], '.')
plt.plot(p2['tdetx'], p2['tdety'], '.')
