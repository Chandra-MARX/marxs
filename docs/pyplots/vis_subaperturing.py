import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from marxs import optics, simulator, source
from marxs.missions import chandra

# define the instrument with mirror, gratings and detectors

marxm = chandra.HRMA(os.path.join(os.path.dirname(optics.__file__), 'hrma.par'))
hetg = chandra.HETG()
aciss = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])

keeppos = simulator.KeepCol('pos')
chand = simulator.Sequence(elements=[marxm, hetg, aciss])

# Define source and run photons through Chandra
star = source.PointSource(coords=SkyCoord(30., 30., unit='deg'),
                          energy=2., flux=1.)
pointing = source.FixedPointing(coords=SkyCoord(30., 30., unit='deg'))
photons = star.generate_photons(20000)
photons = pointing(photons)
photons = chand(photons)


# Color gratings according to the sector they are in
sectorcol = dict(zip('ABCDEFG', 'rgbcmyk'))
for e in hetg.elements:
     e.display = copy.deepcopy(e.display)
     # First character of "name" is sector id
     e.display['color'] = sectorcol[e.name[1]]

# Color all photons according to the sector they go through
photons['color'] = [sectorcol[hetg.elements[int(i)].name[1]] for i in photons['facet']]

# Make plot
ind = (photons['probability'] > 0) & (photons['facet'] >=0)
pp = photons[ind  & np.isfinite(photons['tdetx'])]
fig = plt.figure()
for p in pp [(pp['tdetx'] > 4000) & (pp['tdetx'] < 4150)]:
    plt.plot(p['tdetx'], p['tdety'], '.', c=p['color'])
plt.gca().set_aspect("equal")
plt.xlim([4136, 4140])
plt.ylim([2232, 2235])
plt.title('0 th order')
