import astropy.units as u
from astropy.coordinates import SkyCoord
from marxs.source import PointSource, FixedPointing
from marxs.missions import chandra

# Any coordinate will do as long as source and pointing use the same coordinates
coords = SkyCoord(278. * u.deg, -77. * u.deg)
src = PointSource(coords=coords)
pnt = FixedPointing(coords=coords)

# Define the elements that Chandra is made of
aper = chandra.Aperture()
hrma = chandra.HRMA()
hetg = chandra.HETG()
acis = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])

# Ignore nore all photons from the HEG
def onlyMEG(photons):
    photons['probability'][photons['mirror_shell'] < 2] = 0
    return photons

# Save time by putting most photons in order -1/+1
from marxs.optics.grating import OrderSelector
selector = OrderSelector([-1, 0, 1], p=[.45, .1, .45])
for e in hetg.elements:
    e.order_selector = selector


from marxs.design.tolerancing import (moveglobal, run_tolerances_for_energies,
                                      CaptureResAeff, generate_6d_wigglelist,
                                      load_and_plot)

# Next, we generate input for the function that will move the HESS
# around in our simulation. We do five steps for every translation
# direction (-5, -1, 0, 1, and 5 mm) and seven steps for every
# rotation.
changeglobal, changeind = generate_6d_wigglelist([0., 1., 5.] * u.mm,
                                                 [0., 1., 5., 10.] * u.arcmin)

# Which Acis coordinate to use?
resaeff = CaptureResAeff(A_geom=aper.area.to(u.cm**2), orders=[-1, 0, 1], dispersion_coord='tdetx')

from marxs.simulator import Sequence

res = run_tolerances_for_energies(src, [1.5, 4.] * u.keV,
                                      Sequence(elements=[pnt, aper, hrma, onlyMEG]),
                                      Sequence(elements=[hetg, acis]),
                                      moveglobal, hetg,
                                      changeglobal,
                                      resaeff,
                                      t_source=10000)

import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdirname:
    name = os.path.join(tmpdirname, 'wiggle_individual.fits')
    res.write(name)
    tab, fig = load_and_plot(name, ['dx', 'dy', 'dz', 'rx', 'ry', 'rz'])

fig.axes[0].legend()

for i in range(6):
    fig.axes[2 * i].set_ylim([0, 1e3])
    fig.axes[2 * i + 1].set_ylim([0, 70])

fig.set_size_inches(8, 6)
fig.subplots_adjust(left=.09, right=.93, top=.95, wspace=1.)
