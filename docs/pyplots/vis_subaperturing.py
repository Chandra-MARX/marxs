import copy
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import numpy as np

from marxs import optics, simulator, source, design
from marxs.design import rowland

rowland = design.rowland.RowlandTorus(500, 500)

aper = optics.CircleAperture(position=[1200, 0, 0], zoom=[1, 100, 100], r_inner=20)
mirr = optics.FlatStack(position=[1100, 0, 0], zoom=[20, 100, 100],
                        elements=[optics.PerfectLens, optics.RadialMirrorScatter],
                        keywords = [{'focallength': 1100},
                                    {'inplanescatter': 1e-3, 'perpplanescatter': 1e-4}])
gas = design.rowland.GratingArrayStructure(rowland=rowland, d_element=25, x_range=[800, 1000],
                                           radius=[20, 100], elem_class=optics.FlatGrating,
                                           elem_args={'d': 1e-5, 'zoom': [1, 10, 10],
                                                      'order_selector': optics.OrderSelector([-1, 0, 1])})

# Color gratings according to the sector they are in
for e in gas.elements:
     e.display = copy.deepcopy(e.display)
     # Angle from baseline in range 0..pi
     ang = np.arctan(e.pos4d[1,3] / e.pos4d[2, 3]) % (np.pi)
     # pick one of fixe colors
     e.display['color'] = 'rgb'[int(ang / np.pi * 3)]

det_kwargs = {'d_element': 10.5, 'elem_class': optics.FlatDetector,
              'elem_args': {'zoom': [1, 5, 5], 'pixsize': 0.01}}
det = design.rowland.RowlandCircleArray(rowland, theta=[2.3, 3.9], **det_kwargs)

target = SkyCoord(30., 30., unit='deg')
star = source.PointSource(coords=target, energy=.5, flux= 1.)
pointing = source.FixedPointing(coords=target)
instrum = simulator.Sequence(elements=[aper, mirr, gas, det])

photons = star.generate_photons(10000)
photons = pointing(photons)
photons = instrum(photons)

 # Color all photons according to the grating they go through
photons['color'] = [gas.elements[int(i)].display['color'] for i in photons['facet']]

# Make plot
fig = plt.figure()
for p in photons[photons['order'] == 0]:
    plt.plot(p['det_x'], p['det_y'], '.', c=p['color'])
plt.gca().set_aspect("equal")
plt.title('0 th order')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
