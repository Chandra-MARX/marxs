import copy
import numpy as np
import matplotlib.pyplot as plt
from marxs import optics, design, analysis, source, simulator
from marxs.design import rowland
from astropy.coordinates import SkyCoord


rowland = design.rowland.RowlandTorus(5000, 5000)
aper = optics.CircleAperture(position=[12000, 0, 0],
                             zoom=[1, 1000, 1000],
                             r_inner=960)
mirr = optics.FlatStack(position=[11000, 0, 0], zoom=[20, 1000, 1000],
                        elements=[optics.PerfectLens,
                                  optics.RadialMirrorScatter],
                        keywords = [{'focallength': 11000},
                                    {'inplanescatter': 2e-5,
                                     'perpplanescatter': 3e-6}])
mirr.display['opacity'] = 0.5

class ColoredFlatGrating(optics.FlatGrating):
    '''Flat grating that also assigns a color to all passing photons.'''
    def specific_process_photons(self, *args, **kwargs):
        out = super(ColoredFlatGrating, self).specific_process_photons(*args, **kwargs)
        out['colorid'] = self.colorid if hasattr(self, 'colorid') else np.nan
        return out

gas = design.rowland.GratingArrayStructure(rowland=rowland, d_element=180,
                                           x_range=[8000, 10000],
                                           radius=[870, 910],
                                           elem_class=ColoredFlatGrating,
                                           elem_args={'d': 2e-4, 'zoom': [1, 80, 80],
                                                      'order_selector': optics.OrderSelector([-1, 0, 1])})
det_kwargs = {'d_element': 105, 'elem_class': optics.FlatDetector,
              'elem_args': {'zoom': [1, 50, 20], 'pixsize': 0.01}}
det = design.rowland.RowlandCircleArray(rowland, theta=[3.1, 3.2], **det_kwargs)
projectfp = analysis.ProjectOntoPlane()

gratingcolors = 'bgr'
for e in gas.elements:
    e.display = copy.deepcopy(e.display)
    e.ang = (np.arctan(e.pos4d[1,3] / e.pos4d[2, 3]) + np.pi/2) % (np.pi)
    e.colorid = int(e.ang / np.pi * 3)
    e.display['color'] = gratingcolors[e.colorid]

instrum = simulator.Sequence(elements=[aper, mirr, gas, det, projectfp])
star = source.PointSource(coords=SkyCoord(30., 30., unit='deg'),
                          energy=1., flux=1.)
pointing = source.FixedPointing(coords=SkyCoord(30., 30., unit='deg'))

photons = star.generate_photons(4000)
photons = pointing(photons)
photons = instrum(photons)
ind = (photons['probability'] > 0) & (photons['facet'] >=0)

fig = plt.figure()
ax0 = fig.add_subplot(221, aspect='equal')
ax1 = fig.add_subplot(222, aspect='equal', sharey=ax0)
ax0h = fig.add_subplot(223, sharex=ax0)
ax1h = fig.add_subplot(224, sharex=ax1, sharey=ax0h)

ind0 = photons['order'] == 0
# It is arbitrary which direction if called +1 or -1 order
ind1 = np.abs(photons['order']) == 1
out = ax0h.hist(photons['proj_x'][ind & ind0], range=[-1, 1], bins=20,
                lw=0, color='0.8')
out = ax1h.hist(photons['proj_x'][ind & ind1], range=[61, 63], bins=20,
                lw=0, color='0.8')

for i in [0, 2, 1]:  # make green plotted on top
    c = gratingcolors[i]
    indcolor = photons['colorid'] == i
    ax0.plot(photons['proj_x'][ind & ind0 & indcolor],
             photons['proj_y'][ind & ind0 & indcolor],
             '.', color=c)
    ax1.plot(photons['proj_x'][ind & ind1 & indcolor],
             photons['proj_y'][ind & ind1 & indcolor],
             '.', color=c)
    out = ax0h.hist(photons['proj_x'][ind & ind0 & indcolor],
                    range=[-1, 1], bins=20,
                    histtype='step', color=c, lw=2)
    out = ax1h.hist(photons['proj_x'][ind & ind1 & indcolor],
                    range=[61, 63], bins=20,
                    histtype='step', color=c, lw=2)

ax0.set_xlim([-.5,0.5])
ax1.set_xlim([61.65, 62.4])
ax0h.set_xlim([-.5,0.5])
ax1h.set_xlim([61.65, 62.4])

ax0.text(0.1, 0.9, '0th order', transform=ax0.transAxes)
ax1.text(0.1, 0.9, '1st order', transform=ax1.transAxes)
ax0h.set_xlabel('dispersion direction [mm]')
ax0.set_ylabel('cross-dispersion [mm]')
ax0.tick_params(axis='x', labelbottom='off')
ax1.tick_params(axis='both', labelleft='off', labelbottom='off')
ax1h.tick_params(axis='y', labelleft='off')
ax0h.set_ylabel('photons / bin')

for ax in [ax0, ax1, ax0h, ax1h]:
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

fig.subplots_adjust(hspace=0, wspace=0, left=.18, right=.96, top=.97, bottom=.1)
