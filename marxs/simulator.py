class SimulationSetupError(Exception):
    pass


class Simulator(object):
    '''
    .. todo::
    This needs to take control of the output columns.
    I want the optical elements to be able to add columns as needed, so
    Simulator needs to go through and rename them or pass in only the
    necessary columns (pos, dir, energy, pol, maybe prop, tag).
    Decide how to handle prob. Multiply when relevant or after output.
    Introduce unique tag for photon. Useful for photon splitting.
    In source or tagging stage after that?

    Example:

    >>> mysource = source.ConstantPointSource(coords=(30., 30.), flux=1e-4, energy=2.)
    >>> sky2mission = source.ConstantPointing(coords=(30., 30.))
    sequence = [,
                source.Sky2Mission(30., 30.),
                mirror.GaussianBlur(0.5),
                detector.InfiniteCCD(0.5)]
    '''

    def __init__(self, sequence):
        if not callable(getattr(sequence[0], 'generate_photon', None)):
            raise SimulationSetupError('{0} cannot generate photons.'.format(str(sequence[0])))
        if not callable(getattr(sequence[-1], 'save')):
            raise SimulationSetupError('{0} does not have a method to output results.'.format(str(sequence[-1])))

        self.sequence = sequence

    def run():
        pass

import marxs
import marxs.math
import marxs.math.pluecker
import marxs.source
import marxs.source.source
import marxs.optics
import marxs.optics.aperture
import marxs.optics.mirror
import marxs.optics.detector
import marxs.design
import marxs.design.rowland

reload(marxs)
reload(marxs.math)
reload(marxs.math.pluecker)
reload(marxs.source)
reload(marxs.source.source)
reload(marxs.optics)
reload(marxs.optics.aperture)
reload(marxs.optics.mirror)
reload(marxs.optics.detector)
reload(marxs.design.rowland)



mysource = marxs.source.source.ConstantPointSource((30., 30.), 1., 300.)
photons = mysource.generate_photons(11)
mypointing = marxs.source.source.FixedPointing(coords=(30., 30.))
photons = mypointing.process_photons(photons)
myslit = marxs.optics.aperture.SquareEntranceAperture(size=2)
photons = myslit.process_photons(photons)
mm = marxs.optics.mirror.ThinLens(focallength=10)
photons = mm.process_photons(photons)

for x in np.arange(0., -13, -2):
    p = photons[:]
    mdet = marxs.optics.detector.InfiniteFlatDetector(position=np.array([x, 0, 0]))
    p = mdet.process_photons(p)
    plt.plot(p['det_y'], p['det_z'], 's', label='{0}'.format(x))

plt.legend()
plt.savefig('examples/movescreen.png')




# User source. Should I make that a general test case and move to module later?
class SymbolSource(marxs.source.source.Source):
    def __init__(self, coords, flux, energy, polarization=np.nan, effective_area=1, size=1):
        self.coords = coords
        self.flux = flux
        self.effective_area = effective_area
        self.rate = flux * effective_area
        self.energy = energy
        self.polarization = polarization
        self.size = size

    def generate_photons(self, t):
        n = t  * self.rate
        elem = np.random.choice(3, size=n)

        ra = np.empty(n)
        ra[:] = self.coords[0]
        dec = np.empty(n)
        dec[:] = self.coords[1]
        ra[elem == 0] += self.size * np.random.random(np.sum(elem == 0))
        ra[elem == 1] += self.size
        dec[elem == 1] += 0.5 * self.size * np.random.random(np.sum(elem == 1))
        ra[elem == 2] += 0.8 * self.size
        dec[elem == 2] += 0.3 * self.size * np.random.random(np.sum(elem == 2))


        out = np.empty((6, n))
        out[0, :] = np.arange(0, t, 1. / self.rate)
        out[1,:] = ra
        out[2,:] = dec
        out[3, :] = self.energy
        out[4,:] = self.polarization
        out[5, :] = 1.
        return Table(out.T, names = ('time', 'ra', 'dec', 'energy', 'polarization', 'probability'))

mysource = SymbolSource((30., 30.), 1., 300., size=1)
photons = mysource.generate_photons(500)
mypointing = marxs.source.source.FixedPointing(coords=(30., 30.))
photons = mypointing.process_photons(photons)
myslit = marxs.optics.aperture.RectangleAperture(zoom=2)
photons = myslit.process_photons(photons)
mm = marxs.optics.mirror.ThinLens(focallength=10)
photons = mm.process_photons(photons)
mdet = marxs.optics.detector.InfiniteFlatDetector(position=np.array([-10, 0, 0]))
photons = mdet.process_photons(photons)

plt.clf()
plt.plot(photons['det_y'], photons['det_z'], 's', label='{0}'.format(x))
plt.savefig('examples/complexsource10.png')





mysource = marxs.source.source.ConstantPointSource((30., 30.), 1., 1.)
photons = mysource.generate_photons(10000)
mypointing = marxs.source.source.FixedPointing(coords=(30, 30.))
photons = mypointing.process_photons(photons)

import marxs.optics.marx
marxm = marxs.optics.marx.MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0,0]))
photons = marxm.process_photons(photons)

for x, c in zip([1e5, 8e4,4e4,0], 'bgrcmk'):
    p = photons[:]
    mdet = marxs.optics.detector.FlatDetector(position=np.array([x, 0, 0]), zoom=1e5,pixsize=1.)
    p = mdet.process_photons(p)
    ind = p['probability'] > 0
    plt.plot(p['det_x'][ind], p['det_y'][ind], c+'s', label='{0}'.format(x))
    plt.plot(p['det_x'][~ind], p['det_y'][~ind], c+'.')

plt.legend()


### for Ralf - step 1

import numpy as np
from marxs.source import ConstantPointSource, FixedPointing
from marxs.design import RowlandTorus, find_radius_of_photon_shell, GratingArrayStructure
from marxs.optics import MarxMirror, FlatGrating, uniform_efficiency_factory, FlatDetector, EfficiencyFile

mysource = ConstantPointSource((30., 30.), 1., 1.)
mypointing = FixedPointing(coords=(30, 30.))
marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0,0]))

photons = mysource.generate_photons(10000)
photons = mypointing.process_photons(photons)
photons = marxm.process_photons(photons)

# design the gratings
radius0 = find_radius_of_photon_shell(photons, 0, 9e4)

mytorus = RowlandTorus(9e4/2, 9e4/2)

gratingeff = uniform_efficiency_factory()
mygas = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'd':0.002, 'order_selector': gratingeff})

catorders = EfficiencyFile('/melkor/d1/guenther/marx/xraysurveyor/sim_input/Si-ox_p200_th15_dc02_d6110.dat', orders=np.arange(2, -13, -1))
mygascat = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'd':0.0002, 'order_selector': catorders})



pg = photons[photons['mirror_shell'] == 0]
pg = mygas.process_photons(pg)
p = pg[pg['probability'] > 0]
mdet = FlatDetector(position=np.array([0e3, 0, 0]), zoom=1e5, pixsize=1.)

p = mdet.process_photons(p)

plt.clf()
plt.plot(p['det_x'], p['det_y'], 'k.')
ind = (p['order'] == 3)
plt.plot(p['det_x'][ind], p['det_y'][ind], 'ro')
ind = (p['order'] == 2)
plt.plot(p['det_x'][ind], p['det_y'][ind], 'b+')




for x, c in zip([0, 0.31, 3.], 'bgrcmk'):
    p = pg[:]
    mdet = FlatDetector(position=np.array([x, 0, 0]), zoom=1e5,pixsize=1.)
    p = mdet.process_photons(p)
    ind = (p['probability'] > 0) & (p['order'] == 3)
    plt.plot(p['det_x'][ind], p['det_y'][ind], c+'s', label='{0}'.format(x))
    plt.plot(p['det_x'][~ind], p['det_y'][~ind], c+'.')

plt.legend()



## TBD: add check in code if d_facet >=zoom of grating!
