import numpy as np
import transforms3d
from marxs.source import ConstantPointSource, FixedPointing
from marxs.design import RowlandTorus, find_radius_of_photon_shell, GratingArrayStructure
from marxs.optics import MarxMirror, FlatGrating, uniform_efficiency_factory, FlatDetector, EfficiencyFile

mysource = ConstantPointSource((30., 30.), energy=1., flux=1.)
mypointing = FixedPointing(coords=(30, 30.))
marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0,0]))

photons = mysource.generate_photons(100000)
photons = mypointing.process_photons(photons)
photons = marxm.process_photons(photons)

# design the gratings
radius0 = find_radius_of_photon_shell(photons, 0, 9e4)

mytorus = RowlandTorus(9e4/2, 9e4/2)

gratingeff = uniform_efficiency_factory()
mygas = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'd':0.0002, 'order_selector': gratingeff})

#catorders = EfficiencyFile('/Users/hamogu/MITDropbox/projects/xraysurveyor/sim_input/Si-ox_p200_th15_dc02_d6110.dat', orders=np.arange(2, -13, -1))
#blaze = transforms3d.axangles.axangle2mat(np.array([0,1,0]), np.deg2rad(1.5))
#mygascat = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'orientation': blaze, 'd':0.0002, 'order_selector': catorders})



pg = photons[photons['mirror_shell'] == 0]

import cProfile
cProfile.run("pg = mygas.process_photons(pg)", "run9")

runf7 = base
run 9 = facet auskommentiert
run8 = p auskommentiert

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
