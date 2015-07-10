import numpy as np
import transforms3d
import astropy.table
import scipy
import scipy.optimize

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

#gratingeff = uniform_efficiency_factory()
#mygas = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'd':0.0002, 'order_selector': gratingeff})

catfile = '/Users/hamogu/MITDropbox/projects/xraysurveyor/sim_input/Si-ox_p200_th15_dc02_d6110.dat'
catfile = '/melkor/d1/guenther/marx/xraysurveyor/sim_input/Si-ox_p200_th15_dc02_d6110.dat'

catorders = EfficiencyFile(catfile, orders=np.arange(2, -13, -1))
blaze = transforms3d.axangles.axangle2mat(np.array([0,1,0]), np.deg2rad(1.5))
mygascat = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'orientation': blaze, 'd':0.0002, 'order_selector': catorders})


from marxs.design.rowland import design_tilted_torus

R, r, pos4d = design_tilted_torus(9e4, np.deg2rad(3), np.deg2rad(6))
mytorustilt = RowlandTorus(R, r, pos4d=pos4d)
mygascattilt = GratingArrayStructure(mytorustilt, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'orientation': blaze, 'd':0.0002, 'order_selector': catorders})



pg = photons[photons['mirror_shell'] == 0]

pg = mygascattilt.process_photons(pg)

p = pg[pg['probability'] > 0]
mdet = FlatDetector(position=np.array([0e3, 0, 0]), zoom=1e5, pixsize=1.)

p = mdet.process_photons(p)

plt.clf()
plt.plot(p['det_x'], p['det_y'], 'k.')
ind = (p['order'] == 3)
plt.plot(p['det_x'][ind], p['det_y'][ind], 'ro')
ind = (p['order'] == 2)
plt.plot(p['det_x'][ind], p['det_y'][ind], 'b+')




for x, c in zip([0,250., 500.0, 777, 1000], 'bgrcmk'):
    p = pg[:]
    mdet = FlatDetector(position=np.array([x, 0, 0]), zoom=1e5,pixsize=1.)
    p = mdet.process_photons(p)
    ind = (p['probability'] > 0) & (p['order'] == -8)
    plt.plot(p['det_x'][ind], p['det_y'][ind], c+'s', label='{0}'.format(x))
    plt.plot(p['det_x'][~ind], p['det_y'][~ind], c+'.')

plt.legend()



## TBD: add check in code if d_facet >=zoom of grating!

def find_best_detector_position(photons):

    def width(x, photons):
        mdet = FlatDetector(position=np.array([x, 0, 0]), zoom=1e5,pixsize=1.)
        photons = mdet.process_photons(photons)
        return np.std(photons['det_y'])

    return scipy.optimize.minimize(width, 0, args=(photons,), options={'maxiter': 20, 'disp': True})

# output non-vector columns
pout = p[:]
for c in p.colnames:
    if len(p[c].shape) > 1:
        pout.remove_column(c)
# output facet information
facet = np.arange(len(mygascattilt.facets))
facet_tab = astropy.table.Table({'facet':facet, 'facet_x': facet, 'facet_y': facet, 'facet_z':facet})
for i in facet:
    facet_tab['facet_x'][i] = mygascattilt.facets[i].pos4d[0, 3]
    facet_tab['facet_y'][i] = mygascattilt.facets[i].pos4d[1, 3]
    facet_tab['facet_z'][i] = mygascattilt.facets[i].pos4d[2, 3]

photfac = astropy.table.join(pout[~pout['facet'].mask], facet_tab)
photfac.write('../xraysurveyor/facets.fits', overwrite=True)
