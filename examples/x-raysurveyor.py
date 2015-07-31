import numpy as np
import transforms3d
import astropy.table
import scipy
import scipy.optimize

from marxs.source import ConstantPointSource, FixedPointing
from marxs.design import RowlandTorus, find_radius_of_photon_shell, GratingArrayStructure
from marxs.optics import MarxMirror, FlatGrating, uniform_efficiency_factory, FlatDetector, EfficiencyFile, constant_order_factory
from marxs.design.rowland import design_tilted_torus
from marxs.math.pluecker import h2e


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
catorders = constant_order_factory(-8)


blaze = transforms3d.axangles.axangle2mat(np.array([0,1,0]), np.deg2rad(1.5))
R, r, pos4d = design_tilted_torus(9e4, np.deg2rad(ang), np.deg2rad(2*ang))

mytorustilt = RowlandTorus(R, r, pos4d=pos4d)
mygascattilt = GratingArrayStructure(mytorustilt, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'orientation': blaze, 'd':0.0002, 'order_selector': catorders})

pg = photons[(photons['mirror_shell'] == 0) & (photons['probability'] > 0)]

pg = mygascattilt.process_photons(pg)

#    plot_rays(pg[(pg['order']==-8) & (pg['facet']==300)], np.array([0.e5, 1.05e5]), [0,2], color=c)
#    plot_rays(pg[(pg['order']==-8) & (pg['facet']==3)], np.array([0.e5, 1.05e5]), [0,2], color=c)
#    print ang, np.mean(np.rad2deg(pg['blaze'][pg['order']==-8]))
#   #plt.hist(np.rad2deg(pg['blaze'][pg['order']==-8]))

mygascattilt.facet_uncertainty = generate_facet_uncertainty(len(mygascattilt.facet_uncertainty), np.ones(3), np.zeros(3))
mygascattilt.generate_facets(mygascattilt.facet_class, mygascattilt.facet_args)
p1 = photons[(photons['mirror_shell'] == 0) & (photons['probability'] > 0)]
p1 = mygascattilt.process_photons(p1)


mygascattilt.facet_uncertainty = generate_facet_uncertainty(len(mygascattilt.facet_uncertainty), np.zeros(3), np.deg2rad(np.ones(3)/60.))
mygascattilt.generate_facets(mygascattilt.facet_class, mygascattilt.facet_args)
p2 = photons[(photons['mirror_shell'] == 0) & (photons['probability'] > 0)]
p2 = mygascattilt.process_photons(p2)

mdet = FlatDetector(position=np.array([535, 0, 0]), zoom=1e5, pixsize=1.)
pg = mdet.process_photons(pg)
p1 = mdet.process_photons(p1)
p2 = mdet.process_photons(p2)
plt.scatter(pg['det_x'], pg['det_y'], c='b')
plt.scatter(p1['det_x'], p1['det_y'], c='g')
plt.scatter(p2['det_x'], p2['det_y'], c='r')


p = pg.copy()

mdet = FlatDetector(position=np.array([0e3, 0, 0]), zoom=1e5, pixsize=1.)

P = mdet.process_photons(p)

plt.figure()
plt.plot(p['det_x'], p['det_y'], 'k.')
ind = (p['order'] == 3)
plt.plot(p['det_x'][ind], p['det_y'][ind], 'ro')
ind = (p['order'] == 2)
plt.plot(p['det_x'][ind], p['det_y'][ind], 'b+')




for x, c in zip([0, xbest.x, 1000], 'bgrcmk'):
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

xbest = find_best_detector_position(pg[pg['order']==-8])
p = pg[pg['probability'] > 0]

for x in [-500, 0, xbest.x, 1000, 1500,5000,5e4]:
    mdet = FlatDetector(position=np.array([x, 0, 0]), zoom=1e5, pixsize=1.)
    p = mdet.process_photons(p)
    p.rename_column('det_x', 'det_x_{0}'.format(x))
    p.rename_column('det_y', 'det_y_{0}'.format(x))

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

def plot_rays(photons, limits, axes=[0,1], **kwargs):
    for p in photons:
        point = h2e(p['pos']).data[axes]
        vec = h2e(p['dir']).data[axes]
        plt.plot(point[0] + limits * vec[0], point[1] + limits * vec[1], **kwargs)

from transforms3d.affines import compose
from transforms3d.euler import euler2mat

def generate_facet_uncertainty(n, xyz, angle_xyz):
    translation = np.random.normal(size=(n, 3)) * xyz[np.newaxis, :]
    rotation = np.random.normal(size=(n, 3)) * angle_xyz[np.newaxis, :]
    return [compose(t, euler2mat(a[0], a[1], a[2], 'sxyz'), np.ones(3)) for t, a in zip(translation, rotation)]

