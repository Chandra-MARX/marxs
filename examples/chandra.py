import numpy as np

from marxs.missions import chandra
from marxs.source import ConstantPointSource, FixedPointing
from marxs.optics import MarxMirror
from marxs.analysis import find_best_detector_position, measure_FWHM

mysource = ConstantPointSource((30., 30.), energy=1., flux=50.)
mypointing = FixedPointing(coords=(30, 30.))
# mypointing = chandra.LissajousDither(coords=(30.,30.), roll=0.)
marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0,0]))
hetg = chandra.HETG()
acis = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])
#mydet = FlatDetector(zoom=1e5, pixsize=23.985e-3)

# Chandra class is not finished yet.
Chandra = chandra.Chandra(sequence=[mypointing, marxm, hetg, acis])

photons = mysource.generate_photons(5000)
#photons = Chandra(photons)

photons = mypointing(photons)
photons = marxm(photons)
photons = photons[photons['probability'] > 0]

photons = hetg(photons)
photons['hetgy'] = photons['pos'].data[:,1]/photons['pos'].data[:,3]
photons['hetgz'] = photons['pos'].data[:,2]/photons['pos'].data[:,3]

photons = acis(photons)
mypointing.write_asol(photons, 'asol.fits')
Chandra.write_evt(photons, 'photons.fits')

p = photons[(photons['order']==-1) & (photons['facet'] > 194)]
xbest = find_best_detector_position(p, col='det_x')
print(xbest.x)
print(measure_FWHM(p['det_x']), measure_FWHM(p['det_y']))

#for x in [-50, 0, 50, xbest.x]:
#    mdet = FlatDetector(position=np.array([x, 0, 0]), zoom=1e5, pixsize=1.)
#    photons = mdet.process_photons(photons)
#    photons.rename_column('det_x', 'det_x_{0}'.format(x))
#    photons.rename_column('det_y', 'det_y_{0}'.format(x))


import astropy
# Make output for Glue
# output non-vector columns
pout = photons[:]
for c in pout.colnames:
    if len(pout[c].shape) > 1:
        pout.remove_column(c)

# output facet information
facet = np.arange(len(hetg.elements))
facet_tab = astropy.table.Table({'facet':facet, 'facet_x': facet, 'facet_y': facet, 'facet_z':facet})
for i in facet:
    facet_tab['facet_x'][i] = hetg.elements[i].pos4d[0, 3]
    facet_tab['facet_y'][i] = hetg.elements[i].pos4d[1, 3]
    facet_tab['facet_z'][i] = hetg.elements[i].pos4d[2, 3]

photfac = astropy.table.join(pout, facet_tab)
photfac.write('chandra.fits', overwrite=True)


mfits = astropy.table.Table.read('point.fits')
for c in mfits.colnames:
    if len(mfits[c].shape) > 1:
        mfits.remove_column(c)

mfits.write('marx.fits', overwrite=True)
