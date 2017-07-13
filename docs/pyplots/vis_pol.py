import numpy as np
from transforms3d import euler
from astropy.io import ascii
import matplotlib.pyplot as plt

from marxs.source import LabPointSourceCone
from marxs.optics import FlatBrewsterMirror, FlatDetector
from marxs.simulator import Sequence


light = LabPointSourceCone(position=[200., 0., 0.], direction=[-1., 0, 0], half_opening=0.02)
# keep a copy of the position and direction for later
light_pos = light.position.copy()
light_dir = light.dir.copy()

rot1 = np.array([[2.**(-0.5), 0, -2.**(-0.5)],[0, 1., 0],[2.**(-0.5), 0, 2.**(-0.5)]])
m1 = FlatBrewsterMirror(orientation=rot1,zoom=[1, 10, 30])
# Keep a copy of the initial position for later
m1pos4d = m1.pos4d.copy()

rot2 = np.array([[2.**(-0.5), 0, 2.**(-0.5)],[0, 1., 0],[-2.**(-0.5), 0, 2.**(-0.5)]])
m2 = FlatBrewsterMirror(orientation=rot2,zoom=[1, 10, 30], position=[0, 0, 2e2])

det = FlatDetector(position=[50, 0, 2e2], zoom=[1., 20., 20.])

experiment = Sequence(elements=[m1, m2, det])

angles = np.arange(0, 2 * np.pi, 0.1)
n_detected = np.zeros_like(angles)

for i, angle in enumerate(angles):
    rotmat = np.eye(4)
    rotmat[:3, :3] = euler.euler2mat(angle, 0, 0, 'szxy')
    light.position = np.dot(rotmat, light_pos)
    light.dir = np.dot(rotmat, light_dir)
    m1.pos4d = np.dot(rotmat, m1pos4d)
    rays = light(10000)
    rays = experiment(rays)
    n_detected[i] = rays['probability'].sum()

# Per email from Herman Marshal, see
# http://adsabs.harvard.edu/abs/2013SPIE.8861E..1DM
labdata = '''
      Angle        Rate     Uncertainty
#      (deg)   (cnt/s/.1mA)  (cnt/s/.1mA)
      89.8900     0.320045   0.00624663
      44.8900     0.132169   0.00401622
      44.8900     0.133989   0.00402895
      90.2400     0.319983   0.00621355
     0.250000   0.00816994   0.00117923
      14.8300    0.0210306   0.00162740
      29.9000    0.0705870   0.00295397
      45.1100     0.134869   0.00403902
      60.2600     0.212002   0.00503627
      75.3900     0.274137   0.00573363
      127.460     0.180642   0.00461368
     -30.0300    0.0939831   0.00341587
'''
obs = ascii.read(labdata)
# Normalize the rate to 1
factor = np.max(obs['Rate'])
obs['Rate'] /= factor
obs['Uncertainty'] /= factor
# Definition of angle zero differs in the Marshal et al paper
obs['Angle'] += 90

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.rad2deg(angles), n_detected / np.max(n_detected), lw=3)
ax.plot(obs['Angle'], obs['Rate'], 'ro', ms=8)
ax.set_xlim([0, 360])
ax.set_ylim([0, 1.05])
ax.set_xlabel('rotation angle [degree]')
ax.set_ylabel('photon rate [scaled]')
