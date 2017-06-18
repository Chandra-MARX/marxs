import numpy as np
from transforms3d import euler

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
    rays = light(1000)
    rays = experiment(rays)
    n_detected[i] = rays['probability'].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.rad2deg(angles), n_detected, lw=3)
ax.set_xlim([0, 360])
ax.set_ylim([0, 520])
ax.set_xlabel('rotation angle [degree]')
ax.set_ylabel('number of detected photons')
