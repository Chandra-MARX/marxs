import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from marxs.source.labSource import LabConstantPointSource
from marxs.optics.multiLayerMirror import MultiLayerMirror
from marxs.optics.detector import FlatDetector

'''This is an example test script that uses a source, multilayer mirror, and detector.
Note that string1 and string2 depend on the paths of those files.
'''

string1 = '../marxs/optics/data/A12113.txt'
string2 = '../marxs/optics/data/ALSpolarization2.txt'
a = 2**(-0.5)
rotation1 = np.array([[a, 0, -a],
					  [0, 1, 0],
					  [a, 0, a]])
rotation2 = np.array([[0, 0, 1],
					  [0, 1, 0],
					  [-1, 0, 0]])

source = LabConstantPointSource([100., 0., 0.], -1., 1., 0.35, direction='-x')
mirror = MultiLayerMirror(string1, string2, position=np.array([0., 0., 0.]), orientation=rotation1)
detector = FlatDetector(1., position=np.array([0., 0., 25.]), orientation=rotation2, zoom = np.array([1, 100, 100]))

photons = source.generate_photons(1e7)
photons = mirror.process_photons(photons)
photons = detector.process_photons(photons)

p = photons[photons['probability'] > 0]
r = np.random.uniform(size=len(p))

q = p[p['probability'] > r]

plt.clf()
fig = plt.figure()
ax = fig.gca()
ax.plot(q['det_x'], q['det_y'], '.')
ax.set_xlim([-100,100])
ax.set_ylim([-100,100])
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.quiver(np.zeros(len(q)), np.zeros(len(q)), np.zeros(len(q)), q['polarization'][:,0], q['polarization'][:,1], q['polarization'][:,2])
ax2.set_xlim([-20,20])
ax2.set_ylim([-20,20])
ax2.set_zlim([-20,20])
plt.show()

a = sum(abs(q['polarization'][:,0]))
b = sum(abs(q['polarization'][:,1]))

print a,b