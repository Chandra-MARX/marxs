import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from marxs.source.labSource import LabPointSource
from marxs.optics.multiLayerMirror import MultiLayerMirror
from marxs.optics.detector import FlatDetector

'''This is an example test script that uses a source, multilayer mirror, and detector.
Note that string1 and string2 depend on the paths of those files.
'''

def normalEnergyFunc(mean, stdDev):
	'''This creates and returns a function that assigns energies to photons randomly
	according to the normal distribution with the given mean and standard deviation.
	'''
    def energyFunc(photonTimes):
        n = len(photonTimes)
        return np.random.normal(mean, stdDev, n)
    return energyFunc


string1 = '../marxs/optics/data/A12113.txt'
string2 = '../marxs/optics/data/ALSpolarization2.txt'
a = 2**(-0.5)
rotation1 = np.array([[a, 0, -a],
					  [0, 1, 0],
					  [a, 0, a]])
rotation2 = np.array([[0, 0, 1],
					  [0, 1, 0],
					  [-1, 0, 0]])

source = LabPointSource([100., 0., 0.], direction='-x', energy = normalEnergyFunc(0.31, 0.003), flux = 1e9)
mirror = MultiLayerMirror(string1, string2, position=np.array([0., 0., 0.]), orientation=rotation1)
detector = FlatDetector(1., position=np.array([0., 0., 25.]), orientation=rotation2, zoom = np.array([1, 100, 100]))

photons = source.generate_photons(0.01)
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
#ax2.quiver(np.zeros(len(q)), np.zeros(len(q)), np.zeros(len(q)), q['polarization'][:,0], q['polarization'][:,1], q['polarization'][:,2])
ax2.quiver(q['pos'][:, 0] * 10, q['pos'][:, 1] * 10, q['pos'][:, 2] * 10, q['polarization'][:,0], q['polarization'][:,1], q['polarization'][:,2], length = 15)
ax2.set_xlim([-100,100])
ax2.set_ylim([-100,100])
ax2.set_zlim([-350,-150])
plt.show()

a = sum(abs(q['polarization'][:,0]))
b = sum(abs(q['polarization'][:,1]))

#print a,b