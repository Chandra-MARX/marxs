import numpy as np

from marxs.optics.polarization import polarization_vectors

def test_random_polarization():
	'''tests the angles to vector function for polarization
	
	This test makes sure that there is no obvious bias in a single direction.
	*** NOTE *** It is possible, but extremely unlikely, for this test to fail by chance.
	TODO: Perform a statistical calculation to show this test is reasonable.
	TODO: Test for other possible issues (ex: polarization in +/-x axis would not be detected)
	'''
	v_1 = np.random.uniform(size=3)
	v_1 /= np.linalg.norm(v_1)

	n = 100000

	dir_array = np.tile(v_1, (n, 1))
	angles = np.random.uniform(0, 2 * np.pi, n)

	polarization = polarization_vectors(dir_array, angles)

	assert np.allclose(polarization, polarization / np.linalg.norm(polarization, axis=1)[:, np.newaxis])

	x = sum(polarization[:, 0])
	y = sum(polarization[:, 1])
	z = sum(polarization[:, 2])

	v_2 = np.array([x, y, z])
	
	assert np.isclose(np.dot(v_1, v_2), 0)

	assert np.linalg.norm(v_2) < 0.01 * n