import numpy as np
from ..polarization import polarization_vectors, paralleltransport_matrix
from .. import utils

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

def test_paralleltransport():
    '''Test parallel transport of vectors in a simple, analytical example.'''
    dir1 = np.array([[1,-1,0],
                     [1,-1,0],
                     [1,-1,0],
                     [1,-1,0],
                     [1,-1,0]])
    dir2 = np.array([[1,-1,0],
                     [1,1,0],
                     [1,1,0],
                     [1,0.2,0],
                     [1,0.2, 0]])
    dir1 = utils.norm_vector(dir1)
    dir2 = utils.norm_vector(dir2)

    polin = np.array([[1,1,1],
                      [0,0,1],
                      [1, 1,0],
                      [0,0,1],
                      [1, 1,0]])
    polin = utils.norm_vector(polin)

    pmat = paralleltransport_matrix(dir1, dir2)
    out = np.einsum('ijk,ik->ij', pmat, polin)
    # for dir1==dir2 result is unchanged
    assert np.allclose(polin[0, :], out[0, :])
    # s vector is unchanged
    assert np.allclose(polin[:, 2], out[:, 2])
    # result has len 1
    assert np.allclose(np.linalg.norm(out, axis=1), 1)
    # result is perpendicular to dir2
    assert np.allclose(np.einsum('ij,ij->i', out, dir2), 0)

    pmat = paralleltransport_matrix(dir1, dir2, replace_nans=False)
    out = np.einsum('ijk,ik->ij', pmat, polin)
    # dir1==dir2 gives nan
    assert all(np.isnan(out[0, :]))
