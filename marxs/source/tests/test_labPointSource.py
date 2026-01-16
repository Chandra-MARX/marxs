# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import transforms3d
import pytest
import astropy.units as u

from ..labSource import LabPointSourceCone


def test_labsource_eukledian():
    with pytest.raises(ValueError) as err:
        source = LabPointSourceCone(position=[1, 2, 3, 1.])
    assert "in Euclidean coordinates" in str(err.value)


def test_photon_generation():
    '''This tests the lab point source. It checks that the starting points
    are all at the sources position.
    '''
    pos = [1., 1., 1.]
    rate = 10 / u.s
    source = LabPointSourceCone(position=pos, flux=rate, energy=5. * u.keV)

    photons = source.generate_photons(1. * u.s)
    assert np.all(photons['pos'] == np.ones([10, 4]))


def test_directions_range_cone():
    '''This tests that all of the photons in the returned table have directions within the given parameters.
    - Ensures that all direciton vectors are normed
    - Ensures that all direction vectors are within specified range
    '''

    # parameters
    pos = [5 * np.random.random(), 5 * np.random.random(), 5 * np.random.random()]
    rate = 10 * np.random.random() / u.s
    direction = [5* np.random.random(),5* np.random.random(),5* np.random.random()]
    delta = ((np.pi / 2) * (np.random.random() * 2 - 1))

    # run simulation
    source = LabPointSourceCone(position=pos, half_opening=delta, flux=rate,
                                energy=5. * u.keV, direction = direction)
    photons = source.generate_photons(1. * u.s)


    # norm direction
    direction = direction / np.sqrt(np.dot(direction, direction))

    # if the photon direction has the correct magnitude, then it will be on the unit sphere. All that is left to assert, then, is that it is sufficiently close to the direction vector.
    # take any orthogonal axis to direction. By symmetry it will be the same for any orthogonal axis.
    if (direction[0] == 0) and (direction[1]== 0):
        if (direction[2] == 0):
            #direction is zero vector
            raise ValueError('zero vector')
        else:
            axis = [0, 1, 0]
    else:
        axis = [-direction[1], direction[0], 0]

    # now we find a direction vector that is rotated the maximum distance from the beam direction
    rotationMatrix = transforms3d.axangles.axangle2mat(axis, delta)
    farVector = np.dot(rotationMatrix, direction)
    displacement = farVector - np.array(direction)
    maxDistanceSquared = np.dot(displacement, displacement)

    # Assert (correct magnitude) and (sufficiently close)

    # Prepares assessment of vector length
    directionMatrix = np.array(photons['dir']) # 2D array of photon directions
    dotProducts = np.sum((directionMatrix * directionMatrix), axis = 1) # the magnitudes^2 of each vector in an array
    correctSizes = [ (abs(1 - i) < 0.001) for i in dotProducts]

    # Prepares assessment of direction
    direction = np.array( np.append(direction, [0], axis = 0) )
    photonDisplacements = directionMatrix - direction
    displacementDistancesSquared = np.sum((photonDisplacements * photonDisplacements), axis = 1)
    withinRange = [(maxDistanceSquared > i) for i in displacementDistancesSquared]

    assert all(correctSizes) and all(withinRange)
