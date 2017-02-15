# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from ..random import RandomArbitraryPdf

# Any number will do. Just make it repeatable.
np.random.seed(12324)

def test_return_right_pdf():
    x = np.arange(10.)
    f = x ** 2.
    rand = RandomArbitraryPdf(x, f)
    draws = rand(10000)
    draws.sort()
    # there should be very few small numbers
    assert draws[5000] > 5
    # but a few of them
    assert draws[2] <= 3
    # check range
    assert draws[0] >=0
    assert draws[-1] < 10.

def test_unequal_bins():
    '''Almost all results should be from the 1-100 range because we are talking
    probability DENSITY.'''
    rand = RandomArbitraryPdf(np.array([0,1,100]), np.ones(3))
    draws = rand(10000)
    draws.sort()
    assert draws[1000] > 1
