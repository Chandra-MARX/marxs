# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

class RandomArbitraryPdf(object):
    '''Take random draw from an arbitrary (and arbitrarily binned) pdf.

    The PDF is approximated as piecewice constant.
    The object is initialized with a parameters that describe the PDF, the object can
    then be called with the number of random samples that are required.

    Parameters
    ----------
    x : np.array
        **Upper** bin edge for input bins
    pdf : np.array
        Value of the pdf for each bin. ``pdf[0]`` is ignored, since the lower edge
        of that bin is undefined.
    randomize_in_bin : bool
        If ``True`` randomize the return value over each bin. If ``False`` the return
        value will be exactly the **upper** bin edge.
    sort : bool
        When calculating the cdf from a pdf that covers a large dynamical range,
        round-off errors may occur. If ``True`` this sorts the input bins in size
        (keeping an index for reverse the operation later) to avoid numerical errors.

    Examples
    --------
    First, and example with two bins (1-2 and 2-3). The probability density for each
    bin is 1 and 6, respectively (the 0.2 is ignored, because the range XXX - 1 is not
    a valid bin since the lower bound is undefined).

    >>> import numpy as np
    >>> # Set seed so this example returns exact same numbers every time
    >>> np.random.seed(0)
    >>> a = RandomArbitraryPdf(np.array([1,2,3]), np.array([0.2,1,6]))
    >>> a(10)
    array([ 2.79172504,  2.52889492,  2.56804456,  2.92559664,  2.07103606,
            2.0871293 ,  2.0202184 ,  2.83261985,  2.77815675,  2.87001215])

    As you can see, in this case the chance to draw from the 1-2 interval is only
    1/7 and it did not happen this time.

    If, instead, we want to have exact (upper) bin edges returned it looks like this:
    >>> a = RandomArbitraryPdf(np.array([1,2,3]), np.array([0,1,6]), randomize_in_bin=False)
    >>> a(10)
    array([3, 3, 3, 3, 2, 3, 3, 3, 3, 3])

    As expected, most numbers are 3, with a 2 mixed in.

    References
    ----------
    https://en.wikipedia.org/wiki/Pseudorandom_number_generator#Non-uniform_generators

    http://stackoverflow.com/questions/21100716/
    '''
    def __init__(selv, x, pdf, randomize_in_bin=True, sort=True):
        if not len(x) == len(pdf):
            raise ValueError('x and pdf must have same number of elements.')
        if not np.all(np.array(pdf) >= 0):
            raise ValueError('pdf cannot have negative elements.')

        selv.x = np.asarray(x)
        selv.bin_width = np.hstack(([0], np.diff(x)))
        if not np.all(selv.bin_width >=0):
            raise ValueError('x must be input in increasing order.')
        pdf = np.asarray(pdf) * selv.bin_width
        selv.sort = sort
        selv.randomize_in_bin = randomize_in_bin

        # sort the pdf - otherwise bins with small numbers might be lost to round-off errors
        # idea is from http://stackoverflow.com/questions/21100716/
        if selv.sort:
            selv.sortindex = np.argsort(pdf)
            selv.pdf = pdf[selv.sortindex]
        else:
            selv.pdf = pdf
        # cumulative distribution function
        selv.cdf = np.cumsum(selv.pdf)

    def __call__(selv, N):
        """Draw from the distribution function. See docstring of class."""
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high=selv.cdf[-1], size=N)
        # Now here is the difficult and comparatively expensive part:
        # We need a reverse lookup to find the bin in the cdf so that we an use it
        # to map this back to the x values of the pdf
        # np.searchsorted is a binary search with O(log n).
        index = np.searchsorted(selv.cdf, choice)
        #if necessary, map the indices back to their original ordering
        if selv.sort:
            index = selv.sortindex[index]
        if selv.randomize_in_bin:
            return selv.x[index - 1] + selv.bin_width[index] * np.random.rand(N)
        else:
            return selv.x[index]
