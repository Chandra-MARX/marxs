from .base import SimulationSequenceElement

class SimulationSetupError(Exception):
    pass


class SimulationSequence(SimulationSequenceElement):
    '''Base class for predefined simulation sequence.

    Parameters
    ----------
    sequence : list
        The elements of this list are all optical elements that process photons.
    preprocess_steps : list of callables
        Each callable must accept a photon table on input and return a photon table on output.
    postprocess_steps : list of callables
        Each callable must accept a photon table on input and return a photon table on output.

    Example
    -------
    The following example shows a complete marxs simulation.
    First, we import the required modules:

    >>> from marxs import source, optics
    >>> from marxs.simulator import SimulationSequence

    Then, we build up the parts of the simulation, source, pointing model and hardware
    of our instrument:

    >>> mysource = source.ConstantPointSource(coords=(30., 30.), flux=1e-3, energy=2.)
    >>> sky2mission = source.FixedPointing(coords=(30., 30.))
    >>> aper = optics.RectangleAperture(position=[50., 0., 0.])
    >>> mirr = optics.ThinLens(focallength=10, position=[10., 0., 0.])
    >>> ccd = optics.FlatDetector(pixsize=0.05)
    >>> sequence = [sky2mission, aper, mirr, ccd]
    >>> my_instrument = SimulationSequence(sequence)

    Finally, we run one set of photons through the instrument:

    >>> photons_in = mysource.generate_photons(1e5)
    >>> photons_out = my_instrument(photons_in)

    Now, let us check where the photons fall onthe detector:

    >>> set(photons_out['detpix_x'].round())
    set([19.0, 20.0])

    As expected, they fall right around the center of the detector (row 19 and 20 of a
    40 * 40 pixel detector).
    '''

    def __init__(self, sequence, preprocess_steps=[], postprocess_steps=[], **kwargs):
        for elem in sequence + preprocess_steps + postprocess_steps:
            if not callable(elem):
                raise SimulationSetupError('{0} is not callable.'.format(str(sequence[0])))
        self.sequence = sequence
        self.preprocess_steps = preprocess_steps
        self.postprocess_steps = postprocess_steps
        super(SimulationSequence, self).__init__(**kwargs)

    def process_photons(self, photons):
        for elem in self.sequence:
            for p in self.preprocess_steps:
                photons = p.process_photons(photons)
            photons = elem.process_photons(photons)
            for p in self.postprocess_steps:
                photons = p.process_photons(photons)
        return photons
