from .base import SimulationSequenceElement

class SimulationSetupError(Exception):
    pass


class SimulationSequence(SimulationSequenceElement):
    '''Base class for predefined simulation sequence.

    Parameters
    ----------
    sequence : list
        The elements of this list are all optical elements that process photons.
    preprocess_steps : list
        The elements of this list are functions or callable objects that accept a photon list as input
        and returns no output (but changing the photon list in place, e.g. adding meta-data is
        allowed)  (*default*: ``[]``). All ``preprocess_steps`` are run before *every* optical element
        in the sequence.
        An example would be a function that writes the photon list to disk as a backup before
        every optical element or prints some informational message.
        If your function returns a modified photon list, treat it as an optical element and place it
        in `sequence`.
    postprocess_steps : list
        See `preprecess_steps` except that thee steps are run *after* each sequnece element
         (*default*: ``[]``).


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
    >>> my_instrument = SimulationSequence(sequence=sequence)

    Finally, we run one set of photons through the instrument:

    >>> photons_in = mysource.generate_photons(1e5)
    >>> photons_out = my_instrument(photons_in)

    Now, let us check where the photons fall on the detector:

    >>> set(photons_out['detpix_x'].round())
    set([19.0, 20.0])

    As expected, they fall right around the center of the detector (row 19 and 20 of a
    40 * 40 pixel detector).
    '''

    def __init__(self, **kwargs):
        self.sequence = kwargs.pop('sequence')
        self.preprocess_steps = kwargs.pop('preprocess_steps', [])
        self.postprocess_steps = kwargs.pop('postprocess_steps', [])
        for elem in self.sequence + self.preprocess_steps + self.postprocess_steps:
            if not callable(elem):
                raise SimulationSetupError('{0} is not callable.'.format(str(elem)))
        super(SimulationSequence, self).__init__(**kwargs)

    def process_photons(self, photons):
        for elem in self.sequence:
            for p in self.preprocess_steps:
                p(photons)
            photons = elem(photons)
            for p in self.postprocess_steps:
                p(photons)
        return photons
