class SimulationSetupError(Exception):
    pass


class Simulator(object):
    '''
    .. todo::
    This needs to take control of the output columns.
    I want the optical elements to be able to add columns as needed, so
    Simulator needs to go through and rename them or pass in only the
    necessary columns (pos, dir, energy, pol, maybe prop, tag).
    Decide how to handle prob. Multiply when relevant or after output.
    Introduce unique tag for photon. Useful for photon splitting.
    In source or tagging stage after that?

    Example:

    >>> from marxs import source
    >>> mysource = source.ConstantPointSource(coords=(30., 30.), flux=1e-4, energy=2.)
    >>> sky2mission = source.FixedPointing(coords=(30., 30.))

    #sequence = [,
    #            source.Sky2Mission(30., 30.),
    #            mirror.GaussianBlur(0.5),
    #            detector.InfiniteCCD(0.5)]
    '''

    def __init__(self, sequence):
        if not callable(getattr(sequence[0], 'generate_photon', None)):
            raise SimulationSetupError('{0} cannot generate photons.'.format(str(sequence[0])))
        if not callable(getattr(sequence[-1], 'save')):
            raise SimulationSetupError('{0} does not have a method to output results.'.format(str(sequence[-1])))

        self.sequence = sequence

    def run():
        NotImplementedError
