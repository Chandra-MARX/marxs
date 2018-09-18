# Licensed under GPL version 3 - see LICENSE.rst
from functools import wraps

import numpy as np
from astropy.table import Table, Row

from ..math.geometry import Geometry, FinitePlane
from ..base import SimulationSequenceElement

class OpticalElement(SimulationSequenceElement):
    '''Base class for all optical elements in marxs.

    This class cannot be used to instanciate an optical element directly,
    rather it serves as a base class from with other optical elements will be
    derived.

    At the very minumum, any derived class needs to implement `__call__` which
    typically calls `intersect` and either `process_photon` or
    `process_photons`. If the interaction with the photons (e.g. scattering of
    a mirror surface) can be implemented in a vectorized way using numpy array
    operations, the derived class should overwrite `process_photons`
    (`process_photon` is not used in this case).  If no vectorized
    implementation is available, it is sufficient to overwrite
    `process_photon`.  Marxs will call `process_photons`, which (if not
    overwritten) contains a simple for-loop to loop over all photons in the
    array and call `process_photon` on each of them.

    '''

    default_geometry = FinitePlane
    '''If no geometry is specified on initialization, an instance of this class will be used.'''

    # SimulationSequenceElement has display none, but now we have a geometry
    # so we don't need that default here.
    display = {}

    def __init__(self, **kwargs):

        geometry = kwargs.pop('geometry', self.default_geometry)
        if isinstance(geometry, Geometry):
            self.geometry = geometry
        elif issubclass(geometry, Geometry):
            self.geometry = geometry(kwargs)

        super(OpticalElement, self).__init__(**kwargs)

        if not hasattr(self, "loc_coos_name"):
            self.loc_coos_name = self.geometry.loc_coos_name

    @property
    def pos4d(self):
        return self.geometry.pos4d

    def process_photon(self, dir, pos, energy, polarization):
        '''Simulate interaction of optical element with a single photon.

        This is called from the `process_photons` method in a loop over all
        photons. That method also collects the output values and inserts them
        into the photon list. ``process_photon`` can return any number of
        values in additon to the required dir, pos, etc.. Define a class
        attribute ``output_columns`` as a list of strings to determine how into
        which column these numbers should be inserted.

        Parameters
        ----------
        dir : `numpy.ndarray`
            4-d direction vector of ray in homogeneous coordinates
        pos : `numpy.ndarray`
            4-d position of last interaction pf the photons with any optical
            element in homogeneous coordinates. Together with ``dir`` this
            determines the equation of the ray.
        energy : float
            Photon energy in keV.
        polarization : float
            Polarization angle of the photons.

        Returns
        -------
        dir : `numpy.ndarray`
            4-d direction vector of ray in homogeneous coordinates
        pos : `numpy.ndarray`
            4-d position of last interaction pf the photons with any optical
            element in homogeneous coordinates. Together with ``dir`` this
            determines the equation of the ray.
        energy : float
            Photon energy in keV.
        polarization : float
            Polarization angle of the photons.
        probability : float
            Probability that the photon passes this optical element. Set to 0
            if the photon is absorbed, to 1 if it passes and to number between
            0 and 1 to express a probability that the photons passes.
        other : floats
            One number per entry in `output_columns`.

        '''
        raise NotImplementedError
        return dir, pos, energy, polarization, probability, any, other, output, columns

    def __call__(self, photons):
        intersect_out = self.geometry.intersect(photons['dir'].data,
                                                photons['pos'].data)
        return self.process_photons(photons, *intersect_out)

    def process_photons(self, photons, intersect, interpos, intercoos):
        '''Simulate interaction of optical element with photons - vectorized.

        Derived classes should overwrite this function or `process_photon`.

        Parameters
        ----------
        photons: `astropy.table.Table` or `astropy.table.Row`
            Table with photon properties
        intersect : array
            Boolean array marking which photons should be processed by this
            element.
        interpos, intercoos : array (N, 4)
            The array ``interpos`` contains the intersection points in the
            global coordinate system, ``intercoos`` in a local coordiante
            system (2d in most cases).

        Returns
        -------
        photons: `astropy.table.Table` or `astropy.table.Row`
            Table with photon properties.
            If possible, the input table is modified in place, but in some
            cases this might not be possible and the returned Table may be
            a copy. Do not rely on either - use ``photons.copy()`` if you want
            to ensure you are working with an independent copy.

        '''
        if intersect.sum() > 0:
            if hasattr(self, "specific_process_photons"):
                outcols = self.specific_process_photons(photons, intersect, interpos, intercoos)
                self.add_output_cols(photons, list(outcols.keys()))
                for col in outcols:
                    if col == 'probability':
                        photons[col][intersect] *= outcols[col]
                    else:
                        photons[col][intersect] = outcols[col]

            elif hasattr(self, "process_photon"):
                if isinstance(photons, Row):
                    photons = Table(photons)
                outcols = ['dir', 'pos', 'energy', 'polarization', 'probability'] + self.output_columns
                n_intersect = intersect.nonzero()[0]
                for photon, i in zip(photons[intersect], n_intersect):
                    outs = self.process_photon(photon['dir'], photon['pos'],
                                               photon['energy'],
                                               photon['polarization'])
                    for a, b in zip(outcols, outs):
                        if a == 'probability':
                            photons['probability'][i] *= b
                        else:
                            photons[a][i] = b
            else:
                raise AttributeError('Optical element must have one of three: specific_process_photons, process_photon, or override process_photons.')

            self.add_output_cols(photons, self.loc_coos_name)
            # Add ID number to ID col, if requested
            if self.id_col is not None:
                photons[self.id_col][intersect] = self.id_num
            # Set position in different coordinate systems
            photons['pos'][intersect] = interpos[intersect]
            photons[self.loc_coos_name[0]][intersect] = intercoos[intersect, 0]
            photons[self.loc_coos_name[1]][intersect] = intercoos[intersect, 1]

        return photons


class FlatOpticalElement(OpticalElement):
    pass


class FlatStack(FlatOpticalElement):
    '''Convenience class for several flat, stacked optical elements.

    This class is meant to simplify the specification of a single physical
    element, that fullfills several logical functions, e.g. a detector can be
    seen as a a sequence of a contamination layer (which modifies the
    probability of a photon reaching the CCD), a QE filter (which modifies the
    probability of detecting the photon), and the pixelated CCD (which sorts
    the photons in pixels). All these things can be approximated as happening
    in the same physical spotlocation, and thus it is convenient to treat all
    three functions as one element.

    Parameters
    ----------
    elements : list of classes
        List of class names specifying the layers in the stack
    keywords : list of dicts
        Dictionaries specifying the properties of each layer (do not set the
        position of individual elements)

    Examples
    --------
    In this example, we will define a single flat CCD with a QE of 0.5 for all
    energies.

    >>> from marxs.optics import FlatStack, EnergyFilter, FlatDetector
    >>> myccd = FlatStack(position=[0, 2, 2], zoom=2,
    ...     elements=[EnergyFilter, FlatDetector],
    ...     keywords=[{'filterfunc': lambda x: 0.5}, {'pixsize': 0.05}])

    '''

    def __init__(self, **kwargs):
        elements = kwargs.pop('elements')
        keywords = kwargs.pop('keywords')
        super(FlatStack, self).__init__(**kwargs)
        self.elements = []
        for elem, k in zip(elements, keywords):
            self.elements.append(elem(pos4d=self.pos4d, **k))

    def specific_process_photons(self, *args, **kwargs):
        return {}

    def process_photons(self, photons, intersect=None, interpos=None, intercoos=None):
        '''
        Parameters
        ----------
        intersect, interpos, intercoos : array (N, 4)
            The array ``interpos`` contains the intersection points in the global
            coordinate system, ``intercoos`` in the local (y,z) system of the grating.
        '''
        if intersect.sum() > 0:
            # This line calls FlatOpticalElement.process_photons to add ID cols
            # and local coos if requested (this could also be done by any of
            # the contained sequence elements, but we want the user to be able
            # to specify that for either of them).
            photons = super(FlatStack, self).process_photons(photons, intersect, interpos, intercoos)
            for e in self.elements:
                photons = e.process_photons(photons, intersect, interpos, intercoos)

        return photons


def photonlocalcoords(f, colnames=['pos', 'dir']):
    '''Decorator for calculation that require a local coordinate system

    This is specifically meant to wrap the :meth:`process_photons` methods of
    any :class:`OpticalElement`; the current implementation expects the call
    signature of :meth:`process_photons`.

    This decorator transforms coordinates from the global system to the local
    system before a function call and back to the global system again after
    the function call.

    Parameters
    ----------
    f : callable with signature ``f(self, photons, *args, **kwargs)``
        The function to be decorated. In the signature, ``photons`` is a
        `astropy.table.Table`.
    colnames : list of strings
        List of all column names in the photon table to be transformed into a
        different coordinate system.
    '''
    @wraps(f)
    def wrapper(self, photons, *args, **kwargs):
        # transform to coordsys if single instrument
        invpos4d = np.linalg.inv(self.pos4d)
        for n in colnames:
            photons[n] = np.einsum('...ij,...j', invpos4d, photons[n])
        photons = f(self, photons, *args, **kwargs)
        # transform back into coordsys of satellite
        for n in colnames:
            photons[n] = np.einsum('...ij,...j', self.pos4d, photons[n])
        return photons

    return wrapper
