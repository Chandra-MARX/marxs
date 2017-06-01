# Licensed under GPL version 3 - see LICENSE.rst
from functools import wraps
from copy import copy

import numpy as np
from astropy.table import Table, Row

from ..math.pluecker import point_dir2plane, dir_point2line, intersect_line_plane
from ..math.utils import e2h, h2e
from ..base import SimulationSequenceElement, _parse_position_keywords

class OpticalElement(SimulationSequenceElement):
    '''Base class for all optical elements in marxs.

    This class cannot be used to instanciate an optical element directly, rather it serves as a
    base class from with other optical elements will be derived.

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

    def geometry(self, key):
        '''This function wraps access to the pos4d matrix.

        This is mostly a convenience method that gives access to vectors from the
        ``pos4d`` matrix in familiar terms with string labels:

        - ``center``: The ``center`` is the origin of the local coordiante system
          of the optical elemement. Typically, if will be the center of the
          active plane, e.g. the center of the mirror surface.
        - :math:`\hat v_{y,z}`: The box stretches in the y and z direction for
          :math:`\pm \hat v_y` and :math:`\pm \hat v_z`. In optics, the relevant
          interaction often happens on a surface, e.g. the surface of a mirror or
          of a reflection grating. In the defaul configuration, this surface is in
          the yz-plane.
        - :math:`\hat v_x`: The thickness of the box is not as important in many
          cases,
          but is useful to e.g. render the geometry or to test if elements collide.
          The box reaches from the yz-plane down to :math:`- \hat v_x`. Note, that
          this definition means the size parallel to the y and z axis is
          :math:`2 |\hat v_{y,z}|`, but only :math:`1 |\hat v_{x}|`.
        - :math:`\hat e_{x,y,z}`: When an object is initialized, it automatically
          adds unit vectors in the
          direction of :math:`\hat v_{x,y,z}` called :math:`\hat e_{x,y,z}`.
        - ``plane``: It also adds the homogeneous coordinates of the active plane
          as ``plane``.
        - Other labels get looked up in ``self._geometry`` and if the resulting
          value is a 4-d vector, it gets transformed with ``pos4d``.

        Not all these labels make sense for every optical element (e.g. a curved
        mirror does not really have a "plane").
        Access through this method is slower than direct indexing of ``self.pos4d``.
        '''

        if key == 'center':
            return self.pos4d[:, 3]
        elif key in ['v_x', 'e_x']:
            val = self.pos4d[:, 0]
        elif key in ['v_y', 'e_y']:
            val = self.pos4d[:, 1]
        elif key in ['v_z', 'e_z']:
            val = self.pos4d[:, 2]
        elif key == 'plane':
            return point_dir2plane(self.geometry('center'), self.geometry('e_x'))
        else:
            val = self._geometry[key]
            if isinstance(val, np.ndarray) and (val.shape[-1] == 4):
                val = np.dot(self.pos4d, val)
        if key[:2] == 'e_':
            return val / np.linalg.norm(val)
        else:
            return val

    def __init__(self, **kwargs):
        self.pos4d = _parse_position_keywords(kwargs)
        super(OpticalElement, self).__init__(**kwargs)

    def intersect(self, dir, pos):
        '''Calculate the intersection point between a ray and the element

        Parameters
        ----------
        dir : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the direction of the ray
        pos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of a point on the ray

        Returns
        -------
        intersect :  boolean array of length N
            ``True`` if an intersection point is found.
        interpos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the intersection point. Values are set
            to ``np.nan`` if no intersection point is found.
        interpos_local : `numpy.ndarray` of shape (N, 2)
            y and z coordinates in the coordiante system of the active plane.
        '''
        raise NotImplementedError

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
            4-d position of last interaction pf the photons with any optical element in
            homogeneous coordinates. Together with ``dir`` this determines the equation
            of the ray.
        energy : float
            Photon energy in keV.
        polarization : float
            Polarization angle of the photons.

        Returns
        -------
        dir : `numpy.ndarray`
            4-d direction vector of ray in homogeneous coordinates
        pos : `numpy.ndarray`
            4-d position of last interaction pf the photons with any optical element in
            homogeneous coordinates. Together with ``dir`` this determines the equation
            of the ray.
        energy : float
            Photon energy in keV.
        polarization : float
            Polarization angle of the photons.
        probability : float
            Probability that the photon passes this optical element. Set to 0 if the
            photon is absorbed, to 1 if it passes and to number between 0 and 1 to
            express a probability that the photons passes.
        other : floats
            One number per entry in `output_columns`.
        '''
        raise NotImplementedError
        return dir, pos, energy, polarization, probability, any, other, output, columns

    def __call__(self, photons):
        intersect_out = self.intersect(photons['dir'].data, photons['pos'].data)
        return self.process_photons(photons, *intersect_out)

    def process_photons(self, photons, intersect, interpos, intercoos):
        '''Simulate interaction of optical element with photons - vectorized.

        Derived classes should overwrite this function or `process_photon`.

        Parameters
        ----------
        photons: `astropy.table.Table` or `astropy.table.Row`
            Table with photon properties
        intersect, interpos, intercoos : array (N, 4)
            The array ``interpos`` contains the intersection points in the global
            coordinate system, ``intercoos`` in a local coordiante system (2d in most cases).

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
            self.add_output_cols(photons, self.loc_coos_name)
            # Add ID number to ID col, if requested
            if self.id_col is not None:
                photons[self.id_col][intersect] = self.id_num
            # Set position in different coordinate systems
            photons['pos'][intersect] = interpos[intersect]
            photons[self.loc_coos_name[0]][intersect] = intercoos[intersect, 0]
            photons[self.loc_coos_name[1]][intersect] = intercoos[intersect, 1]

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

        return photons


class FlatOpticalElement(OpticalElement):
    '''Base class for geometrically flat optical elements.

    Compared with `OpticalElement` this adds methods to make the implementation of
    flat elements easier. It adds a default `geometry`, a fast, vectorized method `intersect`,
    and a template to `process_photons`.

    Derived classes have the option to implement their own `process_photons` or, alternatively,
    they can implement a function called
    ``specific_process_photons(self, photons, intersect, interpos, intercoos)`` that returns a dictionary
    of the form ``{'column name': value, ...}`` where value is an array that holds one value for
    each photon that intersects the optical element. In the special case of ``probability`` the
    return value should only contain the probability assigned in **this** element. This value
    will be multiplied with the previous probabilities of each photon automatically.
    '''

    display = {'shape': 'box'}
    _geometry = {}

    loc_coos_name = ['y', 'z']
    '''name for output columns that contain the interaction point in local coordinates.'''

    def __init__(self, *args, **kwargs):
        super(FlatOpticalElement, self).__init__(*args, **kwargs)
        #copy class attribute to instance attribute
        self._geometry = copy(self._geometry)

    def intersect(self, dir, pos):
        '''Calculate the intersection point between a ray and the element

        Parameters
        ----------
        dir : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the direction of the ray
        pos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of a point on the ray

        Returns
        -------
        intersect :  boolean array of length N
            ``True`` if an intersection point is found.
        interpos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the intersection point. Values are set
            to ``np.nan`` if no intersection point is found.
        interpos_local : `numpy.ndarray` of shape (N, 2)
            y and z coordinates in the coordiante system of the active plane
            (not normalized to the dimensions of the element in question, but
            in absolute units).
        '''
        plucker = dir_point2line(h2e(dir), h2e(pos))
        interpos =  intersect_line_plane(plucker, self.geometry('plane'))
        vec_center_inter = - h2e(self.geometry('center')) + h2e(interpos)
        ey = np.dot(vec_center_inter, h2e(self.geometry('e_y')))
        ez = np.dot(vec_center_inter, h2e(self.geometry('e_z')))
        intersect = ((np.abs(ey) <= np.linalg.norm(self.geometry('v_y'))) &
                     (np.abs(ez) <= np.linalg.norm(self.geometry('v_z'))))
        if dir.ndim == 2:
            interpos[~intersect, :3] = np.nan
        # input of single photon.
        elif (dir.ndim == 1) and not intersect:
            interpos[:3] = np.nan
        return intersect, interpos, np.vstack([ey, ez]).T


class FlatStack(FlatOpticalElement):
    '''Convenience class for several flat, stacked optical elements.

    This class is meant to simplify the specification of a single physical
    element, that fullfills several logical functions, e.g. a detector can be seen
    as a a sequence of a contamination layer (which modifies the probability of a photon
    reaching the CCD), a QE filter (which modifies the probability of detecting the photon),
    and the pixelated CCD (which sorts the photons in pixels). All these things can be
    approximated as happening in the same physical spotlocation, and thus it is convenient to
    treat all three functions as one element.

    Parameters
    ----------
    elements : list of classes
        List of class names specifying the layers in the stack
    keywords : list of dicts
        Dictionaries specifying the properties of each layer (do not set the position
        of individual elements)

    Examples
    --------
    In this example, we will define a single flat CCD with a QE of 0.5 for all energies.

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
            # This line calls FlatOpticalElement.process_photons to add ID cols and local coos
            # if requested (this could also be done by any of the contained sequence elements,
            # but we want the user to be able to specify that for either of them).
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
