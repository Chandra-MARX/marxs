# Licensed under GPL version 3 - see LICENSE.rst
'''
.. todo::
   Add a mechanism for unit conversion. This expects all distances in mm and all energies in keV.

.. todo::
   Do something about the effective areas. Can be read out after mirror is initialized.

.. todo::
   Make a setup.py parameter for the marxsource code and marx compiled binary location
'''
import os
import numpy as np
from astropy.table import Table, Column, join
from astropy.extern import six

from ..math.pluecker import h2e, e2h
from ..math.polarization import parallel_transport
from .base import OpticalElement, photonlocalcoords
from .aperture import BaseAperture

try:
    from _marx import lib as marx
    from _marx import ffi
    HAS_MARX = True
except ImportError:
    HAS_MARX = False


class MarxError(Exception):
    '''Error in the compiled Marx C module'''
    pass


class MarxMirror(OpticalElement, BaseAperture):
    '''Interface to MARX mirror module

    This class provides an interface to the `MARX <http://space.mit.edu/ASC/marx/>`_
    mirror module. It requires
    both the MARX source code and compiled MARX binaries as explained in the
    :ref:`Installation instructions for MARXS <sect-installmarxccode>`
    When this model
    is initialized, it requires the path and filename to a MARX setup file.

    The default geometry is such that the focal point is at the origin of the
    coordinate system and the optical axis is along the x-axis, such that
    photons travel from +infinity towards the origin.

    A MarxMirror object does not only act as a mirror, it fulfills the function
    of an aperture at the same time, no further aperture class should be used
    in the same simulation.

    The model reads the relevant MARX mirror model parameters from that file;
    in this way it supports all mirror models that the traditional MARX version
    implements, most notable the "HRMA" and "IXO" mirrors. Details on the
    "HRMA" mirror (which is used to simulate Chandra) can be found at
    http://space.mit.edu/ASC/marx/indetail/hardwaremodel.html#hrma-model . The
    "IXO" mirror model is a simplified version with some generalizations that
    works very similar in general, but is not documented because (from a MARX
    standpoint) it is only meant for developers.

    Parameters
    ----------
    parfile : string
        Path and filename of a MARX parameter file that sets all MARX
        parameters for the mirror model.
    '''

    def __init__(selv, parfile, **kwargs):
        # If the state shared between different object that use the s
        # same C module? In that case I need to add a lock so that only
        # one object of this class can exist at any one time.
        if not HAS_MARX:
            raise MarxError('MARX C code is not available. Please see installation instructions.')
        if not os.path.isfile(parfile):
            raise IOError('MARX parameter file {0} does NOT exist.'.format(parfile))
        else:
            selv.parfile = parfile
        if six.PY2:
            selv.cparfile = marx.pf_open_parameter_file(parfile, 'r')
        else:
            selv.cparfile = marx.pf_open_parameter_file(bytes(parfile, 'utf8'), bytes('r', 'utf8'))
        out = marx.marx_mirror_init(selv.cparfile)
        if out < 0:
            raise MarxError('Mirror cannot be initialized. Probably missing parameters or syntax error in {0}.'.format(parfile))

        super(MarxMirror, selv).__init__(**kwargs)

    @staticmethod
    def _table2c(photons):
        # Arrays assigned here in python need to keep a reference
        # somewhere, otherwise they would be garbage collected
        # and the pointer would suddenly be invalid.
        # To do so, this function adds them to the photons table header.
        # If this turns out to be too slow, then I have to move it to
        # Cython that can access the native numpy array memory.
        n = len(photons)
        cp = ffi.new('Marx_Photon_Attr_Type[]', n)
        keep_cffi_pointers = {}
        keep_cffi_pointers['cp'] = cp
        for i in range(n):
            cp[i].energy = photons['energy'][i]
            pos = h2e(photons['pos'][i])
            cp[i].x.x = pos[0]
            cp[i].x.y = pos[1]
            cp[i].x.z = pos[2]
            dir = h2e(photons['dir'][i])
            cp[i].p.x = dir[0]
            cp[i].p.y = dir[1]
            cp[i].p.z = dir[2]
            cp[i].arrival_time = photons['time'][i]
        if 'tag' not in photons.colnames:
            photons.add_column(Column(name='tag', data=np.arange(n)))
        for i, t in zip(range(n), photons['tag']):
            cp[i].tag = t

        c_photon_list = ffi.new('Marx_Photon_Type *')
        c_photon_list.attributes = cp
        c_photon_list.n_photons = n
        c_photon_list.max_n_photons = n
        c_photon_list.total_time = np.max(photons['time'])
        c_photon_list.start_time = np.min(photons['time'])

        sorted_index = np.argsort(photons['energy'].data)
        sorted_index = np.ascontiguousarray(sorted_index, dtype=np.uintc)
        keep_cffi_pointers['sorted_index'] = sorted_index  # keep alive
        c_photon_list.sorted_index = ffi.cast('unsigned int*', sorted_index.ctypes.data)

        sorted_energies = np.sort(photons['energy'].data)
        sorted_energies = np.ascontiguousarray(sorted_energies, dtype=np.float)
        keep_cffi_pointers['sorted_energies'] = sorted_energies  # keep alive
        c_photon_list.sorted_energies = ffi.cast('double*', sorted_energies.ctypes.data)

        c_photon_list.num_sorted = n  # all photons in list are valid
        # Fields not mentioned here are irrelevant for the mirror
        # module and can have any value they are initialized to.

        return c_photon_list, keep_cffi_pointers

    @staticmethod
    def _c2table(c_photon_list):
        '''

        We'll first collect all the data in numpy arrays and make
        a new table. Might not be the most efficient solution,
        but should work for now.
        Make work first, improve later.

        To-Do: keep absorbed?
        '''
        n_valid = c_photon_list.num_sorted
        cp = c_photon_list.attributes

        energy = np.empty(n_valid)
        time = np.empty(n_valid)
        tag = np.empty(n_valid, dtype=int)
        pos = np.empty((n_valid, 3))
        dir = np.empty((n_valid, 3))
        unreflected = np.empty(n_valid, dtype=bool)
        vblocked = np.empty(n_valid, dtype=bool)
        shell = np.empty(n_valid, dtype=int)

        for i in range(n_valid):
            energy[i] = cp[i].energy
            pos[i, 0] = cp[i].x.x
            pos[i, 1] = cp[i].x.y
            pos[i, 2] = cp[i].x.z
            dir[i, 0] = cp[i].p.x
            dir[i, 1] = cp[i].p.y
            dir[i, 2] = cp[i].p.z
            time[i] = cp[i].arrival_time
            tag[i] = cp[i].tag
            unreflected[i] = cp[i].flags & marx.PHOTON_UNREFLECTED
            vblocked[i] = cp[i].flags & marx.PHOTON_MIRROR_VBLOCKED
            shell[i] = cp[i].mirror_shell

        pos = e2h(pos, 1)
        dir = e2h(dir, 0)
        photons = Table([pos, dir, energy, time, tag,
                         unreflected, vblocked, shell],
                        names=['pos', 'dir', 'energy', 'time', 'tag',
                               'unreflected', 'mirror_vblocked', 'mirror_shell'])
        return photons

    @photonlocalcoords
    def _process_photons_in_c(selv, photons, verbose):
        '''Wrap the MARX C module'''
        c_photon_list, keep_cffi_pointers = selv._table2c(photons)
        out = marx.marx_mirror_reflect(c_photon_list, verbose)
        if out != 0:
            raise MarxError('Error in marx_mirror_reflect.')
        return selv._c2table(c_photon_list)

    def __call__(selv, photons_in, verbose=0):
        selv.add_colpos(photons_in)
        new_photons = selv._process_photons_in_c(photons_in, verbose)
        photons = join(new_photons, photons_in, keys='tag',
                       uniq_col_name='{col_name}{table_name}',
                       table_names=['', '_beforemirror'])
        # Probability column needs special treatment, because the probability
        # is not just replaced with the new number, it's multiplicative.
        # First, match tag in to tag after join (which may have fewer photons)
        insorted = np.argsort(photons_in['tag'])
        ypos = np.searchsorted(photons_in['tag'][insorted], photons['tag'])
        indices = insorted[ypos]
        photons['probability'] *= photons_in['probability'][indices]
        photons['probability'][photons['unreflected'] | photons['mirror_vblocked']] = 0
        pol = parallel_transport(photons_in['dir'][insorted], photons['dir'],
                                 photons_in['polarization'][insorted])
        photons['polarization'] = pol
        return photons

    @property
    def area(selv):
        '''Area of the aperture.

        This does not take into account any projection effects for
        apertures that are not perpendicular to the optical axis.
        '''
        return marx.Marx_Mirror_Geometric_Area





# Turn into a test?
# pf = marx.pf_open_parameter_file('./marxs/optics/hrma.par', 'r')
# out = marx.marx_mirror_init(pf)
# out should be 0, but I get 2 ?!?
# Will continue and leave that mystery as a mystery for now, but I bet I'll
# need to get back to it.
