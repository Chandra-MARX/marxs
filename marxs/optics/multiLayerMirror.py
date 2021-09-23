# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.io import ascii

from .base import FlatOpticalElement, FlatStack
from ..math.utils import norm_vector, e2h, h2e


class FlatBrewsterMirror(FlatOpticalElement):
    '''Flat mirror operated at the Brewster angle.

    Calculation of the Fresnel coefficients can be computationally
    intense and also requires knowledge of the refractive index for
    the appropriate material.  The ``FlatBrewsterMirror`` simplifies
    this for a mirror that is known to be operated at the Brewster
    angle.

    This mirror assumes that all photons arrive at the Brewster angle
    where only s (senkrecht = direction perpendicular to plane of
    incidence) polarisation is reflected.  It also assumes that all
    photons that are not reflected (i.e. those that are transmitted)
    are lost. No transmitted photons are returned, instead the
    probability of the reflected photons is adjusted to account for
    this overall loss.

    '''
    display = {'color': (0., 1., 0.),
               'shape': 'box',
               'box-half': '+x',
               }

    def fresnel(self, photons, intersect, intersection, local):
        '''The incident angle can easily be calculated from e_x and photons['dir'].

        Returns
        -------
        refl_s, refl_p : np.array or float
            Reflection probability for s and p polarized photons.
            Typically, the number will depend on the incident angle and energy
            of each photon and thus the return value will be a vector.
        '''
        return 1., 0.

    def specific_process_photons(self, photons, intersect, intersection,
                                 local):
        directions = norm_vector(photons['dir'].data[intersect])
        # save the direction of the incoming photons as beam_dir
        beam_dir = h2e(directions)

        # reflect the photons (change direction) by transforming to
        # local coordinates
        pos4d_inv = np.linalg.inv(self.pos4d)
        directions = directions.T
        directions = np.dot(pos4d_inv, directions)
        directions[0, :] *= -1
        directions = np.dot(self.pos4d, directions)
        new_beam_dir = directions.T

        # split polarization into s and p components
        # - s is s polarization (perpendicular to plane of incidence)
        # - p is p polarization (in the plane of incidence)

        # First, make basis vectors.
        v_s = np.cross(beam_dir, self.geometry['e_x'][0:3])
        v_s /= np.linalg.norm(v_s, axis=1)[:, np.newaxis]
        v_p = np.cross(beam_dir, v_s)

        polarization = h2e(photons['polarization'].data[intersect])
        p_v_s = np.einsum('ij,ij->i', polarization, v_s)
        p_v_p = np.einsum('ij,ij->i', polarization, v_p)

        fresnel_refl_s, fresnel_refl_p = self.fresnel(photons, intersect,
                                                      intersection, local)
        # Calculate new intensity ~ (E_x)^2 + (E_y)^2
        Es2 = fresnel_refl_s * p_v_s ** 2
        Ep2 = fresnel_refl_p * p_v_p ** 2
        intensity = Es2 + Ep2
        # Sometimes, the numerics in the square make probabilities > 1
        # So, we clip, but raise an error is it's >> 1
        if np.any(intensity > 1.001):
            raise ValueError('Intensity cannot be > 1')

        # parallel transport of polarization vector
        # v_s stays the same by definition
        new_v_p = np.cross(h2e(new_beam_dir), v_s)
        new_pol = norm_vector(-Es2[:, np.newaxis] * v_s +
                              Ep2[:, np.newaxis] * new_v_p)

        return {'dir': new_beam_dir,
                'probability': np.clip(intensity, 0, 1),
                'polarization': e2h(new_pol, 0)}


class MultiLayerEfficiency(FlatOpticalElement):
    '''The Multilayer mirror with varying layer thickness along one axis

    The distance between layers (and thus best reflected energy) changes along
    the local y axis.
    All reflectivity data is assumed to be for a single, desired angle. There
    is currently no way to enter varying reflection that depends on the angle
    of incidence.

    Provide reflectivity data in a file with columns:

    - 'X(mm)' - position along the "changing" axis (local y axis)
    - 'Peak lambda' - wavelength with maximum reflection at a given position
    - 'Peak' - maximum reflection at a given position
    - 'FWHM(nm)' - full width half max, measure of width of reflection
      Gaussian peaks

    Provide polarization data in a file with columns:

    - 'Photon energy' - energy of the photon in keV
    - 'Polarization' - Fraction polarized in the more reflective direction, so
      that randomly polarized light would have a value of 0.5.

    Parameters
    ----------
    reflFile: string
        path, filename, and .txt extension for reflection data file
    testedPolarization: string
        path, filename, and .txt to a text file containing a table
        with photon energy and fraction polarization for the light
        used to test the mirrors and create the reflectivity file

    '''
    def __init__(self, **kwargs):
        self.fileName = kwargs.pop('reflFile')
        self.polFile = kwargs.pop('testedPolarization')
        super().__init__(**kwargs)

    def interp_files(self, photons, local):
        # read in correct reflecting probability file, now in table format
        reflectFile = ascii.read(self.fileName)

        # find reflectivity adjustment due to polarization of light in
        # reflectivity testing
        polarizedFile = ascii.read(self.polFile)
        tested_polarized_fraction = np.interp(photons['energy'], polarizedFile['Photon energy'] / 1000, polarizedFile['Polarization'])

        # find probability of being reflected due to position
        local_x = local[:, 0] / np.linalg.norm(self.geometry['v_y'])
        local_coords_in_file = reflectFile['X(mm)'] / np.linalg.norm(self.geometry['v_y']) - 1
        # interpolate 'Peak lambda', 'Peak' [reflectivity],
        # and 'FWHM(nm)' to the actual photon positions
        peak_wavelength = np.interp(local_x, local_coords_in_file,
                                    reflectFile['Peak lambda'])
        max_refl = np.interp(local_x, local_coords_in_file,
                             reflectFile['Peak']) / tested_polarized_fraction
        spread_refl = np.interp(local_x, local_coords_in_file,
                                reflectFile['FWHM(nm)'])

        return peak_wavelength, max_refl, spread_refl

    def specific_process_photons(self, photons, intersect, intersection,
                                 local):
         # wavelength is in nm assuming energy is in keV
        wavelength = 1.23984282 / photons['energy'].data[intersect]
        peak_wavelength, max_refl, spread_refl = self.interp_files(photons[intersect], local[intersect])

        # the standard deviation squared of the Gaussian reflectivity functions
        # of each photon's wavelength
        c_squared = (spread_refl ** 2) / (8. * np.log(2))
        # skip the case when there is no Gaussian
        # (this is assumed to just be the zero function)
        c_is_zero = (c_squared == 0)

        refl_prob = np.zeros(len(wavelength))
        refl_prob[~c_is_zero] = max_refl[~c_is_zero] * np.exp(-((wavelength[~c_is_zero] - peak_wavelength[~c_is_zero]) ** 2) / (2 * c_squared[~c_is_zero]))
        return {'probability': refl_prob / 100}


class MultiLayerMirror(FlatStack):
    def __init__(self, **kwargs):
        super().__init__(elements=[FlatBrewsterMirror,  MultiLayerEfficiency],
                         keywords=[{},
                                   {'reflFile': kwargs.pop('reflFile'),
                                    'testedPolarization': kwargs.pop('testedPolarization')}],
                         **kwargs)
