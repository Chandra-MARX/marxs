.. currentmodule:: marxs.source.source

.. _sources:

Define the source in a marxs run
================================

A "source" in marxs is anything that sends out X-ray photons. This can be an astrophyical object (such as a star or an AGN, but also a dust cloud that scatters photons from some other direction into our line-of-sight) or a man-made piece of hardware, such as a X-ray tube in the lab or a radioactive calibration source in a sattelite. An important distinction in Marxs is weather the source is located at a finite distance (lab source) or so far away that all rays can be treated as parallel (astrophysical source).

For each type of source, we need to specifiy the following properties:

- Total flux (or flux density)
- Spectrum of photon energies
- Polarization

Marxs offers many options to specify the flux, spectrum and polarization that are designed to make common use cases very easy, while allowing for arbitrarily complex models if needed. Examples are given in :ref:`sect-source-fluxenpol`, the formal specification is written in `Source`.

In addition, we need to give the location of the source and its size and shape (most of the currently implemented sources are point sources, but additional shapes will be added in the future):

- **Astrophysical source**: Needs Ra, Dec coordiantes on the sky and a pointing model to translate from sky coordinates to the coordiante system of the satellite. See :ref:`sect-source-radec`. 
- **Lab source**: Needs x,y,z coordianates in the coordinate system of the experiement as explained in :ref:`pos4d`. Lab source are described in more detail in :ref:`sect-source-lab`.


 .. _sect-source-fluxenpol:

Specify flux, energy and polarizarion for a source
--------------------------------------------------

The source flux, the energy and the polarization of sources are specified in the same way for astrophysical sources and lab sources. We show a few examples here and spell out the full specification below.

Flux
^^^^
.. todo::

   Currently, all fluxes are given assuming that the effective area of the instrument is 1 mm^2 and there is no mechanism to set the effective area to a different value.

The source flux can just be a number, giving the total counts / second (if no number is given, the default is ``flux=1``).

     >>> from __future__ import print_function
     >>> from marxs.source import PointSource
     >>> star = PointSource(coords=(23., 45.), flux=5.)
     >>> photons = star.generate_photons(20)
     >>> print(photons['time'][:6])
     time
     ----
     0.0
     0.2
     0.4
     0.6
     0.8
     1.0

This will generate 5 counts per second for 20 seconds with an absolutely constant (no Poisson) rate, so the photons list will contain 100 photons.

``flux`` can also be set to a function that takes the total exposure time as input and returns a list of times, one per photon. In the following example we show how to implement a `Poisson process <https://en.wikipedia.org/wiki/Poisson_process>`_ where the time intervals between two photons are `exponentially distributed <https://en.wikipedia.org/wiki/Poisson_process#Properties>`_ with an average rate of 100 events per second (so the average time difference between events is 0.01 s):

    >>> from scipy.stats import expon
    >>> def poisson_rate(exposuretime):
    ...     times = expon.rvs(scale=0.01, size=exposuretime * 0.01 * 2.)
    ...     return times[times < exposuretime]
    >>> star = PointSource(coords=(0,0), flux=poisson_rate)

Note that this simple implementation is incomplete (it can happen by chance that it does not generate enough photons). Marxs provides a better implementation called `~marxs.source.source.poisson_process` which will generate the appropriate function automatically given the expected rate:

    >>> from marxs.source.source import poisson_process
    >>> star = PointSource(coords=(11., 12.), flux=poisson_process(100.))

Energy
^^^^^^
Similarly to the flux, the input for ``energy`` can just be a number, which specifies the energy of a monochromatic source in keV (the default is ``energy=1``):

    >>> FeKalphaline = PointSource(coords=(255., -33.), energy=6.7)
    >>> photons = FeKalphaline.generate_photons(5)
    >>> print(photons['energy'])
    energy
    ------
       6.7
       6.7
       6.7
       6.7
       6.7

We can also specify a spectrum, by giving binned energy and flux density values. The energy values are taken as the *upper* egde of the bin; the first value of the flux density array is ignored since the lower bound for this bin is undefined. The spectrum can either be in the form of a (2, N) `numpy.ndarray` or it can be some type of table, e.g. an `astropy.table.Table` or a `dict <dict>` with columns named "energy" and "flux" (meaning: "flux density" in counts/s/unit area/keV). In the following exmaple, we specify the same spectrum in three differect ways (the plots look a little different because photon energies are randomly drawn from the spectrum, so there is a Poisson uncertainty):

.. plot:: pyplots/sourcespectrum.py
   :include-source:


If the input spectrum is in some type of file, e.g. fits or ascii, the `astropy.table.Table` `read/write interface <https://astropy.readthedocs.org/en/stable/io/unified.html>`_ offers a convenient way to read it into python:

    >>> from astropy.table import Table
    >>> spectrum = Table.read('AGNspec.dat', format='ascii')  # doctest: +SKIP
    >>> agn = PointSource(energy=spectrum)  # doctest: +SKIP

Lastly, "energy" can be a function that assigns energy values based on the timing of each photon. This allows for time dependent spectra. As an example, we show a function where the photon energy is 0.5 keV for times smaller than 5 s and 2 keV for larger times.
  
    >>> from marxs.source.source import Source
    >>> import numpy as np
    >>> def time_dependent_energy(t):
    ...     en = np.ones_like(t)
    ...     en[t <= 5.] = 0.5
    ...     en[t > 5.] = 2.
    ...     return en
    >>> mysource = Source(energy=time_dependent_energy)
    >>> photons = mysource.generate_photons(7)
    >>> print(photons['time', 'energy'])
    time energy
    ---- ------
     0.0    0.5
     1.0    0.5
     2.0    0.5
     3.0    0.5
     4.0    0.5
     5.0    0.5
     6.0    2.0

Polarization
^^^^^^^^^^^^
An unpolarized source can be created with ``polarization=None`` (this is also the default). In this case, a random polarization is assigned to every photon. The other options are very similar to "energy": Allowed are a constant value or a table of some form (see examples above) with two columns "angle" and "probability" (really "probability density") or a numpy array where the first column represents the angle and the second one the probability density. Here is an example where most polarizations are randomly oriented, but an orientation around :math:`35^{\circ}` (0.6 in radian) is a lot more likely.

    >>> angles = np.array([0., 0.5, 0.7, 2 * np.pi])
    >>> prob = np.array([1, 1., 8., 1.])
    >>> polsource = PointSource(coords=(0.,0.), polarization={'angle': angles, 'probability': prob})

Lastly, if polarization is a function, it will be called with time and energy as parameters allowing for time and energy dependent polarization distributions. The following function returns a 50% polarization fraction in the 6.4 keV Fe flourescence line after a a certain features comes into view at t=1000 s.

    >>> def polfunc(time, energy):
    ...     pol = np.random.uniform(high=2*np.pi, size=len(time))
    ...     ind = (time > 1000.) & (energy > 6.3) & (energy < 6.5)
    ...     # set half of all photons with these conditions to a specific polarization angle
    ...     pol[ind & (np.random.rand(len(time))> 0.5)] = 1.234
    ...     return pol
    >>> polsource = Source(energy=tablespectrum, polarization=polfunc)   # doctest: +SKIP

	
.. _sect-source-radec:

Specify the position for an astrophysical source
------------------------------------------------

An astrophysical source in Marxs must be followed by a pointing model as first optical element that translates the sky coordiantes into the coordinate system of the sattellite (see `pos4d`) and an entrace aperture that selects an initial position for each ray (all rays from astrophysical sources are parallel, thus the position of the source on the sky only determines the direction of a photon but not if it hits the left or the right side of a mirror). See :ref:`sect-apertures` for more details.


The following astropysical sources are included in marxs:

- `marxs.source.PointSource`
- `marxs.source.SymbolFSource`
	       
Sources can be used with the following pointing model:

- `marxs.source.FixedPointing`

.. _sect-source-lab:

Specify the position for a laboratory source
--------------------------------------------

Sources in the lab are specified in the same coordinate system used for all other optical elements, see `pos4d` for details.

The following laboratory sources are provided:

- `~marxs.source.FarLabPointSource`
- `~marxs.source.LabPointSource`
- `~marxs.source.LabPointSourceCone`

Design your own sources and pointing models
-------------------------------------------

The base class for all marxs sources is `Source`. The only method required for a source is ``generate_photons``. We recommend to look at the implementation of the included sources to see how this is done best.

- `marxs.source.Source`


Reference/API
=============

.. automodapi:: marxs.source
