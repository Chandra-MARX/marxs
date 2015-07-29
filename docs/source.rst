.. module:: marxs.source

.. _sources:

Define the source in a marxs run
================================

A "source" in marxs is anything that sends out X-ray photons. This can be an astrophyical object (such as a star or an AGN, but also a dust cloud that scatters photons from some other direction into our line-of-sight) or a man-made piece of hardware, such as a X-ray tube in the lab or a radioactive calibration source in a sattelite. An important distinction in marx is weather the source is located at a finite distance (lab source) or so far away that all rays can be treated as parallel (astrophysical source).

For each type of source, we need to specifiy the following properties:

- Total flux (or flux density)
- Spectrum of photon energies
- Polarization

Marxs offers many options to specify the flux, spectrum and polarization that are designed to make common use cases very easy, while allowing for arbitrarily complex models if needed, e.g. a spectrum and polarization that changes continuously over time due to a flare in the lightcurve. Examples and details are given in :ref:`sect-source-fluxenpol`.

In addition, we need to give the location of the source and its size and shape (most of the currently implemented sources are point sources, but additional shapes will be added in the future):

- **Astrophysical source**: Needs Ra, Dec coordiantes on the sky and a pointing model to translate from sky coordinates to the coordiante system of the satellite. See :ref:`sect-source-radec`. 
- **Lab source**: Needs x,y,z coordianates in the coordinate system of the experiement as explained in :ref:`pos4d`. Lab source are described in more detail in :ref:`sect-source-lab`.


 .. _sect-source-fluxenpol:

Specify flux, energy and polarizarion for a source
--------------------------------------------------

The source flux, the energy and the polarization of sources are specified in the same way for astrophysical source and lab source. We show a few examples here and spell out the full specification below.

Flux
^^^^
.. todo::

   Currently, all fluxes are given assuming that the effective area of the instrument is 1 mm^2 and there is no mechanism to set the effective area to a different value.

The source flux can just be a number, giving the total counts / second.

     >>> from marxs.optics import PointSource
     >>> star = PointSource(flux=5.)
     >>> photons = star.generate_photons(20)

This will generate 5 counts per second for 20 seconds with an absolutely constant (no Poisson) rate, so the photons list will contain 100 photons. ``flux`` can also be set to a function that takes the total exposure time as input and returns a list of times, one per photon. In the following example we show how to implement a `Poisson process <https://en.wikipedia.org/wiki/Poisson_process>`_ where the time intervals between two photons are `exponentially distributed <https://en.wikipedia.org/wiki/Poisson_process#Properties>`_ with an average rate of 100 events per second (that is 0.01 s between events):

    >>> from scipy.stats import expon
    >>> def poisson_rate(exposuretime):
    ...     times = expon.rvs(scale=0.01, size=exposuretime * 0.01 * 2.)
    ...     return times[times < exposuretime]
    >>> star = PointSource(flux=poisson_rate)

Note that this simple implementation is incomplete (it can happen by chance that it does not generate enough photons), instead we recommend using `poisson_process` which will generate the appropriate function automatically given the expected rate:

    >>> from marxs.source.source import poisson_process
    >>> star = PointSource(flux=poisson_process(100.)

Energy
^^^^^^
Similarly to the flux, the input for ``energy`` can just be a number, which speficies a monochromatic source:

    >>> emissionline = PointSource(energy=6.7)
    >>> photons = emissionline.generate_photons(5)
    >>> print photons['energy']
    energy
    ------
       6.7
       6.7
       6.7
       6.7
       6.7



          >>> from marxs.optics.source import Source
          >>> def time_dependent_energy(t):
          ...     en = np.ones_like(t)
          ...     en[t > 100.] = 2.
          ...     return en
          >>> mys = Source(energy=f)

          In this example, the photons have the energy 1 keV before 100 and 2. keV after.

.. _sect-source-radec:

Speficy the position for an astrophysical source
------------------------------------------------


.. _sect-source-lab:

Specify the position for a laboratory source
--------------------------------------------
