*************************
Running MARXS at a glance
*************************

A MARXS simulation can be customized in many different ways to adapt to different instruments and science questions. Therefore, some classes support quite complex input. Before discussing all options in detail, we want to present a complete MARX run with typical parameters here.

Our question is the following: `Miller et al. (2003) <http://adsabs.harvard.edu/abs/2003ApJ...585L..37M>`_ observed two ultra-luminous X-ray source in NGC 1313 with XMM-Newton. If we take a grating spectrum of one of these sources with Chandra, will be get any trace from the second trace on the corner of the detector?

We will simulate two point sources in Chandra, one on the optical axis and one located a few arcmin off-axis. MARXS includes a simplified description of the Chandra X-ray observatory in the `marxs.missions` module.

Creating two sources
====================

First, we need to set up the two point sources with a spectrum and a total flux.
We have to set the positions of our sources as an `astropy.coordinates.SkyCoord`::

   >>> from astropy.coordinates import SkyCoord
   >>> ngc1313_X1 = SkyCoord("3h18m19.99s -66d29m10.97s")
   >>> ngc1313_X2 = SkyCoord("3h18m22.34s -66d36m03.68s")

In many cases, the spectrum of the sources will come from a table that is generated from a simulation or from a table written by a program such as XSPEC or `Sherpa <http://cxc.harvard.edu/sherpa/>`_. MARXS sources accept a wide range of input formats (see :ref:`sect-source-fluxenpol`) including plain numpy arrays for an energy grid and a flux density grid in photons/s/cm**2/bin.
Our two sources here have a simple power-low spectrum. We ignore extinction for the moment which would significantly reduce the flux we observe in the soft band. 

   >>> import numpy as np
   >>> energies = np.arange(.3, 8., .01)
   >>> spectrum1 = 6.3e-4 * energies**(-1.9)
   >>> spectrum2 = 3.9e-4 * energies**(-2.4)

Marxs sources have separate parameters for the ``energy`` (shape of the spectrum) and the ``flux`` (normalization of the spectrum and temporal distribution). In many cases we want to apply the same spectrum with different normalizations to different sources, so it is useful to separate the shape of the spectrum and the total flux. We simulate a source that is constant in time, so the photon arrival times will just be Poisson distributed. In this case, the input spectrum is already properly normalized, so we just sum it all up to to get the total flux. Note that MARXS interprets the energy array as the *upper* energy bound for each bin. The lower energy bound is gives as the upper energy bound from the previous bin. This leaves the *lower* bound of the first bin undefined and thus we do not include it in the sum.

   >>> from marxs.source import poisson_process
   >>> flux1 = (spectrum1[1:] * np.diff(energies)).sum()
   >>> flux2 = (spectrum2[1:] * np.diff(energies)).sum()
   >>> flux1 = poisson_process(flux1)
   >>> flux2 = poisson_process(flux2)

The last thing that the source needs to know is how large the geometric opening of the telescope is, so that the right number of photons are generated. So, we create an instance of the Chandra mirror with the parameter file included in the MARXS distribution::

   >>> import os
   >>> from marxs import optics
   >>> parfile = os.path.join(os.path.dirname(optics.__file__), 'hrma.par')
   >>> HRMA = optics.MarxMirror(parfile) # doctest: +IGNORE_OUTPUT
   Reading data files from classic MARX C code ...

Now, can can finally create our sources:
 
   >>> from marxs.source import PointSource
   >>> src1 = PointSource(coords=ngc1313_X1, energy={'energy': energies, 'flux': spectrum1},
   ...                    flux=flux1, geomarea=HRMA.area)
   >>> src2 = PointSource(coords=ngc1313_X2, energy={'energy': energies, 'flux': spectrum2},
   ...                    flux=flux2, geomarea=HRMA.area)

See :ref:`sources` for more details.
   
Set up observation and instrument configuration
===============================================
We need to specify the pointing direction and the roll angle (if different from 0)::

   >>> from marxs.missions import chandra
   >>> pointing = chandra.LissajousDither(coords=ngc1313_X1)

and also set up the instrument components. The mirror was already initialized above, and the rest is easy because it is already included in MARXS (but keep in mind that the Chandra model used here includes only the geometry. It leaves out many important effects which require aceess to CALDB data, such as ACIS contamination, order probabilities for gratings etc. Use `classic marx`_ to simulate Chandra data for real.)
::

   >>> hetg = chandra.HETG()
   >>> acis = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])  # doctest: +IGNORE_OUTPUT
   Some warnings that ACIS 1024 * pixel size does not match chip size exactly ...

That was easy!

If you are simulating an instrument that is not part pf the MARXS package itself you can either specify everything yourself (see :ref:`sect-optics` for more details) or use files provided, e.g. by the instrument team. In that case, you have probably also received more detailed instructions on how to use these files.

Run the simulation
==================
In MARXS, a source generates a photon table and all following elements process this photon table by changing values in a column or adding new columns. Thus, we can stop and inspect the photons after every step (e.g. to look at their position right after they exit the mirror) or just execute all steps in a row::

   >>> p1 = src1.generate_photons(1e4)  # 10 ks exposure time
   >>> p1 = pointing(p1)
   >>> p1 = HRMA(p1)
   >>> p1 = hetg(p1)
   >>> p1 = acis(p1)

We can do the same thing for the photons from the second source::

   >>> p2 = src2.generate_photons(1e4)  # 10 ks exposure time
   >>> p2 = pointing(p2)
   >>> p2 = HRMA(p2)
   >>> p2 = hetg(p2)
   >>> p2 = acis(p2)

New we can either merge the two photon lists to a single list similar to what we would get from a real Chandra observation::

  >>> from astropy import table
  >>> p = table.vstack([p1, p2])

or make use of the fact that in the simulation (unlike in real life) we know exactly which photon came from which source and keep them separate.

.. _sect-runexample-look:

Look at simulation results
==========================

In MARXS, if you want to know the number of photons that are expected to be detected, you select the photons in the output list that hit the detector and then add up all the probabilities::

  >>> ind = p['CCD_ID'] > 0
  >>> 'Expected number of photons: {}'.format(p['probability'][ind].sum()) # doctest:+ELLIPSIS
  'Expected number of photons: ...'
  
If, instead, you are looking for a list of detected photons which has the same noise levels, you need to draw a subset of events from the photon list::

  >>> pobs = p[p['probability'] < np.random.uniform(len(p))]

For more details on the MARXS output see :ref:`sect-results`.

We can now look at the distribution of photons on the detector::

  >>> from matplotlib import pyplot as plt
  >>> line = plt.plot(p1['tdetx'], p1['tdety'], '.')

Notice that the plot is not scaled equally in the x and y axis and thus the zeroth order looks a little streched. Also, we plot in detector coordinates. Because of the dither, the photons look a little smeared out (A plot zooming in on the zeroth order would show the dither pattern.)
  
.. plot:: pyplots/runexample.py

Only photons from the first source (blue) are visible, no photon from the second source is seen because its distance to source 1 is larger than the Chandra field-of-view. (We could have seen that without running a simulation.)

For more details on visualization see :ref:`visualization`.
   
