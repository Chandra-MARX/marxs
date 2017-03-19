.. _sect-results:

************
MARXS Output
************

MARXS output is a photon list in the form of an `astropy.table.Table`. Depending on the optical elements in the simulation, this table can have quite a few columns. Some of the columns contain the input, such as the true Ra, Dec and energy of the photon; others track the photons through the simulations, e.g. a column could list which grating or which detector the photon hits. Detectors typically add columns that contain the X, Y value of the photon when it is detected.

.. _sect-results-output:

Output columns
==============
The most important columns during the simultion are the homogeneous space position contained in column "pos" and velociy "dir". Those columns will be part of the output. ``pos`` is always set the position where the photon interacted last with an optical element. ``dir`` is the direction it is traveling in.
Photons from astrophysical sources have the follwoing properties: Time, energy, true sky coordinates (RA and dec) and polarization angle (measures North through east). There is also a probability column (see below), which is initialized to one::

  >>> from astropy.coordinates import SkyCoord
  >>> from marxs import optics, source
  >>> coords = SkyCoord(345., -35., unit='deg')
  >>> src = source.PointSource(coords=coords, energy=0.25)
  >>> point = source.FixedPointing (coords=coords)
  >>> photons = src.generate_photons(5)
  >>> photons.colnames
  ['energy', 'probability', 'polangle', 'time', 'ra', 'dec']
  >>> photons['probability']
  <Column name='probability' dtype='float64' length=5>
  1.0
  1.0
  1.0
  1.0
  1.0

The pointing takes the (RA, dec) coordiantes on the sky and transforms it to an (x,y,z) vector in spacecraft coordinates (actually, we use homogeneous coordiantes so all vectors are really (x,y,z,w) vectors, see :ref:`coordsys`); similarly, it takes the polarization angle and turns it into a vector::
  
  >>> photons = point(photons)
  >>> photons.colnames[6:]
  ['dir', 'polarization']

Other optical elements can, but do not have to add columns with more information. This is defined when the instrument is set up, e.g. a CCD detector typically adds columns named "detpix_x" and "detpix_y" which contain the pixel coordinates of the detected event. If the column name is not self-explenatory, check the documentation for a specific instrument which should explain what each column in the output table means, so there is no need to delve into the details here. For instrument builders, see :ref:`sect-optics-output`.
 

Tracking the photon probability
===============================
For every photon, MARXS tracks the probability that this specific photon makes it to the detector. So, if a photon, for example, passes through an absorbing filter that absorbes 50% of all photons of this energy, each photon will have the value in its column "probability" multiplied by 0.5. This way, all photons that are send out by the source, will end up in the final photon list, although some of them will have a probability of 0. Another approach commonly used in Monte-Carlo ray-trace codes is to draw a random number, and if it is below 0.5, the photons is transmitted through the filter, above 0.5 it is discoarded. However, that reduces the number of photons in the final distribution list and thus increases the random scatter in the results. Simulations need to be run with a larger number of photons to see faint features.

Save results to disk
====================
The photon list is just an `astropy.table.Table` which can be saved to many formats out-of-the-box, most importantly to fits files::

  >>> photons.write('mysim_result.fits')

Sometimes, an instrument specification might provide a specific writing method to add more metadata and header keywords to the table before it is written.

Visualize results
=================
The most important output of a MARXS simulation will be the photon event list with simulated detector (x, y) coordinates for each photon. This can be plotted in the same python session (see :ref:`sect-runexample-look` for an example) or in any plotting package of your choice that reads the photon event list from a file.
