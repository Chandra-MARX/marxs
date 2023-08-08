**********************************
Define your own simulation element
**********************************

Marxs comes with a selection of useful optical elements and for many cases these elements are sufficient to define your mission. However, if that is not the case, you may want to write your own code to deal with those pieces that marxs is missing. We are very interested in adding more elements to marxs and we would love to hear from you. Just open an issue or pull request to add your code to marxs at `our github repro <https://github.com/Chandra-MARX/marxs>`_.

In the simplest case, an element of the simulation can be any function or class with a ``__call__`` method that accepts a photon list as input and returns a (usually modified) photon list as output. As a simple example we show an element that represent an absorbing plate that
just absorbes half of all photons coming though (i.e. reduces the probability of each photon to be detected by 1/2).
First, let us generate some photons coming from some astrophysical source::

  >>> from astropy.coordinates import SkyCoord
  >>> from marxs import source, optics
  >>> import astropy.units as u
  >>> mysource = source.PointSource(coords=SkyCoord(181., -23., unit='deg'))
  >>> photons = mysource.generate_photons(10 * u.s)
  >>> mypointing = source.FixedPointing(coords=SkyCoord('12h01m23s -22d59m12s'))
  >>> photons = mypointing(photons)

Our example is very simple. We can do this with a simple function::

  >>> def myabs1(photons):
  ...     photons['probability'] *= 0.5
  ...     return photons
  >>> photons = myabs1(photons)

A more complicated way to achieve the same thing would be::

  >>> from marxs.optics import OpticalElement
  >>> class AbsorbAllPhotonsElement(OpticalElement):
  ...     def __call__(self, photons):
  ...         photons['probability'] *= 0.5
  ...         return photons
  >>> myabs2 = AbsorbAllPhotonsElement()
  >>> photons = myabs2(photons)

In this simple case, a function might be sufficient but most optical elements
are more complex. For example, they have a finite size, they cannot be
vectorized and must loop over all photons one-by-one, or they should offer the user some diagnostic to analyze the result of the ray-trace such as an index column that lists which CCD was hit or which mirror shell reflected a particular photon. For those cases, marxs provides base classes that do much of the work:

- `marxs.base.MarxsElement`
- `marxs.base.SimulationSequenceElement`
- `marxs.optics.OpticalElement`
- `marxs.optics.FlatOpticalElement`

Check the description of the classes in this list for details. Optical elements in marxs are derived from `~marxs.optics.OpticalElement` or `~marxs.optics.FlatOpticalElement`. The `~marxs.optics.OpticalElement` requires the user to implement an ``intersect`` method (see `~marxs.optics.OpticalElement.intersect` for a describtion of input and output parameters). Then either of the following will work:

  - Override `~marxs.optics.OpticalElement.process_photons` with the method signature.
  - For a vectorized approach, just add a method called ``specific_process_photons``, that has the same input signature as `~marxs.optics.OpticalElement.process_photons` and outputs a dictionary, where the keys are columns names in the photon list and the values are `numpy.ndarray` objects that hold the result for the intersecting photons.
  - Instead, if the derived element has a method called ``process_photon`` (see `~marxs.optics.OpticalElement.process_photon` for the required signature) then it will loop through the list of intersecting photons one-by-one and record the output values of that function in the photons table.

A `~marxs.optics.FlatOpticalElement` already has an `~marxs.optics.FlatOpticalElement.intersect` method implementing the default box geometry discussed in :ref:`pos4d`, so the best us is to implement a ``specific_process_photons`` method.

For performance, we recommend to use vecorized implementation and change the values of the photon table in place (as opposed to making copies of parts of the table and joining them back together) whereever possible.

Example for a derived class
===========================

A simple example for an optical element is the baffle - a rectangular opening in a large metal plate. It can be inserted into the beam and all photons that do not pass through the hole will be absorbed. In practice, the metal plate is not infinite, but has an outer bound, too, but all photons passing on the outside are either absorbed by the satellite housing ot the tube of the lab beamline and are lost from the experiment anyway and do not need to be traced any further.

.. literalinclude:: ../marxs/optics/baffles.py

The code above uses the `~marxs.optics.FlatOpticalElement.intersect` method it inherits to calculate which photons intersect at all and if so, at which coordinate. This point is entered into the ``photons['pos']``.

Note that the baffle can be rectangular without writing a single line of code here. ``baf = Baffle(zoom=[1,3,2])`` would create a baffle of size 6*4 (the outer boundaries are 3 or 2 mm from the center) because the `~marxs.optics.OpticalElement` can interpret all :ref:`pos4d` keywords and the `marxs.optics.FlatOpticalElement.intersect` makes use of this information when it calculates intersection points.

The baffle does not add any new descriptive columns to the output, thus the value `~marxs.base.SimulationSequenceElement.id_col` defined in a base class is not overwritten. However, the baffle does have a class dictionary `~marxs.optics.baffles.Baffle.display` that contains default parameters for plotting (see :ref:`visualization`).

Non-physical elements
=====================

Above, we discussed elements that are physically present, such as mirrors or detectors. The simulations also allows elements that are not a piece of hardware, for example, a function that just saves a copy of the photon list to disk or prints some diagnostic information would also work::

  >>> def backupcopy(photons):
  ...     photons.write('backup.fits', overwrite=True)
  ...     return photons
  >>> photons = backupcopy(photons)

Why have we chosen to return a copy of photons here, although no value in the photon list was changed? This allows us to sneak in the new function into a list of optical elements::

  >>> from marxs import simulator
  >>> myinstrument = simulator.Sequence(elements=[myabs1, backupcopy, myabs2])

Reference / API
===============

.. automodapi:: marxs.base
