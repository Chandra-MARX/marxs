*************************************
Contributing to the MARXS development
*************************************

The design of MARXS is optimized with the following principles and ideas in mind:

Flexibility
  In the design and proposal phase of a mission many different science goals will be
  simulated for different satellite geometries and different instrument configurations. MARXS shall
  make it easy to simulate arbitrary combinations of optical elements in very few lines of code.

Generality
  MARXS aims to provide classes that cover all the common elements in X-ray instruments -
  e.g. Wolter Type I mirrors, baffles, gratings, multi-layer mirrors, detectors. The classes can
  be extended for special use cases not covered by MARXS modules.

Simplicity
  MARXS solves the ray-trace problem in a simple prescription, e.g. it solves the grating equation
  (simple, general physics), but requires an input that selects the grating
  order for each photon. (Solving Maxwell's equations to calculate these probabilities is a messy
  business that requires a lot of material properties and detailed design input.)

Developer time is more valuable than CPU time.
  MARXS shall be fast enough to run on a desktop PC for
  a resonable number of photons (say a million photons can be ray-traded in a few minutes), but
  the flexibility of the code is more important than speed optimizations.

Order of optical elements is known.
  MARXS assumes that the order in which photons hit (or miss)
  optical elements is known, e.g. entrance aperture to mirror to grating to detector.

  
We welcome contributions to MARXS through `github <https://github.com/Chandra-MARX/marxs/>`_ by opening issues for bugs, questions, or feature requests, or, even better, through pull requests.

Interaction with `classic MARX`_
================================
You might want to tell git to ignore the local path
that you put into ``setup.cfg`` to avoid committing and pushing that to the
repro accidentially::

  git update-index --assume-unchanged setup.cfg

(However, note that this will make some manual fixing necessary if the upstream
``setup.cfg`` changes, e.g. because we decide to add a new option. See 
``git help update-index`` for more explanation.)
