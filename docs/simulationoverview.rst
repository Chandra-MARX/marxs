==========================
Ray-trace a x-ray mission
==========================

A ray-trace of a new X-ray mission or laboratory experiment is done in several steps:

#. Describe the set-up (e.g. position of optical elements, size of detector pixels).
#. Run the simulation for a larger number of photons for a specific source (e.g. a star, an AGN)
#. Analyse the results.

In this section, we explain how the ``marxs`` package can be used to set-up a ray-trace for a known
geometric configuration, which tools this package offers to design new missions, where the details
of the instrument are still worked out with the help of the ray-trace simulations and how to run
simulations.

If you already have a ``marxs`` file that contains your instrument parameters from the instrument
team or provided in the example directory of this package, you can skip the first two chapters of
this section and read about running simulations immidiately.

.. toctree::
   :maxdepth: 2

   optics
   design
   run

