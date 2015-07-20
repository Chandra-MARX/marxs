.. module:: marxs.optics


Define the design of an instrument / mission
============================================

A simulation is defined by creating objects for all optical elements such as mirrors and gratings and
sorting these optical elements in the order in which a photons encounters them. Marxs works under the
assumption that the order of elements is fixed. A photon does not have to interact with each element
(e.g. it can miss a grating, but still be detected by another focal plane instrument), but under no circumstance will a photon ever go back
to an element it already passed.

There are ways to describe layouts where photons split between different paths, such as in the
design of XMM-Newton where some photons pass the gratings and get refelcted into the RGS detectors,
while others continue on their path without ever encountering a grating until they hit the imaging
detectors. Such complex setups are described in :ref:`complexgeometry`.


Define an optical element for the simulation
--------------------------------------------
All optical elements have to be derived from :obj:`marxs.optics.base.OpticalElement` and they all share a common interface. 

Optical elements are generated with the following keywords:

- Optical elements are placed in 3-d space with the ``pos4d`` keyward, or the ``zoom``, ``orientation`` and ``position`` keywords as explained in :ref:`coordsys`.
- Optical elements can have a name:

  >>> from marxs.optics import FlatDetector
  >>> det1 = FlatDetector(name='back-illuminated CCD No 1')

- Individual optical elements might add more keywords (e.g. the pixel size for a detector or the filename of a parameter file); those are listed below for the individual optical elements.

List of optical elements provided by marxs
------------------------------------------
Note that elements which generate photons (astropysical sources or lab sources) are not listed here. See :ref:`sources`.

.. autoclass:: RectangleAperture

.. autoclass:: Baffle
	       
.. autoclass:: ThinLens

.. autoclass:: MarxMirror

.. autoclass:: FlatDetector

Diffraction gratings
^^^^^^^^^^^^^^^^^^^^
The gratings implmented in marxs solve the diffration equation, but not Maxwell's equations. Thus, they cannot determine the probability for a photons to be diffracted into a particular order. Instead, gratings accept a keyword ``order_selector`` that allows to pass in function (or other callable) that assigngs each diffrated photons to a gratings order. The following ``order_selector`` functions are provided in marxs:

.. autofunction:: uniform_efficiency_factory

.. autofunction:: constant_order_factory

.. autofunction:: EfficiencyFile

These order selector functions can be used with the following gratings:	      
	     
.. autoclass:: FlatGrating

.. autoclass:: CATGrating
	       






.. _complexgeometry:

Complex designs
---------------

.. todo::

   Write about complex designs with parallel path.
