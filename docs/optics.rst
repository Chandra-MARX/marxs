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

- Optical elements are placed in 3-d space with the ``pos4d`` keyward, or the ``zoom``, ``orientation`` and ``position`` keywords as explained in :ref:`coordsys`:

  >>> from marxs.optics import FlatDetector
  >>> det1 = FlatDetector(position=[5., 1., 0.], zoom=[1, 50.2, 50.2], pixsize=0.1)  
  
- Optical elements can have a name:

  >>> from marxs.optics import FlatDetector
  >>> det1 = FlatDetector(name='back-illuminated CCD No 1')

- Individual optical elements might add more keywords (e.g. the pixel size for a detector or the filename of a parameter file); those are listed below for the individual optical elements, e.g.:

  >>> from marxs.optics import FlatDetector
  >>> det1 = FlatDetector(pixsize=0.04)



List of optical elements provided by marxs
------------------------------------------
Note that elements which generate photons (astropysical sources or lab sources) are not listed here. See :ref:`sources`.
For convenience all of the commonly elements listed in the following can be imported directly from ``marxs.optics`` such as:

    >>> from marxs.optics import Baffle

This short from is recommended over the long form:

    >>> from marxs.optics.baffle import Baffle

.. autosummary::
   :toctree: API

   aperture.RectangleAperture
   baffle.Baffle
   mirror.ThinLens
   marx.MarxMirror
   detector.FlatDetector


Diffraction gratings
^^^^^^^^^^^^^^^^^^^^
The gratings implemented in marxs solve the diffration equation, but not Maxwell's equations. Thus, they cannot determine the probability for a photon to be diffracted into a particular order. Instead, gratings accept a keyword ``order_selector`` that expects a function (or other callable) that assigngs each diffrated photon to a gratings order. For example, the following code makes a grating where the photons are distributed with equal probability in all orders from -2 to 2:

   >>> from marxs.optics import FlatGrating, uniform_efficiency_factory
   >>> select_ord = uniform_efficiency_factory(2)
   >>> mygrating = FlatGrating(d=0.002, order_selector=select_ord)

The grating module contains different classes for gratings and also different pre-defined ``order_selector`` function. Use the code in those functions as a template to define your own ``order_selector``.
   
.. autosummary::
   :toctree: API

   grating.FlatGrating
   grating.CATGrating
   grating.constant_order_factory
   grating.uniform_efficiency_factory
   grating.EfficiencyFile
	       






.. _complexgeometry:

Complex designs
---------------

.. todo::

   Write about complex designs with parallel path.
