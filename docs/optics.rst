.. _sect-optics:

*******************************************
Optical elements that make up an instrument
*******************************************

A simulation is defined by creating objects for all optical elements such as mirrors and gratings and
sorting these optical elements in the order in which a photon encounters them. Marxs works under the
assumption that the order of elements is fixed. A photon does not have to interact with each element
(e.g. it can miss a grating, but still be detected by another focal plane instrument), but under no circumstance will a photon ever go back
to an element it already passed.

There are ways to describe layouts where photons split between different paths, such as in the
design of XMM-Newton where some photons pass the gratings and get reflected into the RGS detectors,
while others continue on their path without ever encountering a grating until they hit the imaging
detectors. Such complex setups are described in :ref:`complexgeometry`.


.. _args for optical elements:

Define an optical element for the simulation
============================================
All optical elements have to be derived from :obj:`marxs.optics.base.OpticalElement` and they all share a common interface. 

Optical elements are generated with the following keywords:

- Optical elements are placed in 3-d space with either the ``pos4d`` keyword, or the ``zoom``, ``orientation`` and ``position`` keywords as explained in :ref:`coordsys`:

  >>> from marxs.optics import FlatDetector
  >>> det1 = FlatDetector(position=[5., 1., 0.], zoom=[1, 50.2, 50.2], pixsize=0.1)  
  
- Optical elements can have a name:

  >>> from marxs.optics import FlatDetector
  >>> det1 = FlatDetector(name='back-illuminated CCD No 1')

- Individual optical elements might add more keywords (e.g. the pixel size for a detector or the filename of a parameter file); those are listed in the documentation for individual optical elements.
  Here is an example for a detector:

  >>> from marxs.optics import FlatDetector
  >>> det1 = FlatDetector(pixsize=0.04)


.. _sect-optics-output:

Output columns for the photon list
==================================
Each MARXS run starts with a source and a pointing. The source generates the physical properties of each photon (energy, polarization) and places it in the universe (RA, dec, time). The pointing transforms position and polarization angle into vectors in spacecraft coordiantes (these vectors are called "dir" and "polarization") - see :ref:`sect-results-output`.

Next, we need to assign an initial position in spacecraft coordinates to each photon. This is done by an aperture. One common approach in ray-tracing is to subdivide the entrace aperture into a regular grid and trace one ray per grid cell; alternatively, a large number of photons can be randomly distributed on the aperture. MARXS takes this second approach because (i) the result does not only depend on the position in the entrance aperture, but on a number of other parmeters (energy, polarization, time for time variable sources or pointings) and (ii) the photon list produced by the simulation closely resembles the photons lists from real observations.
The column "pos" in the table holds the photon position as assigned on the aperture in homogeneous coordinates.

After passing through the aperture, the photon list will have at least the following columns:

pos
  Position where the photon last interacted with an optical element in homogeneous coordinates.
dir
  Direction of the phootn ray in homogeneous coordiantes.
polarization
  Polarization vector in homogeneous coordinates (can be complex for cicularly polarized light).
energy
  in keV
probability
  Probability that the photon is not absorbed anywhere until this point
  
Optical elements may add any number of diagnostic columns in their code. 

The values of these columns can change with every interaction. All other columns are just passed along. They carry information that helps interpret the results of a ray-trace, e.g. the (RA, Dec) sky coordinate where the photons was originally emitted.

Most optical elements in MARXS support at least two ways to add columns:

Which optical element did a photon pass?
  Instruments often have several identical CCDs or several gratings. If an element has a property ``id_col``
  set to a string and ``id_num`` set to a number, then all photons passing it will have ``id_num`` set in the
  column ``id_col``.
  Photons that do not pass through any element of this collection have ``photons[id_col]==-1``.
  If a photon passes through multiple elements with the same ``id_col`` the ``id_num`` recorded is the last
  element passed.

Where did a photon hit an element?
  Flat optical elements (those inheriting from `marxs.optics.base.FlatOpticalElement`) have an attribute
  called ``loc_coos_name``. Setting this to a list of two strings, will cause the element to add two columns with those names that hold the local coordiantes (normalized to the range -1 to 1 for both coordinates). If the column names already exisit, values will be overwritten. The default are the generic names "x" and "y".

The following example demonstrates these two ways of adding columns with diangostic information where exactly the photon passed which element. First, we generate a bunch of photons all parallel to the x-axis.

  >>> import numpy as np
  >>> from marxs.utils import generate_test_photons
  >>> photons = generate_test_photons(5)
  >>> photons['pos'].data[:, 1] = np.arange(5)

Then, we add two CCD detectors::

  >>> det1 = FlatDetector(pixsize=0.1)
  >>> det1.loc_coos_name
  ['det_x', 'det_y']
  >>> det1.id_col = 'CCD_ID'
  >>> det1.id_num = 1
  >>> det2 = FlatDetector(pixsize=0.2, position=[0, 2.5, 0.])
  >>> det2.id_col = 'CCD_ID'
  >>> det2.id_num = 2

Last, we process the photons through both detectors::
  
  >>> photons = det1(photons)
  >>> photons = det2(photons)
  >>> photons  # doctest: +IGNORE_OUTPUT
  <Table length=5>
  polarization [4]  energy   pos [4]     dir [4]   probability  det_x   det_y  detpix_y detpix_x  CCD_ID
      float64      float64   float64     float64     float64   float64 float64 float64  float64  float64
  ---------------- ------- ----------- ----------- ----------- ------- ------- -------- -------- -------
        0.0 .. 0.0     1.0 0.0 .. -1.0 -1.0 .. 0.0         1.0     0.0     0.0      9.5      9.5     1.0
        0.0 .. 0.0     1.0 0.0 .. -1.0 -1.0 .. 0.0         1.0     1.0     0.0      9.5     19.5     1.0
        0.0 .. 0.0     1.0 0.0 .. -1.0 -1.0 .. 0.0         1.0    -0.5     0.0      4.5      2.0     2.0
        0.0 .. 0.0     1.0 0.0 .. -1.0 -1.0 .. 0.0         1.0     0.5     0.0      4.5      7.0     2.0
        0.0 .. 0.0     1.0  1.0 .. 1.0 -1.0 .. 0.0         1.0     nan     nan      nan      nan    -1.0 

In this example, the two detectors are set up by hand. MARXS offers the class `marxs.simulator.Paralell` to set up elements with parallel functions, such as several CCD detectors next to each other.
	
List of optical elements provided by marxs
==========================================
Note that elements which generate photons (astropysical sources or lab sources) are not listed here. See :ref:`sources`.
For convenience the following elements can be imported directly from `marxs.optics` such as:


  >>> from marxs.optics import Baffle

This short from is recommended over the long form:

  >>> from marxs.optics.baffle import Baffle

.. _sect-apertures:
    
Entrance apertures
------------------

For astrophysical sources, we assume that the rays are parallel when they reach the experiment. The direction of the photons is given by the location of the source on the sky and the pointing model, but we still need to select which of the parallel rays we select for the simulation. This is done by an optical element that we call an "aperture" in Marxs. In the case of the `~marxs.optics.MarxMirror` this fuctionality is already included in the code that describes the mirror. For designs that do not use the `~marxs.optics.MarxMirror` the following entrance apertures are included in Marxs:

- `marxs.optics.MultiAperture`
- `marxs.optics.RectangleAperture`
- `marxs.optics.CircleAperture`
- `marxs.optics.MultiAperture`



General optical elements
------------------------
   
-   `marxs.optics.Baffle`
-   `marxs.optics.PerfectLens`
-   `marxs.optics.ThinLens`
-   `marxs.optics.MarxMirror`
-   `marxs.optics.RadialMirrorScatter`
-   `marxs.optics.RandomGaussianScatter`
-   `marxs.optics.GlobalEnergyFilter`
-   `marxs.optics.EnergyFilter`
-   `marxs.optics.FlatDetector`
-   `marxs.optics.CircularDetector`
-   `marxs.optics.FlatBrewsterMirror`



Diffraction gratings
--------------------
The gratings implemented in marxs solve the diffration equation, but not Maxwell's equations. Thus, they cannot determine the probability for a photon to be diffracted into a particular order. Instead, gratings accept a keyword ``order_selector`` that expects a function (or other callable) that assigngs each diffrated photon to a gratings order. For example, the following code makes a grating where the photons are distributed with equal probability in all orders from -2 to 2:

   >>> from marxs.optics import FlatGrating, OrderSelector
   >>> select_ord = OrderSelector([-2, -1, 0, 1, 2])
   >>> mygrating = FlatGrating(d=0.002, order_selector=select_ord)

The grating module contains different classes for gratings and also an `~marxs.optics.OrderSelector`.
   
- `marxs.optics.FlatGrating`
- `marxs.optics.CATGrating`
- `marxs.optics.OrderSelector`
- `marxs.optics.EfficiencyFile`
	       
.. _complexgeometry:

Complex designs
===============

Most optical elements only change rays that intersect them (`~marxs.optics.Baffle` is an exception - it set the probability of all photons that do **not** intersect the central hole to 0.). Thus, any complex experiement can just be constructed as a list of optical elements, even if many photons only interact with some of those elements.

Marxs offers classes to make handling complext designs more comfortable:
`~marxs.simulator.Sequence`, `~marxs.simulator.Parallel`, `~marxs.simulator.ParallelCalculated`, 
and `~marxs.optics.FlatStack`:

The class `~marxs.simulator.Sequence` can be used to group several optical elements. There are two use cases:

- Group several optical elements that are passed by each photon in sequence.
- Group several different parallel elements, e.g. a CCD detector and a proportional counter that are both mounted in the focal plane.

In contrast, `~marxs.simulator.Parallel` is meant for designs with identical
elements, e.g. a camera consisting of four CCD chips; it required a list of
positions for each of these CCDs as an input.
`~marxs.simulator.ParallelCalculated` is a class for objects where the
positions of individual elements are not input manually, but are calculated by
some algorithm.
  
`~marxs.optics.FlatStack` is a special case of the `~marxs.simulator.Sequence` where several flat optical elements are passed by the photons in sequence and all elements are so close to each other, that this can be treated as a single interaction. An example is contamination on a CCD detector, which can be modeled as a Sequence of an `~marxs.optics.EnergyFilter` and a `~marxs.optics.FlatDetector`.


Reference / API
===============
.. automodapi:: marxs.optics

.. automodapi:: marxs.simulator
