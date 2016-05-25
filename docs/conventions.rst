*********************
Coordinates and units
*********************

.. _coordsys:
.. _pos4d:

Coordinate system
=================
Marxs employs a cartesian coordinate system. All optical elements can be freely placed at any position
and any angle in this space, but we recommend the following conventions for simplicity (examples and
predefined missions in this pckage follow those conventions as far as possible):

- The optical axis of the telescope coincides with the x-axis. A source on the optical axis is
  at :math:`x=+\infty` and photons travel towards the origin.
- The origin of the coordiante system is at the telescope focus.
- The y-axis coincides with the dispersion direction of gratings.

Marxs uses `homogeneous coordinates <https://en.wikipedia.org/wiki/Homogeneous_coordinates>`_, which
describe position and direction in a 4 dimensional coordinate space, for example
:math:`[3, 0, 0, 1]` describes a point at x=3, y=0 and z=0; :math:`[3, 0, 0, 0]` describes the
vector from the origin to that point. Homogeneous coordinates have one important advantage compared
with a normal 3-d description of euklidean space: In homogenous coordinates, rotation, zoom, and
translations together can be described by a :math:`[4, 4]` matrix and several of these operations can
be chained simply by multiplying the matrices.

Every optical element in marxs (and some other objects, too) has such a :math:`[4, 4]` matrix
associated with it in an attribute called ``element.pos4d``.

All optical elements have some default location and position. Typically, their active surface (e.g.
the surface of a mirror or detector) is in the y-z plane. The center is at the origin of the
coordiante system and the default size in each dimenstion is 1, measured from the center.
Thus, e.g. the default definition for a `marxs.optics.detector.FlatDetector`, makes a detector surface with
the following corners (in 3-d x,y,z coordinates): [0, -1, -1], [0, -1, 1], [0, 1, 1] and [0, 1, -1].
This detector has the thickness 1, but that only matters for display purposes, since the ray-trace
calculates the intersection with the "active surface" in the y-z plane independent of the
thickeness in x-direction. There are exceptions to these defaults, those are noted in the description
of the individual optical elements.
See the following sketch, where the "active surface" is shown in red:

.. plot:: plots/sketch_coords.py optical_element_coords
   :scale: 50
   :align: center
   :include-source: False
   :alt: Layout of the default size and position of an optical elements. The "active surface" (e.g. the surface of a mirror) is shown in red.


In order to place elements in the experiment, the optical element needs to be
scaled (zoomed), rotatated and translated to the the new position.
There are two ways to specify that:

- Pass a :math:`[4,4]` matrix to the optical element:

      >>> import numpy as np
      >>> from marxs.optics import Baffle
      >>> pos4d = np.array([[1., 0., 0., 5.],
      ...                   [0., 2., 0., 0.],
      ...                   [0., 0., 2., 0.],
      ...                   [0., 0., 0., 1.]])
      >>> baf1 = Baffle(pos4d=pos4d)

  This sides of this baffle have length 4 (from -2 to 2). The center is on the x-axis, but 5 units
  upwards from the origin.

- Use the keywords ``orientation``, ``zoom``, and ``position``. The code will construct the ``pos4d``
  matrix from those keywords, by first scaling with ``zoom``, then applying the rotation matrix
  ``orentation`` and lastly, moving the center of the optical element to ``position``.
  The same baffle as above can also be written as:

      >>> baf2 = Baffle(zoom=[1., 2., 2.], position=[5., 0., 0.])
      >>> np.all(baf2.pos4d == baf1.pos4d)
      True


  It is an error to specify ``pos4d`` and any of the ``orientation``, ``zoom``, and ``position``
  keywords at the same time.

.. note::

   The `transforms3d <https://matthew-brett.github.io/transforms3d/index.html>`_ package provides
   several functions to calculate 3-d and 4-d matrices from Euler angles, axis and angle,
   quaternions and several other representations. This package is installed together with marxs.
  
Physical units
==============
Most elements of ``marxs`` are scale free, and no specific unit is attached to by the code. However,
some elements require an explicit scale (e.g. the grating dispersion depends on the grating constant
and the photon energy), in this case the following conventions apply:

- Length: base unit is mm.
- Energy: base unit is keV.
- Angles: Always expressed in radian.

To avoid confusion, we recommend using mm as unit of lenght throughout marxs.
