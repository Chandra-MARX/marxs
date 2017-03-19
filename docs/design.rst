***********************************
Tools to help the instrument design
***********************************

In the previous sections, we either used pre-defined instrument configurations or
placed all components "by hand", giving their absolute coordinates in mm. But how do we
know the best places for gratings, detectors etc.?
In some cases, analytical work can tell us where e.g. diffraction gratings and detectors need
to go to achieve the best possible resolving power. The module `marxs.design` implements those
formulas. It currently focusses on a spectrometer in a Rowland Torus design (see e.g.
`Paerels 1999 <http://adsabs.harvard.edu/abs/1999LNP...520..347P>`_ and
`Paerels 2010 <http://adsabs.harvard.edu/abs/2010SSRv..157...15P>`_ for a derivation), but we
welcome distributions for other configurations.

In the following example, we define a Rowlandtorus based spectrometer. In this particular
example, we select the radius of the Rowland Circle to be very small (1 m) and also make the
grating constant of the gratings very small. This way we get a much larger diffraction angle
than usual and we end up with a model that does real ray-tracing (albeit with some unrealistic
input values) and delivers very clear images to demonstante the operating principles
for teaching. We begin by importing the relevant packages and then we define the Rowland torus.
In this case, both radii of the torus are set to 500 mm, so the Rowland circle in the x,y plane
touches the focal point at (0,0) and has a diameter of 1 m, so grating on the optical axis would
be 1 m away from the focal point::

  >>> import numpy as np
  >>> from marxs import optics
  >>> from marxs import simulator
  >>> from marxs.design import rowland
  >>> my_torus = rowland.RowlandTorus(500, 500)

Our instrument approximates an imaging optic by using a circular aperture that illuminates
a `marxs.optics.PerfectLens` with some additional scatter
(an `marxs.optics.RadialMirrorScatter`). The reflection on the mirror and the scatter always
happen at the same position, so those two functions are summarized in a single element
(a `marxs.optics.FlatStack`) as explained in :ref:`complexgeometry`::
  
  >>> aperture = optics.CircleAperture(position=[1200, 0, 0], zoom=[1, 100, 100])
  >>> mirror = optics.FlatStack(position=[1100, 0, 0], zoom=[20, 100, 100],
  ...                            elements=[optics.PerfectLens, optics.RadialMirrorScatter],
  ...                            keywords = [{'focallength': 1100},
  ...                                        {'inplanescatter': 1e-3, 'perpplanescatter': 1e-4}])

Next, behind the mirror, we place a `~marxs.design.rowland.GratingArrayStructure`, which is simply
a collection of gratings. The class of the grating is defined in the parameter ``elem_class`` and
any keyword arguments needed to initialize every individual grating are passed in ``elem_args``. In
this case, the is the grating constand ``d``, the size of the gratings set by the ``zoom`` and the
``order_selector`` that gives the probabilities to decide into which diffraction order a particular
photons is diffracted. Note that ``position`` and ``orientation`` are missing from this list,
because they will be calcualted by the `~marxs.design.rowland.GratingArrayStructure`.

The remaining parameters set how the `~marxs.design.rowland.GratingArrayStructure` places the
gratings. Looking along the optical axis, it will select y and z positions for the gratings
such that they lay in a ring, where the inner and outer radius is given by the ``radius`` parameter.
The x-position of each grating is adjusted to make sure it touches the Rowland torus. To do so,
we need to pass `~marxs.design.rowland.GratingArrayStructure`
an instance of the Rowland Torus, the distance ``d_element`` between gratings (measured
center to center - note that this is typically larger than the size of each element to account for
the frame the gratings are mounted on), and the range along the optical axis (the x-axis of the torus)
in which the gratings are expected to fall (the equation of the torus is solved numerically and the
solver needs a bounded regions with a unique solution to work)::

  >>> gas = rowland.GratingArrayStructure(rowland=my_torus, d_element=25, x_range=[800, 1000],
  ...                                     radius=[20, 100], elem_class=optics.FlatGrating,
  ...                                     elem_args={'d': 1e-5, 'zoom':[1, 10, 10],
  ...                                                'order_selector': optics.OrderSelector([-1, 0, 1])})

Last, we place detectors on the inner part of the Rowland circle. This is very similar to the
`~marxs.design.rowland.GratingArrayStructure`, except that the CCD detectors are not placed in a
2D pattern, but just in a 1D line along the Rowland Circle. Since we do not want to fill the entire
:math:`2 \pi` of the circle, we set a limit on ``theta``. All other parameters work the same way as above::
  
  >>> det = rowland.RowlandCircleArray(my_torus, theta=[np.pi - 0.8, np.pi + 0.8],
  ...                                  d_element=10.5, elem_class=optics.FlatDetector,
  ...                                  elem_args={'zoom': [1, 5, 5], 'pixsize': 0.01})

Last, all the elements above as combined into a single element::
  
  >>> demo = simulator.Sequence(elements=[aperture, mirror, gas, det])


Reference / API
===============
.. automodapi:: marxs.design
