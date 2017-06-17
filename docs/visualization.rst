.. currentmodule:: marxs.visualization

.. _visualization:

*****************
Visualize results
*****************
A good visualization can be a very powerful tool to understand the results of a simulation and to help to refine the design of an instrument or to see how a specific observation setup can be improved.



A 3 dimensional picture
=======================
Specifically when designing an instrument, we might want a 3-D rendering of the mirrors, detectors, and gratings as well as the photon path. We want to see where photons are absorbed and where they are scattered.

Depending on the target audience of the visualization (e.g. a detailed technical rendering for a publication, or a 3D display in a webpage) different output formats are required. `marxs.visualization` supports different backends for display and rendering with different display options. Not every backend supports every option and we welcome code contributions to improve the plotting.

Currently, the following plotting packages are supported: 

- `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`__ is a python package for interactive 3-D displays that uses VTK underneath. Using plotting routines from `marxs.visualization.mayavi` scenes can be displayed right in the same Python session or `jupyter notebook <http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks>`_ where the MARXS simulation is running or be exported as static images (e.g. PNG) or as X3D suitable for inclusion in a web page or `AAS journal paper <http://adsabs.harvard.edu/abs/2016ApJ...818..115V>`_.
- `three.js <https://threejs.org/>`__ is a javascript package to render 3d content in a web-browser using WebGL technology. MARXS does not display this content directly, but outputs code (`marxs.visualization.threejs`) or json formatted data (`marxs.visualization.threejsjson`) that can be included in a web page.

All backends support a function called ``plot_object`` which plots objects in the simulation such as apertures, mirrors, or detectors and a function called ``plot_rays`` which shows the path the photons have taken through the instrument in any specific simulation. In the example below, we use the `~marxs.visualization.mayavi` backend to explain these functions. The arguments for these functions are very similar for all backends, but some differences are unavoidable. For example, `marxs.visualization.mayavi.plot_rays` needs a reference to the scene where the rays will be added, while `marxs.visualization.threejs.plot_rays` requires a file handle where relevant javascript commands can be written - see :ref:`sect-vis-api` for a more details.

Each optical element in MARXS has some default values to customize its looks in its ``display`` property, which may or may not be used by the individual backend since not every backend supports the same display settings.
The most common settings are ``display['color']`` (which can be any RGB tuple or any color valid in `matplotlib <http://matplotlib.org>`_) and ``display['opacity']`` (a number between 0 and 1).

First, we need to set up mirrors, gratings and detectors (here gratings on a Rowland torus)::

  >>> from marxs import optics, design
  >>> from marxs.design import rowland
  >>> rowland = design.rowland.RowlandTorus(500, 500)
  >>> aper = optics.CircleAperture(position=[1200, 0, 0], zoom=[1, 100, 100], r_inner=20)
  >>> mirr = optics.FlatStack(position=[1100, 0, 0], zoom=[20, 100, 100],
  ...                         elements=[optics.PerfectLens, optics.RadialMirrorScatter],
  ...                         keywords = [{'focallength': 1100},
  ...                                     {'inplanescatter': 1e-3, 'perpplanescatter': 1e-4}])
  >>> gas = design.rowland.GratingArrayStructure(rowland=rowland, d_element=25, x_range=[800, 1000],
  ...                                            radius=[20, 100], elem_class=optics.FlatGrating,
  ...                                            elem_args={'d': 1e-5, 'zoom': [1, 10, 10],
  ...                                                       'order_selector': optics.OrderSelector([-1, 0, 1])})
  >>> det_kwargs = {'d_element': 10.5, 'elem_class': optics.FlatDetector,
  ...               'elem_args': {'zoom': [1, 5, 5], 'pixsize': 0.01}}
  >>> det = design.rowland.RowlandCircleArray(rowland, theta=[2.3, 3.9], **det_kwargs)


Usually, ``display`` is a class dictionary, so any change on any one object will affect all elements of that class::

  >>> optics.FlatDetector.display['color'] = 'orange'
   
To change just one particular element we copy the class dictionary to that element and set the color. In this example, we calculate the angle that each grating has from the y-axis. We split that in three regions and color the gratings accordingly::

  >>> import copy
  >>> import numpy as np
  >>> for e in gas.elements:
  ...     e.display = copy.deepcopy(e.display)
  ...     ang = np.arctan(e.pos4d[1,3] / e.pos4d[2, 3]) % (np.pi)
  ...     e.display['color'] = 'rgb'[int(ang / np.pi * 3)]


.. _sect-vis-example:

Using MARXS to visualize sub-aperturing
=======================================
In this example, we use MARXS to explain how sub-aperturing works. Continuing the code from above, we define an instrument setup, run a simulation, and plot results both in 2d and in 3d. This is a fully functional ray-trace simulation; the only unrealsitic point is the grating constant of the diffraction gratings which is about an order of magnitude smaller than current gratings.


2d plot
-------

The ``hetg`` has several hundred gratings. We want to know how photons that go through a particular set of gratings are  distributed throughcompared to the total point-spread function. Therefore, we assign a color to every sector of gratings and in our plots, we color gratings in this sector and photons that passed through it accordingly.

.. plot:: pyplots/vis_subaperturing.py
   :include-source:

On the plot, we see that photons from each sector (e.g. the sector that colors photons red) form a long strip on the detector. Only adding up photons from all sectors gives us a rounf PSF. Subaperturing is the idea to disperse only photons from one sector, such that the PSF in cross-dospersion direction is large, but the dispersion in cross-dispersion direction is much smaller. Thus, the spectral resolution of the dispersed spectrum will be higher than for a spectrograph that uses the full PSF. (A detailed discussion is beyond the scope of this manual. Here, we are mostly concerend with showing how MARXS can be used to run the simulations and how to turn those into useful output.)

3d output
---------

Next, we want to show how the instrument that we use above looks in the 3d.
Very similar to the previous example, we combine mirror, detector, and grating into `~marxs.simulator.Sequence`. Unlike the simulation for the 2d case, we also define a `marxs.simulator.KeepCol` object, which we pass into our `~marxs.simulator.Sequence`. When we call the `~marxs.simulator.Sequence`, the `~marxs.simulator.KeepCol` object will make a copy of a column (here the "pos" column) and store that.
Another changes compared to the 2d plotting is that we generate a lot fewer photons because the green lines indicating the photon paths can obscurre the graitings.
  
.. doctest-requires:: mayavi

  >>> import numpy as np
  >>> from mayavi import mlab
  >>> from astropy.coordinates import SkyCoord
  >>> from marxs import source, simulator
  >>> from marxs.visualization.mayavi import plot_object, plot_rays
  >>> # object to save intermediate photons positions after every step of the simulaion
  >>> pos = simulator.KeepCol('pos')
  >>> instrum = simulator.Sequence(elements=[aper, mirr, gas, det], postprocess_steps=[pos])
  >>> star = source.PointSource(coords=SkyCoord(30., 30., unit='deg'),
  ...                           energy=2., flux=1.)
  >>> pointing = source.FixedPointing(coords=SkyCoord(30., 30., unit='deg'))
  >>> photons = star.generate_photons(1000)
  >>> photons = pointing(photons)
  >>> photons = instrum(photons)
  >>> ind = (photons['probability'] > 0) & (photons['facet'] >=0)
  >>> posdat = pos.format_positions()[ind, :, :]
  >>> fig = mlab.figure()
  >>> obj = plot_object(instrum, viewer=fig)
  >>> rays = plot_rays(posdat, scalar=photons['energy'][ind])
   
.. raw:: html

  <div class="figure" align="center">
  <x3d width='500px' height='400px'> 
  <scene>
  <inline url="_static/subaperturing.x3d"> </inline> 
  </scene> 
  </x3d>
  <p class="caption" style="clear:both;"><span class="caption-text">
  3D view of the instrument set up. Green lines are photon paths. As set above, all detectors are orange and the gratings are red, green, and blue, depending on their position. The mirror and the aperture are shown with their default representation (white box and transparent green plate with the aperture hole). Use your mouse to rotate, pan and zoom. <a href="https://www.x3dom.org/documentation/interaction/">(Detailed instructions for camera navigation)</a> </span></p>
  </div>
        
.. _sect-vis-api:

Reference/API
=============

.. automodapi:: marxs.visualization
.. automodapi:: marxs.visualization.mayavi
   :skip: format_doc
.. automodapi:: marxs.visualization.threejs
   :skip: format_doc
.. automodapi:: marxs.visualization.threejsjson
   :skip: format_doc
.. automodapi:: marxs.visualization.utils
