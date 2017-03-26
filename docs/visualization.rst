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

First, we need to set up mirrors, gratings and detectors. In this example, we use the Chandra toy model included in MARXS::


   >>> from marxs import optics
   >>> from marxs.missions import chandra
   >>> marxm = chandra.HRMA(os.path.join(os.path.dirname(optics.__file__), 'hrma.par'))
   >>> hetg = chandra.HETG()
   >>> aciss = chandra.ACIS(chips=[4,5,6,7,8,9], aimpoint=chandra.AIMPOINTS['ACIS-S'])


Usually, ``display`` is a class dictionary, so any change on any one object will affect all elements of that class (here: We change the color of one ACIS chip, and later they will all have the same color):

   >>> aciss.elements[0].display['color'] = 'orange'
   
To change just one particular element (here the ACIS-S3 chip, which is the forth chip in the ACIS-S array) we copy the class dictionary to that element and set the color:

    >>> import copy
    >>> aciss3 = aciss[3]
    >>> aciss3.display = copy.deepcopy(aciss3.display)
    >>> aciss3.display['color'] = 'r'

.. _sect-vis-example:

Using MARXS to visualize sub-aperturing
=======================================
In this example, we use MARXS to explain how sub-aperturing works. Continuing the code from above, we define an instrument setup, run a simulation, and plot results both in 2d and in 3d. 

We already have the mirror (called ``marxm`` above), the gratings (called ``hetg``) and the detector array (``aciss``). For pure convenience we combine those three things in `~marxs.simulator.Sequence` and also define a `marxs.simulator.KeepCol` object, which we pass into our `~marxs.simulator.Sequence`. When we call the `~marxs.simulator.Sequence`, the `~marx.simulator.KeepCol` object will make a copy of a column (here the "pos" column) and store that.

The ``hetg`` has several hundred gratings. We want to know how photons that go through a particular set of gratings are  distributed throughcompared to the total point-spread function. Therefore, we assign a color to every sector of gratings and in our plots, we color gratings in this sector and photons that passed through it accordingly.

.. plot:: plots/vis_subaperturing.py
   :include-source:

On the plot, we see that photons from each sector (e.g. the sector that colors photons red) form a long strip on the detector. Only adding up photons from all sectors gives us a rounf PSF. Subaperturing is the idea to disperse only photons from one sector, such that the PSF in cross-dospersion direction is large, but the dispersion in cross-dispersion direction is much smaller. Thus, the spectral resolution of the dispersed spectrum will be higher than for a spectrograph that uses the full PSF. (A detailed discussion is beyond the scope of this manual. Here, we are mostly concerend with showing how MARXS can be used to run the simulations and how to turn those into useful output.)
      
Later, we just connect all the stored positions for a photon by a line and we will see the ray path taken through the instrument:



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
