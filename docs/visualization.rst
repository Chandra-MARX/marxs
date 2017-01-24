.. currentmodule:: marxs.visualization

.. _visualization:

*****************
Visualize results
*****************
A good visualization can be a very powerful tool to understand the results of a simulation and to help to refine the design of an instrument or to see how a specific observation setup can be improved.

Look at the photon event lists
==============================
In many cases, the most imporant output of a marxs simulation will be the photon event list with simulated detector (x, y) coordiantes for each photon. Much of the analysis can be done with any plotting package of your choice.

.. todo::
   example with matplotlib code and image here


A 3 dimensional picture
=======================
Specifically when designing an instrument, we might want a 3-D rendering of the mirrors, detectors, and gratings as well as the photon path. We want to see where photons are absorbed and where they are scattered.

Depending on the target audience of the visualization (e.g. a detailed technical rendering for a publication, or a 3D display in a webpage) different output formats are required. `marxs.visualization` supports different backends for display and rendering with different display options. Not every backend supports every option and we welcome code contributions to improve the plotting.

Currently, the following plotting packages are supported: 

- `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`__ is a python package for interactive 3-D displays that uses VTK underneath. Using plotting routines from `marxs.visualization.mayavi` scenes can be displayed right in the same Python session or `jupyter notebook <http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks>`_ where the MARXS simulation is running or be exported as static images (e.g. PNG) or as X3D suitable for inclusion in a web page or `AAS journal paper <http://adsabs.harvard.edu/abs/2016ApJ...818..115V>`_.
- `three.js <https://threejs.org/>`__ is a javascript package to render 3d content in a web-browser using WebGL technology. MARXS does not display this content directly, but outputs code (`marxs.visualization.threejs`) or json formatted data (`marxs.visualization.threejsjson`) that can be included in a web page.

All backends support a function called ``plot_object`` which plots objects in the simulation such as apertures, mirrors, or detectors and a function called ``plot_rays`` which shows the path the photons have taken through the instrument in any specific simulation. In the example below, we use the `~marxs.visualization.mayavi` backend to explain these functions. The arguments for these functions are very similar for all backends, but some differences are unavoidable. For example, `marxs.visualization.mayavi.plot_rays` needs a reference to the scene where the rays will be added, while `marxs.visualization.threejs.plot_rays` requires a file handle where relevant javascript commands can be written - see :ref:`sect-vis-api` for a more details.

Each optical element in MARXS has some default values to customize its looks in its ``display`` property, which may or may not be used by the individual backend since not every backend supports the same display settings.
The most common settings are ``display['color']`` (which can be any RGB tuple or any color valid in `matplotlib <http://matplotlib.org>`_) and ``display['opacity']`` (a nubmer between 0 and 1). 






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
