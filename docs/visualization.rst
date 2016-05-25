.. currentmodule:: marxs.visualization

.. _visualization:

Visualize results
=================
A good visualization can be a very powerful tool to understand the results of a simulation and to help to refine the design of an instrument or to see how a specific observation setup can be improved.

Look at the photon event lists
------------------------------
In many cases, the most imporant output of a marxs simulation will be the photon event list with simulated detector (x, y) coordiantes for each photon. Much of the analysis can be done with any plotting package of your choice. 

.. todo::
   example with matplotlib code and image here


A 3 dimensional picture
-----------------------
Specifically when designing an instrument, we might want a 3-D rendering of the mirrors, detectors, and gratings as well as the photon path. We want to see where photons are absorbed and where they are scattered.

Depending on the goal of the visualization (e.g. a beautiful rendering for a presentation, a detailed technical rendering for a publication, or a 3D display that can be zoomed and rotated on the screen) different output formats are required. `marxs.visualization` will eventually support different backends for display and rendering with different display options, but this is a work in progress.

Currently, the following backends are supported (but note that not every element can be displayed in every backend; error messages will indicate this):

- `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`_

Making 3-D plots falls in two parts:

Display rays
^^^^^^^^^^^^
Routines to display rays are collected in `marxs.visualization`, e.g. for display in `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`_ in `marxs.visualization.mayavi`.

Display optical elements
^^^^^^^^^^^^^^^^^^^^^^^^

Elements that make up a marxs model have a ``display`` method, e.g. `marxs.optics.grating.FlatGrating.plot`. This method accepts a ``format`` string to specify the backend, e.g. ``format="mayavi"``. Further keyword arguments and return values depend on the backend and are listed in the following. Each optical element also has some default values to customize its looks in its ``display`` property, which may or may not be used by the individual backend since not every backend supports the same display settings.

mayavi
++++++
:viewer: ``mayavi.core.scene.Scene`` instance

:return value: Object (e.g. a ``mayavi.visual.Box`` instance)

The primary information used for looks is ``display['color']`` which can be any
RGB tuple or, if `matplotlib <http://matplotlib.org>`_ is installed, any valid
`matplotlib.colors.ColorConverter` color. In addtion the plotting routines
attempt to find all calid OpenGL properties by name in the ``display``
dictionary and set those.


Reference/API
=============

.. automodapi:: marxs.visualization
