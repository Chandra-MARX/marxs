Define your own optical element
===============================

Marxs comes with a selection of useful optical elements and for many cases these elements are sufficient to define your mission. However, if that is not the case, you may want to write your own code to deal with those pieces that marxs is missing. We are very interested in adding more elements to marxs and we would love to hear from you. Just click the github link in the sidebar and open an issue or pull request to add your code to marxs.

In this section, we explain how to add a new optical element to marxs. All optical elements must be derived directly or indirectly from the optical element base class `~marxs.optics.OpticalElement`. For flat optical elements marxs already provides `~marxs.optics.FlatOpticalElement`, which adds a default geometry (the box shape discussed in :ref:`pos4d`) and a `~marxs.optics.FlatOpticalElement.intersect` method that calculates where a ray hits the active surface of that element.

The documentation of both classes provides details on the methods and attributes that should be overwritten in derived classes.

Base classes for optical elements
---------------------------------

- `marxs.base.MarxsElement`
- `marxs.base.SimulationSequenceElement`
- `marxs.optics.OpticalElement`
- `marxs.optics.FlatOpticalElement`

Example for a derived class
---------------------------

A simpe example for an optical element is the baffle - a rectangular opening in a large matel plate. Ic can be inserted into the beam and all photons that do not pass through the hole will be absorbed. In practice, the metal plate is not infinite, but has an outer bound, too, but all photons passing on the outside are either absorbed by the satellite housing ot the tube of the lab beamline and are lost from the experiment anyway and do not need to be traced any further.

.. literalinclude:: ../marxs/optics/baffle.py

The code above uses the ``intersect`` method that it inherits from `marxs.optics.FlatOpticalElement.intersect` to calculate which photons intersect at all and if so, at which coordinate. By convention, this point is entered into the ``photons['pos']``, although that would not be strictly necessary in this case, since the rays does not bend in a baffle.

Note that the baffle can be rectangular without writing a single line of code here. ``baf = Baffle(zoom=[1,3,2])`` would create a baffle of size 6*4 (the outer boundaries is 3 or 2 mm from the center) because the `marxs.optics.OpticalElement` can interpret all :ref:`pos4d` keywords and the `marxs.optics.FlatOpticalElement.intersect` makes use of this information when it calculates intersection points.

The baffles does not add any non-standard columns to the output, thus ``output_columns`` stays at the default value ``[]``. It also does not need any more geometric information than the box already defined in `marxs.optics.FlatOpticalElement`.

Reference / API
===============

.. automodapi:: marxs.base
