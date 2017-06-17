1.1 (unreleased)
----------------

New Features
^^^^^^^^^^^^
- Add ability to plot cylinders in Mayavi (for support struts) [#147]

API Changes
^^^^^^^^^^^
- Remove ``marxs.sources.LabPointSource``, which was just a special case of
  `~marxs.sources.LabPointSourceCone`. Instead, set the default values of the
  later so that it reproduces the behaviour of the former. [#144]

- `~marxs.optics.MultiLayerEfficiency` and `~marxs.optics.MultiLayerMirror` now
  expect all parameters as keyword arguments for consistency with the other
  elements in MARXS. [#144]

- ``marxs.visualization.utils.format_saved_positions`` is now a method of
  `~marxs.simulator.KeepCol` with the new name ``format_positions()`` and
  the ``atol`` keyword can be switched off.
  Additionally, `~marxs.simulator.KeepCol` now has a ``__array__`` method.
  This makes the function useful for columns that
  are not positions, but e.g. polarization vectors.
  On the other hand, the ``plot_rays`` functions do not accept
  `~marxs.simulator.KeepCol` objects directly as input any longer.
  [#149, #152]

- According to the docs, a pointing could be initialized with either a 
  `~astropy.coordiantes.SkyCoord` or a tuple that would initialize the
  `~astropy.coordiantes.SkyCoord`. The later option was broken and has 
  been removed entirely. [#151]

Bug fixes
^^^^^^^^^
- Added missing keywords in display dict for some objects and fixed exception
  when plotting things that are not objects. Discovered and fixed as part of
  [#147].

- Polarization after reflection from a mirror used to just parallel transport
  the vector and calculate the probability of the photon based on s and p
  polarization. This needs to be applied to the outgoing polarization vector,
  too. [#148]



Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Improved Documentation [#146]

1.0 (14-Apr-2017)
-----------------
This is the first release intended to use. The Change log will begin starting 
with this release.

0.1 (experimental release)
--------------------------
This release was not intended to be used, but the verisioning scheme in the 
development branch required a tagged commit.
