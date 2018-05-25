1.2 (unreleased)
----------------

New Features
^^^^^^^^^^^^
- Unit milimeter on ``"pos"`` and ``"dir"`` columns of photonlist. [#169]

- Add method `marxs.simulator.Parallel.move_center` to change the ``pos4d``
  value of a `marxs.simulator.Parallel` and adjust position of elements at
  the same time. [#169]

- Refactor `marxs.analysis.analysis.find_best_detector_position` to allow
  for more general objective functions. [#171]

- Update plotting for `marxs.optics.aperture.CircleAperture` to give more 
  flexibility in plotting the inner part of a ring-like aperture. This is
  needed for models or stacked, rind-like apertures. [#180]

API Changes
^^^^^^^^^^^

Bug fixes
^^^^^^^^^

- `marxs.analysis.gratings.resolvingpower_per_order` has been updated to ignore
  photons with probability 0. [#162]

- An index mix-up in `marxs.simulator.ParallelCalculated.calculate_elempos` introduced
  unintended zoom and shear in the elements. [#164]

- [#159] left behind an undefined ``filterfunc``. This is fixed and a
  regression test added. [#165]

- `marxs.analysis.analysis.find_best_detector_position` will now change the
  detector position always along an axis perpendicular to the detector plane.
  [#171]

- `marxs.optics.scatter.RadialMirrorScatter` now works with
  ``inplanescatter=0`` which is useful for parameters studies. [#174]

- Plot only "half-box" for elements such as mirrors where the optical
  interaction occurs on a surface. [#178]

  
  
Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1 (15-Jul-2017)
-----------------

New Features
^^^^^^^^^^^^
- Add ability to plot cylinders in Mayavi (for support struts) [#147]

- Add shape `"None"` (as a string) to avoid plotting an object [#157]

API Changes
^^^^^^^^^^^
- Remove ``marxs.source.LabPointSource``, which was just a special case of
  `~marxs.source.LabPointSourceCone`. Instead, set the default values of the
  later so that it reproduces the behaviour of the former. [#144]

- `~marxs.optics.multiLayerMirror.MultiLayerEfficiency` and
  `~marxs.optics.multiLayerMirror.MultiLayerMirror` now
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

- Remove parameter ``filterfunc`` from `~marxs.analysis.gratings.resolvingpower_from_photonlist` and `~marxs.analysis.analysis.detected_fraction`.
  Instead, the photon list can be filtered before calling these functions
  just as easily. [#159]

Bug fixes
^^^^^^^^^
- Added missing keywords in display dict for some objects and fixed exception
  when plotting things that are not objects. Discovered and fixed as part of
  [#147].

- Polarization after reflection from a mirror used to just parallel transport
  the vector and calculate the probability of the photon based on s and p
  polarization. This needs to be applied to the outgoing polarization vector,
  too. [#148]

- Plotting of Rowland Torus failed in Mayavi due to typo. [#154]


Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Improve Documentation [#146]

- Docs: Add polarization example [#153]

- Docs: Add example to calculate flux from normalized spectrum [#160]

- Add experiemental data for comparison to the polarization example [#158]

1.0 (14-Apr-2017)
-----------------
This is the first release intended to use. The Change log will begin starting 
with this release.

0.1 (experimental release)
--------------------------
This release was not intended to be used, but the verisioning scheme in the 
development branch required a tagged commit.
