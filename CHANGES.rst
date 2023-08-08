1.3 (unreleased)
----------------

New Features
^^^^^^^^^^^^

- `marxs.simulator.Parallel` and `marxs.simulator.Sequence` now have a
  ``first_of_class_top_level`` method to help find certain subelements. [#222]

- Added `marxs.design.tolerancing.run_tolerances_for_energies2` is similar to
  `marxs.design.tolerancing.run_tolerances_for_energies`, but with a
  simplified interface. [#222]

- Output columns from optical elements can now be `None` (they will be ignored)
  or specify the datatype and other Column meta information. [#226]

- To avoid probabilities <0 or >1, values assigned to this column are now
  tested and a ``ValueError`` may be raised. [#230]

- Added a new class `CaptureResAeff_CCDgap` that selects photons differently for
  calculation of the effective area and resolving power to account for chip gaps. [#235]

- Add direct X3D visualization backend to avoid the need for Mayavi. [#238]

API Changes
^^^^^^^^^^^
- ``load_table2d`` has been replaced by ```marxs.utils.tablerows_to_2d``, which perform
  the same reformatting of a table, but does not read it in. [#237]
- ``marxs.missions.mitsnl.InterpolateEfficiencyTable`` now takes a table, instead of a
  filename as parameter. [#237]

- Rename files that had a class of the same name, because this confuses the Sphinx autosummary
  extension. The following files have been renamed: `marxs.missions.chandra.chandra` to
  `marxs.missions.chandra.chandra_global`, `marxs.source.source` to `marxs.source.basesources`,
  and `marxs.optics.baffle` to `marxs.optics.baffles`. Users that follow the recommended practice
  of importing from the higher level such as `from marxs.optics import Baffle` should not notice
  any difference. [#424]

Bug fixes
^^^^^^^^^

- `marxs.missions.mitsnl.catsupportbars` no longer throws an exception if
  no photon hit a facet before. [#221]

- To avoid probabilities <0 or >1, the default interpolation for MIT grating
  efficiency tables is now linear. [#229]

Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updates of CI and simplify _astropy_init and conftest for new Python and
  astropy verisons. [#236]


1.2 (06-Nov-2020)
-----------------

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

- Add new module `marxs.design.tolerancing` to estimate alignment tolerances.
  [#186]

- Add class `marxs.optics.RandomGaussianScatter`. [#192]

- The column names for blaze and order written by
  `marxs.optics.FlatGrating` can now be configured. [#195]

- Add a number of helper functions (incl. plotting) to
  `marxs.design.tolerancing`. [#199]

- Add a module to simulate CAT gratings manufactures in MIT's SNL. [#202]

- Make MIT SNL grating specification more flexible. [#210]


API Changes
^^^^^^^^^^^

- Sources now require input for flux, energy, and polarization as astropy
  quantities. Any table-like input must not be a astropy `QTable` and flux and
  polarization values are *densities* not per-bin. [#218]

- The geometry of a class is now defined as a separate object from the new
  `marxs.math.geometry` module. Before, the geometry was backed into the
  optical components itself which led to significant code duplication. The new
  scheme allows to make e.g. cylindrical and flat gratings with the same
  gratings objects. However, not all geometries work with all objects and there
  is currently no way to check automatically - the user has to use caution or
  check the code. [#182]

- Geometry objects provide item access like this
  ``detector.geometry['center']`` instead of a function interface with
  ``(...)``. [#182]

- Remove ``utils.MergeIdentical`` merge strategy since it is not longer used
  after #189. [#191]

- `~astropy.units.Quantity` is used in more places, and thus
  the base classes need to accept quantities as output from
  "specific_process_photons". This change requires some care and thus
  is done gradually only where needed at this point. [#202]

- `~marxs.optics.RandomGaussianScatter` can be initialized without setting
  scatter, provided that its already defined a class level variable (e.g.
  for subclasses). [#202]

- `~marxs.optics.RandomGaussianScatter` and `~marxs.optics.Scatter` now expect
  angular quantities instead of plain floats. [#216]


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

- With `marxs.optics.MultiAperture` photons would always be sorted by aperture
  number. To fix this, apertures now behave more like other optical elements
  and use ``process_photons``. [#189]

- `marxs.optics.FlatStack` now inherits from `marxs.simulator.BaseContainer`.
  [#196]

- Fix bug in distributions generated by `marxs.source.SphericalDiskSource`. [#202]

- Objects now fail to generate when zoom is set to 0. [#217]

Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Examples are now written for numpy >= 1.14 (which changed some printing
  formats). [#182]

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
  `~astropy.coordinates.SkyCoord` or a tuple that would initialize the
  `~astropy.coordinates.SkyCoord`. The later option was broken and has
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
