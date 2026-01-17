2.1 (unreleased)
----------------

New Features
^^^^^^^^^^^^
- Make X3D output compatible with X_ITE. This includes cleanup up of a few pieces
  that X3DOM is not strict about. [#253]
- Add a new class `marxs.optics.CircularBaffle` that implements a baffle with a circular
  hole. [#258]

API Changes
^^^^^^^^^^^

Bug fixes
^^^^^^^^^

- Fix test failures with numpy 2.0 [#254]

Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.0 (02-May-2024)
-----------------

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

- Added a new class ``CaptureResAeff_CCDgap`` that selects photons differently for
  calculation of the effective area and resolving power to account for chip gaps. [#235]

- Add direct X3D visualization backend to avoid the need for Mayavi. [#238]
- Add several missions under development that were previously developed in separate
  repros. As it turn out, keeping the code separate lead to significant parallel
  work that can be simplified by keeping all of them in marxs itself. However, those
  mission are less fleshed out then marxs itself, less documented and less tested. [#237]
- Add ``CircularMeshGrid`` as a new way to place gratings on the Rowlandtorus. Adapted from
  previous Lynx code. [#237]
- Add various helper functions that previously lived in separate repros. [#237]
- Add methods to fit the LSF to generate a Chandra-style LSFPARM file. [#237]
- Include code for several missions (Athena, Arcus, Lynx, AXIS) into marxs.
  This code had been previously developed separately, since none of it is really ready for
  public consumption (e.g. the Lynx concept has been rejected, Arcus is under continuous redefinition).
  However, given how few people except me currently use MARXS, "practicality beats purity"
  and I'm now including this into marxs. Note that the mission code generally is not documented
  in the docs and not tested as well.
  It's simply either not interesting enough for general users (but needed to
  replicate some of my SPIE proceedings so Moritz Guenther wants it publicly available)
  or evolving too fast to write good docs and tests (e.g. the Arcus effective area
  changes every time the coating or layout changes, tests that check Aeff are
  thus more checks on the CALDB than on the code). [#240]
- A number of updates that were developed for those individual missions and are
  now included into marxs proper, e.g.: A class to make certain plots that display
  the performance loss for misalignments and utilities to run tolerancing etc. [#240]
- Add new functions to `marxs.design.rowland` to construct Rowland tori in a
  double tilted Rowland torus design. [#245]
- Add more options to ``PerfectLens`` to account for non-perfect reflectivity and
  the option to have the optical axis offset from the geometrical center of the element. [#248]
- Add method to create archive files of X3D files. [#251]

API Changes
^^^^^^^^^^^
- For all classes that place elements on the Rowland torus, the API has been
  standardized. That means changes to almost all of them. In particular:

  - ``rowland.RectangularGrid`` replaces ``RowlandTorusArray`` and ``LinearCCDArray``.
    Inputs are specified the y and z ranges now instead of Rowland circle
    :math:`$\phi` coordinates that are hard to visualize since it's measure from the
    center of the Rowland circle and the origin of :math:`$\phi=0` depends on the
    rotation of the torus. ``rowland.RectangularGrid`` can be used for 1D or 2D
    arrangements and thus ``d_element`` must always be two numbers for dimension in
    both directions.
  - All classes now use a minimizer instead of a brackets root finder. So, instead
    of a bracketing interval, they now need a ``guess_distance`` where the minimization
    starts.

- ``load_table2d`` has been replaced by ``marxs.utils.tablerows_to_2d``, which performs
  the same reformatting of a table, but does not read it in. [#237]
- ``marxs.missions.mitsnl.InterpolateEfficiencyTable`` now takes a table, instead of a
  filename as parameter. [#237]
- `effectivearea_from_photonlist` now requires a unit for the input of the effective area. [#237]
- ``CaptureResAeff_CCDgap`` and ``CaptureResAeff`` are moved to
  ``marxs.analysis.gratings``. [#237]
- ``plot_wiggle`` and ``load_and_plot`` are now summarized in the class to make it
  easier to derive from them. [#237]

- Rename files that had a class of the same name, because this confuses the Sphinx autosummary
  extension. The following files have been renamed: `marxs.missions.chandra.chandra` to
  `marxs.missions.chandra.chandra_global`, `marxs.source.source` to `marxs.source.basesources`,
  and `marxs.optics.baffle` to `marxs.optics.baffles`. Users that follow the recommended practice
  of importing from the higher level such as `from marxs.optics import Baffle` should not notice
  any difference. [#424]

- The unit of photon energy is now set to keV. [#252]

- ``CCDRedistNormal`` now inherits from an optical element and takes the sigma for redistribution
  as a keyword argument. [#252]

Bug fixes
^^^^^^^^^

- `marxs.missions.mitsnl.catsupportbars` no longer throws an exception if
  no photon hit a facet before. [#221]

- To avoid probabilities <0 or >1, the default interpolation for MIT grating
  efficiency tables is now linear. [#229]

- Fig bug in ``design_tilted_torus``. The equations implemented previously
  implicitly assumed 2 * alpha = beta. [#250]

Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Previously, MARXS only contained modules for currently operating missions.
  However, in practice, it's mostly used to simulate proposed missions. Thus,
  we changed the policy to include simulation code for proposed missions here.
  That code may not be of general use, but it simplifies the workflow for
  developers considerably and serve example on how to build your own
  instrument from the building blocks provided by the main marxs modules. [#237]

- Updates of CI and simplify _astropy_init and conftest for new Python and
  astropy versions. [#236]


1.2 (06-Nov-2020)
-----------------

New Features
^^^^^^^^^^^^
- Unit millimeter on ``"pos"`` and ``"dir"`` columns of photonlist. [#169]

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
  quantities. Any table-like input must not be an astropy `QTable` and flux and
  polarization values are *densities* not per-bin. [#218]

- The geometry of a class is now defined as a separate object from the new
  `marxs.math.geometry` module. Before, the geometry was backed into the
  optical components itself which led to significant code duplication. The new
  scheme allows making e.g. cylindrical and flat gratings with the same
  gratings objects. However, not all geometries work with all objects and there
  is currently no way to check automatically - the user has to use caution or
  check the code. [#182]

- Geometry objects provide item access like this
  ``detector.geometry['center']`` instead of a function interface with
  ``(...)``. [#182]

- Remove ``utils.MergeIdentical`` merge strategy since it is no longer used
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

- Add experimental data for comparison to the polarization example [#158]

1.0 (14-Apr-2017)
-----------------
This is the first release intended to use. The Change log will begin starting
with this release.

0.1 (experimental release)
--------------------------
This release was not intended to be used, but the ver
sioning scheme in the
development branch required a tagged commit.
