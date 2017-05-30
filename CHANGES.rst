1.1 (unreleased)
----------------

New Features
^^^^^^^^^^^^

API Changes
^^^^^^^^^^^
- Remove ``marxs.sources.LabPointSource``, which was just a special case of
  `~marxs.sources.LabPointSourceCone`. Instead, set the default values of the later
  so that it reproduces the behaviour of the former. [#144]

- `~marxs.optics.MultiLayerEfficiency` and `~marxs.optics.MultiLayerMirror` now expect
  all parameters as keyword arguments for consistency with the other elements in MARXS.
  [#144]

Bug fixes
^^^^^^^^^

Other changes and additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0 (14-Apr-2017)
-----------------
This is the first release intended to use. The Change log will begin starting with this release.

0.1 (experimental release)
--------------------------
This release was not intended to be used, but the verisioning scheme in the development branch required a tagged commit.
