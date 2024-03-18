# Licensed under GPL version 3 - see LICENSE.rst
import copy
import numpy as np
import pytest

from ...optics import FlatDetector, RectangleAperture
from ...design import RowlandTorus
from ..x3d import plot_rays, plot_object

def compare_trimmed(text1, text2):
    for line1, line2 in zip(text1.split('\n'), text2.split('\n')):
        assert line1.strip() == line2.strip()

def test_skip_plot():
    '''Test that plot is skipped with no warning if explicitly
    requested.'''
    det = FlatDetector(zoom=2, position=[2., 3., 4.])
    # First one does not exist, but second one should lead to skip
    # with no warning
    det.display = copy.copy(det.display)
    det.display['shape'] = 'boxxx; None'
    out = plot_object(det)
    out_expected = '''<Scene/>'''
    # check that nothing was added to the scene (i.e. we get an empty scene)
    compare_trimmed(out.XML(), out_expected)

def test_box():
    '''Any box shaped element has the same representation so it's
    OK to test just one of them.'''
    det = FlatDetector(zoom=2, position=[2., 3., 4.])
    det.display = det.display
    det.display['opacity'] = 0.1
    out = plot_object(det)
    out_expected = '''<Scene>
  <Shape>
    <Appearance>
      <Material diffuseColor='1.0 1.0 0.0' transparency='0.9'/>
    </Appearance>
    <IndexedFaceSet colorPerVertex='false' coordIndex='0 2 3 1 -1 4 6 7 5 -1 0 4 6 2 -1 1 5 7 3 -1 0 4 5 1 -1 2 6 7 3 -1' solid='false'>
      <Coordinate point='0.0 1.0 2.0 0.0 5.0 2.0 0.0 1.0 6.0 0.0 5.0 6.0 2.0 1.0 2.0 2.0 5.0 2.0 2.0 1.0 6.0 2.0 5.0 6.0'/>
    </IndexedFaceSet>
  </Shape>
</Scene>
'''
    compare_trimmed(out.XML(), out_expected)

def test_aperture():
    '''Test a plane_with_hole representation.'''
    det = RectangleAperture(zoom=5)
    det.display = det.display
    det.display['opacity'] = 0.3
    out = plot_object(det)
    out_expected = '''<Scene>
  <Shape>
    <Appearance>
      <Material diffuseColor='0.0 0.75 0.75' transparency='0.7'/>
    </Appearance>
    <IndexedTriangleSet colorPerVertex='false' index='0 4 5 0 1 5 1 5 6 1 2 6 2 6 7 2 3 7 3 7 4 3 0 4' solid='false'>
      <Coordinate point='0.0 15.0 15.0 0.0 -15.0 15.0 0.0 -15.0 -15.0 0.0 15.0 -15.0 0.0 5.0 5.0 0.0 -5.0 5.0 0.0 -5.0 -5.0 0.0 5.0 -5.0'/>
    </IndexedTriangleSet>
  </Shape>
</Scene>
'''
    compare_trimmed(out.XML(), out_expected)


def test_surface():
    '''Test an object that's represented as a surface.'''
    rowland = RowlandTorus(R=1000, r=100)
    rowland.display['coo1'] = np.linspace(1, 2, 4)
    rowland.display['coo2'] = np.linspace(-.2, .2, 3)
    out = plot_object(rowland)
    out_expected = '''<Scene>
  <Shape>
    <Appearance>
      <Material diffuseColor='1.0 0.3 0.3' transparency='0.8'/>
    </Appearance>
    <IndexedFaceSet colorPerVertex='false' coordIndex='0 1 4 3 -1 1 2 5 4 -1 3 4 7 6 -1 4 5 8 7 -1 6 7 10 9 -1 7 8 11 10 -1' solid='false'>
      <Coordinate point='1033.0 84.1 -209.4 1054.0 84.1 0.0 1033.0 84.1 209.4 1003.1 97.2 -203.3 1023.5 97.2 0.0 1003.1 97.2 203.3 970.7 99.5 -196.8 990.4 99.5 0.0 970.7 99.5 196.8 939.3 90.9 -190.4 958.4 90.9 0.0 939.3 90.9 190.4'/>
    </IndexedFaceSet>
  </Shape>
</Scene>
'''
    compare_trimmed(out.XML(), out_expected)


def test_rays():
    '''Just two rays to make sure the format is right.'''
    rays = plot_rays(np.arange(12).reshape(2,2,3))
    out_expected = '''<Scene>
  <Shape>
    <Appearance>
      <Material emissiveColor='0.27 0.0 0.33'/>
    </Appearance>
    <LineSet vertexCount='2 2'>
      <Coordinate point='0 1 2 3 4 5 6 7 8 9 10 11'/>
    </LineSet>
  </Shape>
</Scene>
'''
    compare_trimmed(rays.XML(), out_expected)


def test_zip_file(tmp_path):
    """Check that a zipfile is created and contains the right files."""
    zipfile = pytest.importorskip("zipfile", reason="Test writes zip file")
    rays = plot_rays(np.arange(12).reshape(2, 2, 3))
    rays.write_html_archive(base_name=tmp_path / "figure", format="zip")
    assert (tmp_path / "figure.zip").exists()

    zip = zipfile.ZipFile(tmp_path / "figure.zip")
    assert "figure.html" in zip.namelist()
    assert "x3dom.js" in zip.namelist()
    assert "x3dom.css" in zip.namelist()
