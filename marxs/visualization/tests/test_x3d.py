# Licensed under GPL version 3 - see LICENSE.rst
import copy
import numpy as np

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
    out = plot_object(rowland)
    out_expected = '''<Scene>
  <Shape>
    <Appearance>
      <Material diffuseColor='1.0 0.3 0.3' transparency='0.8'/>
    </Appearance>
    <IndexedFaceSet colorPerVertex='false' coordIndex='0 1 4 3 -1 1 2 5 4 -1 3 4 7 6 -1 4 5 8 7 -1 6 7 10 9 -1 7 8 11 10 -1' solid='false'>
      <Coordinate point='1033.0198010324336 84.14709848078965 -209.4034805484464 1054.030230586814 84.14709848078965 0.0 1033.0198010324336 84.14709848078965 209.4034805484464 1003.1214261859155 97.19379013633127 -203.34277992165713 1023.523757330299 97.19379013633127 0.0 1003.1214261859155 97.19379013633127 203.34277992165713 970.6850328291146 99.54079577517649 -196.76759747252672 990.4276451985625 99.54079577517649 0.0 970.6850328291146 99.54079577517649 196.76759747252672 939.28141724382 90.92974268256818 -190.40176944213098 958.3853163452858 90.92974268256818 0.0 939.28141724382 90.92974268256818 190.40176944213098'/>
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
      <Material emissiveColor='0.267004 0.004874 0.329415'/>
    </Appearance>
    <LineSet vertexCount='2 2'>
      <Coordinate point='0 1 2 3 4 5 6 7 8 9 10 11'/>
    </LineSet>
  </Shape>
</Scene>
'''
    compare_trimmed(rays.XML(), out_expected)