import numpy as np

# tests for e2h and h2e


def e2h(e, w):
    '''Convert Euclidean coordinates to homogeneous coordinates

    Parameters
    ----------
    e : np.array
        Input Euclidean coordinates. This can be multidimensional, but the last
        dimension must be of size 3.
    w : float
        ``0`` for directions (points at infinity)
        ``1`` for positions (points in Euclidean space)

    Returns
    -------
    h : np.array
        Homogeneous coordinates. Same shape as ``e`` except that the last
        dimension is now has 4 elements.
    '''
    if not ((w == 0) or (w == 1)):
        raise ValueError('w must be 0 or 1.')
    shape = list(e.shape)
    shape[-1] += 1
    h = np.empty(shape)
    h[..., :3] = e
    h[..., 3] = w
    return h


def h2e(h):
    '''Convert homogeneous coordinates to Euclidean coordinates

    This functions works for points at infinity and for points in real
    euclidean space, but it expects that each time it is called ``h`` contains
    only the one or the other but not a mixture of both.

    Parameters
    ----------
    h : np.array
        Input homogeneous coordinates. This can be multidimensional, but the
        last dimension must be of size 4.

    Returns
    -------
    e : np.array
        Euclidean coordinates. Same shape as ``e`` except that the last
        last dimension is now has 3 elements.
    '''
    if np.all(h[..., 3] != 0):
        return (h[..., :3].T / h[..., 3].T).T
    elif np.all(h[..., 3] == 0):
        return h[..., :3]
    else:
        raise ValueError('Input array must be either all euklidean points or all points at infinity.')


def distance_point_point(h_p1, h_p2):
    if max(h_p1.ndim, h_p2.ndim) == 1:
        axis = 0
    elif max(h_p1.ndim, h_p2.ndim) == 2:
        axis = 1
    else:
        raise ValueError('This function expects 1d or 2d input.')
    return np.linalg.norm(h2e(h_p1) - h2e(h_p2), axis=axis)


def angle_line_plane(p_line, h_plane):
    '''Compute angle between normal to plane and line

    Parameters
    ----------
    p_line : array-like
        Line(s)
    h_plane : array-like
        Plane
    '''
    l = p_line[:3]
    p = h_plane[:3]
    return np.arccos(np.dot(l, p) / (np.linalg.norm(l) * np.linalg.norm(p)))


# Plucker stuff. Can go in separate file at some point...
# Find better naming convention
#  @ L = {U:V}, with 3-tuples U and V, with U.V = 0, and with U non-null.

def e_pointpoint2line(e_p1, e_p2):
    '''p1, p2 are distinct point on line, direction is p1 -> p2'''
    p_line = np.empty(e_p1.shape[:-1] + (6,) )
    p_line[..., :3] = e_p2 - e_p1
    p_line[..., 3:] = np.cross(e_p2, e_p1)
    return p_line


def h_point_h_point2line(h_p1, h_p2):
    '''Do I actually need this, or can I just require to use h2e?'''
    p_line = np.empty(h_p1.shape[:-1] + (6,) )
    p_line[..., :3] = h_p1[..., 3] * h_p2[..., :3] - h_p2[..., 3] * h_p1[..., :3]
    p_line[..., 3:] = np.cross(h_p2[..., :3], h_p1[..., :3])
    return p_line


def dir_point2line(e_dir, e_pos):
    p_line = np.empty(e_dir.shape[:-1] + (6,) )
    p_line[..., :3] = e_dir
    p_line[..., 3:] = np.cross(e_dir, e_pos)
    return p_line


def intersect_line_plane(p_line, h_plane):
    '''
    Parameters
    ----------
    p_line : np.array (last dimension has 6 elements)
        Pluecker coordinates of a line or an array of lines
    h_plane : np.array with ``shape=(4, )``
        Homogeneous coordinates of a plane.

    Returns
    -------
    h_point : np.array
        Homogeneous coordinates of the intersection points of line and plane.
        Has the same shape as ``p_line``, but the last dimension has only 4
        elements.
    '''
    h_point = np.empty(p_line.shape[:-1] + (4,)  )
    h_point[..., :3] = np.cross(p_line[..., 3:], h_plane[:3]) - p_line[..., :3] * h_plane[3]
    h_point[..., 3] = np.dot(p_line[..., :3], h_plane[:3])
    return h_point


def point_dir2plane(h_point, h_dir):
    h_plane = np.empty(4)
    h_plane[:3] = h_dir[:3]
    h_plane[3] = - np.dot(h_point, h_dir) / h_point[3]
    return h_plane

#   @ L = {U:UxQ}, for U the direction of L and Q a point on L.
#   @ L = {qP-pQ:PxQ}, for (P:p) and (Q:q) distinct homogeneous points on L.
#   @ L = {ExF:fE-eF}, for [E:e] and [F:f] distinct planes containing L.

#   @ {U1:V1} =? s{U2:V2} tests if L1 = {U1:V1} equals L2 = {U2:V2}.
#   @ s > 0 if L1 and L2 have same orientation.

#   @ (V.V)/(U.U) is the minimum squared distance of L from the origin.
#   @ (VxU:U.U) is the point of L closest to the origin.
#   @ [UxV:V.V] is the plane through L perpendicular to its origin plane, for
#         non-null V.

#   @ (VxN-Un:U.N) is the point where L intersects plane [N:n] not parallel to L.
#   @ [UxP-Vw:V.P] is the plane containing L and point (P:w) not on L.
#   @ [UxN:V.N] is the plane containing L and direction N not parallel to L.

#   Let N, N1, N2 be unit vectors along the coordinate axes, with U.N non-zero.

#   @ (VxN:U.N) is a point on L if N is not perpendicular to U.
#   @ U and this point both satisfy a plane equation [E:e] if the plane
#         contains L.
#   @ Represent L as U and this point to transform by non-perspective
#         homogeneous matrix.
#   @ Represent L as two points to transform by perspective homogeneous matrix.

#   @ [UxN1:V.N1] and [UxN2:V.N2] are distinct planes containing L.
#   @ P satisfies both these plane equations if L contains P.

#   @ Pnt(t) = (VxU+tU:U.U) parameterizes points on L.
#   @ Pln(t) = (1-t^2)[UxN1:V.N1]+2t[UxN2:V.N2] parameterizes planes through L.

#   @ U1.V2 + U2.V1 =? 0 tests if L1 = {U1:V1} and L2 = {U2:V2} are coplanar
#         (intersect).
#   @ Sum positive if right-handed screw takes one into the other; negative
#         if left-handed.
#   @ U1xU2 =? 0 tests if lines are parallel.
#   Let N be a unit vector along a coordinate axis, with (U1xU2).N non-zero.
#   @ ((V1.N)U2-(V2.N)U1-(V1.U2)N:(U1xU2).N) is the point of intersection, if
#         any.

#   @ [U1xU2:V1.U2] is the common plane for non-parallel lines.
#   Let N, N1, N2 be unit vectors along the coordinate axes, with U1.N non-zero.
#   @ [(U1.N)V2-(U2.N)V1:(V1xV2).N] is the common plane for parallel distinct
#         lines.
#   @ [U1xN1:V1.N1] is the common plane for equal lines through origin.

# Here are two related tricks, a lagniappe, as they say in New Orleans.

#   Let P be the point (x,y,z).
#   @ [(-1,0,0):x] and [(0,-1,0):y] and [(0,0,-1):z] are independent planes
#         through P.

#   Let [E:e] be a plane and N, N1, and N2 unit coordinate axis vectors with
#         E.N non-null.
#   @ Point (-eN:E.N) and distinct direction vectors ExN1 and ExN2 lie in the
#         plane.




