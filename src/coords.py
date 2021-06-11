import numpy as np
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SO3 = SpecialOrthogonal(n=3, point_type="vector")


def coords_n_by_d(coords_1d=None, N=None, d=3):
    if coords_1d is not None and N is None:
        pass
    elif coords_1d is None and N is not None:
        assert N % 2 == 0, "N must be even"
        coords_1d = np.arange(-N // 2, N // 2)
    else:
        assert False, "use either N or coords_1d, not both"

    if d == 2:
        X = np.meshgrid(coords_1d, coords_1d)
    elif d == 3:
        X = np.meshgrid(coords_1d, coords_1d, coords_1d)
    coords = np.zeros((X[0].size, d))
    for di in range(d):
        coords[:, di] = X[di].flatten()
    if d == 3:
        coords[:, [0, 1]] = coords[:, [1, 0]]

    return coords


def deg_to_rad(deg):
    return deg * np.pi / 180


def get_random_quat(num_pts, method="sphere"):
    """
    Get num_pts of unit quaternions with a uniform random distribution.
    :param num_pts: The number of quaternions to return
    : param method:
      hemisphere: uniform on the 4 hemisphere, with x in [0,1], y,z in [-1,1]
      sphere: uniform on the sphere, with x,y,z in [-1,1]
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    u = np.random.rand(3, num_pts)
    u1, u2, u3 = [u[x] for x in range(3)]

    quat = np.zeros((4, num_pts))
    if method == "hemisphere":
        angle = np.pi / 2
    elif method == "sphere":
        angle = 2 * np.pi
    else:
        assert False, "use hemisphere or sphere"

    quat[0] = np.sqrt(1 - u1) * np.sin(angle * u2)
    quat[1] = np.sqrt(1 - u1) * np.cos(angle * u2)
    quat[2] = np.sqrt(u1) * np.sin(angle * u3)
    quat[3] = np.sqrt(u1) * np.cos(angle * u3)

    return np.transpose(quat)


def uniform_rotations(num):
    qs = get_random_quat(num)
    rots = SO3.matrix_from_quaternion(qs)  # num,3,3
    return (rots, qs)
