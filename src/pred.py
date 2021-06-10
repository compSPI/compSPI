import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
from scipy.stats import special_ortho_group
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF

import viz


#
def outlier_measure(X, method="robust_covar"):
    """
    outlier_prediction
    """
    if method == "robust_covar":
        robust_cov = MinCovDet().fit(X)
        measure = np.sqrt(robust_cov.mahalanobis(X))
        offset = 3
    elif method == "isolation_forest":
        clf = IsolationForest(behaviour="new", contamination="auto")
        y_pred = clf.fit(X).predict(X)
        measure = -clf.score_samples(X)
        offset = -clf.offset_
    elif method == "local_outlier_detection":
        clf = LOF(contamination="auto")
        y_pred = clf.fit_predict(X)
        measure = -clf.negative_outlier_factor_
        offset = -clf.offset_
    assignment = np.where(measure < offset, 1, 0)
    return measure, offset, assignment


def pred3d(
    X,
    # quaternion_true,
    # defocus_true,
    defocus_min=0.5,
    defocus_max=2.5,
    #do_ellipse=False,
):
    """ """
    #
    rho = np.linalg.norm(X, axis=1)
    #
    psi = np.arccos(X[:, 0] / np.linalg.norm(X, axis=1))
    theta = np.arccos(X[:, 1] / np.linalg.norm(X[:, 1:], axis=1))
    phi = np.arccos(X[:, 2] / np.linalg.norm(X[:, 2:], axis=1))
    for i in np.arange(X.shape[0]):
        if X[i, 3] < 0.0:
            phi[i] = 2 * np.pi - phi[i]
    quaternion_pred = glomangle_to_quaternion(psi, theta, phi, as_degrees=False)
    #
    defocus_predicted = (defocus_max - defocus_min) * rescale_to_zero_one(
        rho
    ) + defocus_min
    return quaternion_pred, defocus_predicted


def pred3d_mse(
    quaternion_pred,
    # defocus_pred,
    quaternion_true,
    # defocus_true,
    ntry=10000,
):
    """ """
    mse_list = []
    #
    rotmat = np.identity(3)
    mse_min = quat_mse(rotmat, quaternion_pred, quaternion_true)
    mse_list.append(mse_min)
    for i in np.arange(ntry):
        rotmat = special_ortho_group.rvs(3)
        mse = quat_mse(rotmat, quaternion_pred, quaternion_true)
        if mse < mse_min:
            mse_min = mse
            print("{}: rMSE (degrees) = {}".format(i, 180 * np.sqrt(mse_min) / np.pi))
            rotmat_best = rotmat
        mse_list.append(mse)
    fig = plt.figure()
    plt.hist(180 * np.sqrt(mse_list) / np.pi, bins=np.int(ntry / 10))
    plt.xlabel("rMSE (degrees)")
    plt.show()
    return rotmat_best


def quat_mse(rotmat, quaternion_pred, quaternion_true):
    """ """
    rotmat_pred = R.from_quat(quaternion_pred).as_dcm()
    rotated_pred = R.from_dcm(np.dot(rotmat_pred, rotmat)).as_quat()
    dot = np.einsum("ij,ij->i", rotated_pred, quaternion_true)
    angle = 2 * np.arccos(np.abs(dot))
    return np.mean(angle ** 2)


def pred2d(
    X,
    # angle_true,
    defocus_true,
    angle_pred_sign=1.0,
    defocus_min=0.5,
    defocus_max=2.5,
    defocus_rescale="minmax",
    do_ellipse=False,
):
    """ """
    if do_ellipse:
        center, axis, theta_0, score = fitEllipse(X[:, 0:2])
        a = np.max(axis)
        b = np.min(axis)
        e = np.sqrt(1.0 - (b / a) ** 2)
        rho_centered, theta = cart2pol(X[:, 0] - center[0], X[:, 1] - center[1])
        rho_ellipse = b / (np.sqrt(1.0 - (e * np.cos(theta - theta_0)) ** 2))
        rho = rho_centered / rho_ellipse
    else:
        rho, theta = cart2pol(X[:, 0], X[:, 1])
    #
    angle_pred = np.mod(angle_pred_sign * (180 * theta / np.pi + 180) + 360, 360)
    #
    rho_normalized = rescale_to_zero_one(rho)
    if defocus_rescale == "minmax":
        a = defocus_max - defocus_min
        b = defocus_min
    elif defocus_rescale == "quartile":
        Q1_true, Q3_true = np.percentile(defocus_true, [25, 75])
        Q1_pred, Q3_pred = np.percentile(rho_normalized, [25, 75])
        a = (Q3_true - Q1_true) / (Q3_pred - Q1_pred)
        b = Q1_true - a * Q1_pred
    defocus_predicted = a * rho_normalized + b
    #
    return angle_pred, defocus_predicted


def pred2d_mse(
    angle_pred,
    defocus_pred,
    angle_true,
    defocus_true,
    angle_offset_range=np.arange(-100, 100, 10),
    # defocus_offset_range=0,
    angle_weight=None,
    norm_weights=True,
    n_periodicity=1,
):
    """ """
    #
    if angle_weight is None:
        W = np.ones(angle_pred.shape)
    else:
        W = angle_weight
    if norm_weights:
        norm = np.sum(W)
        W /= norm
    #
    modulo = 360
    angle_RMSE_list = []
    for offset in angle_offset_range:
        angle_predicted = np.mod(angle_pred + offset, modulo)
        angle_diff = (
            np.mod(angle_true - angle_predicted + modulo / 2, modulo) - modulo / 2
        )
        #
        if n_periodicity > 1:
            angle_period = 2.0 * np.pi / 3.0
            angle_diff1 = np.mod(angle_diff, angle_period)
            angle_diff2 = np.mod(angle_period - angle_diff, angle_period)
            angle_diff = np.min(angle_diff1, angle_diff2)
        #
        angle_MSE = np.dot(W, (angle_diff) ** 2)
        angle_RMSE_list.append(np.sqrt(angle_MSE))
        print("offset: {} angle RMSE = {}".format(offset, np.sqrt(angle_MSE)))
    #
    defocus_MSE = np.mean((defocus_pred - defocus_true) ** 2)
    defocus_RMSE = np.sqrt(defocus_MSE)
    print("defocus RMSE = {}".format(defocus_RMSE))
    return angle_RMSE_list, defocus_RMSE


def rescale_to_zero_one(X):
    """ """
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def cart2pol(x, y):
    """
    copied from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    phi: angle in radians, in the range [-pi, pi]
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """
    copied from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def cart2sph(x, y, z):
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = x ** 2 + y ** 2
    r = np.sqrt(xy + z ** 2)
    elev = np.arctan2(np.sqrt(xy), z)  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))
    # for elevation angle defined from XY-plane up
    azim = np.arctan2(y, x)
    return r, elev, azim


####################
# ANGLE CONVERSION #
####################


def relangle_to_quaternion(rot, tilt, psi, as_degrees=True):
    """ """
    rotmat = R.from_euler(
        "ZYZ", np.stack((rot, tilt, psi), axis=0).T, degrees=as_degrees
    )
    quaternion = rotmat.as_quat()
    return quaternion


def glomangle_to_quaternion(psi, theta, phi, as_degrees=True):
    """ """
    quaternion = np.empty((psi.shape[0], 4))
    degrees_to_radian = np.pi / 180.0
    if as_degrees:
        psi = degrees_to_radian * psi
        theta = degrees_to_radian * theta
        phi = degrees_to_radian * phi
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    quaternion[:, 0] = spsi * ctheta
    quaternion[:, 1] = spsi * stheta * cphi
    quaternion[:, 2] = spsi * stheta * sphi
    quaternion[:, 3] = cpsi
    return quaternion


####################################
# Non-linear fitting to an ellipse #
####################################
def fitEllipse(data):
    """fitEllipse
    from https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python/48002645
    see also http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    x = data[:, 0]
    y = data[:, 1]
    x = x[:, None]  # np.newaxis]
    y = y[:, None]  # np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    score = np.dot(a, np.dot(S, a))
    return (
        ellipse_center(a),
        ellipse_axis_length(a),
        ellipse_angle_of_rotation(a),
        score,
    )


def ellipse_center(a):
    """ellipse_center"""
    b, c, d, f, _, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    """ellipse_angle_of_rotation"""
    b, c, _, _, _, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi / 2
    else:
        if a > c:
            return np.arctan(2 * b / (a - c)) / 2
        else:
            return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2


def ellipse_axis_length(a):
    """ellipse_axis_length"""
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(abs(up / down1))
    res2 = np.sqrt(abs(up / down2))
    return np.array([res1, res2])


################
# CONE FITTING #
################
def rotate_to_fit_cone(X, ntry):
    """rotate_to_fit_cone"""
    dim = X.shape[1]
    score_list = []
    rotmat = np.identity(dim)  # np.diag([1,1,1])
    X_rotated = np.dot(rotmat, X.T).T
    dist = rescale_to_zero_one(np.linalg.norm(X_rotated[:, 0 : dim - 1], axis=1))
    popt, pcov = curve_fit(linear_1d, X_rotated[:, dim - 1], dist)
    score = np.abs(popt[0])
    #
    score_max = score
    rotmat_max = rotmat
    ibest = 0
    score_list.append(score)
    #
    print("   current best ({}/{}): {}".format(ibest, ntry, score_max))
    for i in np.arange(1, ntry):
        rotmat = special_ortho_group.rvs(dim)
        X_rotated = np.dot(rotmat, X.T).T
        dist = rescale_to_zero_one(np.linalg.norm(X_rotated[:, 0 : dim - 1], axis=1))
        popt, pcov = curve_fit(linear_1d, X_rotated[:, dim - 1], dist)
        score = np.abs(popt[0])
        if score > score_max:
            score_max = score
            ibest = i
            rotmat_max = rotmat
            print("   current best ({}/{}): {}".format(ibest, ntry, score_max))
        score_list.append(score)
    X_best = np.dot(rotmat_max, X.T).T
    fig = plt.figure()
    plt.hist(score_list, bins=np.int(ntry / 10))
    plt.show()
    return X_best


def rotate_to_fit_cone_2d(X, ntry):
    """rotate_to_fit_cone_2d"""
    score_list = []
    rotmat = np.diag([1, 1, 1])
    X_rotated = np.dot(rotmat, X[:, 0:3].T).T
    dist = rescale_to_zero_one(np.linalg.norm(X_rotated[:, 0:2], axis=1))
    popt, pcov = curve_fit(linear_1d, X_rotated[:, 2], dist)
    score = np.abs(popt[0])
    #
    score_max = score
    rotmat_max = rotmat
    ibest = 0
    score_list.append(score)
    #
    print("   current best ({}/{}): {}".format(ibest, ntry, score_max))
    for i in np.arange(1, ntry):
        rotmat = special_ortho_group.rvs(3)
        X_rotated = np.dot(rotmat, X[:, 0:3].T).T
        dist = rescale_to_zero_one(np.linalg.norm(X_rotated[:, 0:2], axis=1))
        popt, pcov = curve_fit(linear_1d, X_rotated[:, 2], dist)
        score = np.abs(popt[0])
        if score > score_max:
            score_max = score
            ibest = i
            rotmat_max = rotmat
            print("   current best ({}/{}): {}".format(ibest, ntry, score_max))
        score_list.append(score)
    X_best = np.dot(rotmat_max, X[:, 0:3].T).T
    fig = plt.figure()
    plt.hist(score_list, bins=np.int(ntry / 10))
    plt.show()
    return X_best


def linear_1d(x, A, B):
    """linear_1d"""
    return A * x + B


#######################
# GENERAL QUADRIC FIT #
#######################


def quadric_model(x, *args):
    """ """
    dim = x.shape[0]
    narg = len(args)
    if narg != dim * (dim + 1) / 2:
        print("Error in dim {} != {}".format(narg, dim * (dim + 1) / 2))
        y_p = 0
    else:
        A = quadric_matrix(dim, *args)
        y_p = np.diag(np.dot(x.T, np.dot(A, x)))
    return y_p


def quadric_matrix(dim, *args, set_a=False):
    """
    in the 3D case, args are given as: a,b,c,d,e,f,g,h,i,j and
    A = [  a  d/2 e/2 g/2 ]
        [ d/2  b  f/2 h/2 ]
        [ e/2 f/2  c  i/2 ]
        [ g/2 h/2 i/2  j  ]
    """
    A = np.zeros((dim, dim))
    n = 0
    for i in np.arange(dim - 1):
        A[i, i] = args[i]
        for j in np.arange(i + 1, dim - 1):
            A[i, j] = args[n] / 2
            A[j, i] = args[n] / 2
        n += 1
    for j in np.arange(dim - 1):
        A[j, dim - 1] = args[n] / 2
        A[dim - 1, j] = args[n] / 2
        n += 1
    A[dim - 1, dim - 1] = args[n]
    if set_a:
        A[0, 0] = 1
    return A
