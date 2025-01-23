import numpy as np


def assemble_matrix(points_2d: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Generates a matrix of the form:

    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1
      .  .  .  . .  .  .  .    .     .      .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]

    given a set of 2D points (u,v) and 3D points (X,Y,Z).

    You will use this matrix to solve for the projection matrix with SVD,
    or with minor modification, NumPy's least squares implementation.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        A: A matrix containing 2D and 3D points, as part of a system
    """
    A = []
    for i in range(points_2d.shape[0]):
        u, v = points_2d[i]
        X, Y, Z = points_3d[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    return np.array(A)



def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs. If you are using
    least squares, you will need to modify the last column of the matrix to
    avoid a degenerate solution.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    U, S, Vt = np.linalg.svd(assemble_matrix(points_2d, points_3d))
    M = Vt[-1].reshape(3, 4)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates. Performed via multiplying
    the projection matrix and 3D points vectors. Remember to divide each
    projected point by the homogenous coordinate to get the final set of
    image coordinates.

    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    ones = np.ones((points_3d.shape[0], 1))
    homogeneous_points_3d = np.hstack((points_3d, ones))
    projected_homogeneous_points = P @ homogeneous_points_3d.T

    s = projected_homogeneous_points[2, :] + 1e-10
    
    projected_points_2d = (projected_homogeneous_points[:2, :] / s).T 


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix. Equations 5
    and 6 from the documentation will be helpful here.

    A useful method will be np.linalg.inv().

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    Q = M[:, :3]
    m4 = M[:, 3]
    
    try:
        Q_inv = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        Q_inv = np.linalg.pinv(Q)
    
    cc = -Q_inv @ m4
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
