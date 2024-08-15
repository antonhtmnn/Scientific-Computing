import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    # Generate Lagrange base polynomials and interpolation polynomial
    for i in range(x.size):
        base_function = 1
        for k in range(x.size):
            if k != i:
                tmp_zaehler = np.poly1d([1, -x[k]])
                tmp_nenner = x[i]-x[k]
                tmp = tmp_zaehler / tmp_nenner
                base_function *= tmp
        base_functions.append(base_function)
        polynomial += base_functions[i] * y[i]

    return polynomial, base_functions


def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # compute piecewise interpolating cubic polynomials
    offset = 0
    for i in range(x.size-1):
        H = np.zeros((4, 4))
        Y = np.zeros(4)
        for j in range(H.shape[0]):
            if j < 2:
                Y[j] = y[j+offset]
            else:
                Y[j] = yp[j-2+offset]
            for k in range(H.shape[1]):
                if k == 0:
                    if j < 2:
                        H[j][k] = x[j+offset]**3
                    else:
                        H[j][k] = 3 * x[j-2+offset]**2
                elif k == 1:
                    if j < 2:
                        H[j][k] = x[j+offset]**2
                    else:
                        H[j][k] = 2 * x[j-2+offset]
                elif k == 2:
                    if j < 2:
                        H[j][k] = x[j+offset]
                    else:
                        H[j][k] = 1
                else:
                    if j < 2:
                        H[j][k] = 1
                    else:
                        H[j][k] = 0
        c = np.linalg.solve(H, Y)
        p = np.poly1d([c[0], c[1], c[2], c[3]])
        spline.append(p)
        offset += 1

    return spline


####################################################################################################
# Exercise 2: Animation

def construct_linear_system(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, int):
    """
    Construct linear system without boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    M: matrix (last two rows are 'reserved' for boundary conditions)
    Y: vector
    """

    n = x.size
    M = np.zeros((4*n-4, 4*n-4))
    Y = np.zeros(4*n-4)

    offset_y = 0
    offset_rows = 0
    offset_columns = 0
    for i in range(n-1):
        for j in range(4):
            if j < 2:
                Y[j+offset_rows] = y[offset_y]
            else:
                Y[j+offset_rows] = 0

            for k in range(4):
                if k == 0:
                    if j < 2:
                        M[j+offset_rows][k+offset_columns] = x[offset_y]**3
                    elif j == 2:
                        M[j+offset_rows][k+offset_columns] = 3*x[offset_y]**2
                        if i != n-2:
                            M[j+offset_rows][k+offset_columns+4] = -(3*x[offset_y]**2)
                    else:
                        M[j+offset_rows][k+offset_columns] = 6*x[offset_y]
                        if i != n-2:
                            M[j+offset_rows][k+offset_columns+4] = -(6*x[offset_y])
                elif k == 1:
                    if j < 2:
                        M[j+offset_rows][k+offset_columns] = x[offset_y]**2
                    elif j == 2:
                        M[j+offset_rows][k+offset_columns] = 2*x[offset_y]
                        if i != n-2:
                            M[j+offset_rows][k+offset_columns+4] = -(2*x[offset_y])
                    else:
                        M[j+offset_rows][k+offset_columns] = 2
                        if i != n-2:
                            M[j+offset_rows][k+offset_columns+4] = -2
                elif k == 2:
                    if j < 2:
                        M[j+offset_rows][k+offset_columns] = x[offset_y]
                    elif j == 2:
                        M[j+offset_rows][k+offset_columns] = 1
                        if i != n-2:
                            M[j+offset_rows][k+offset_columns+4] = -1
                else:
                    if j < 2:
                        M[j+offset_rows][k+offset_columns] = 1
            if j == 0:
                offset_y += 1
        offset_rows += 4
        offset_columns += 4

    return M, Y, n


def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)

    # construct linear system with natural boundary conditions
    M, Y, n = construct_linear_system(x, y)

    # resort linear system (clear first and last row)
    M_new = np.zeros((4*n-4, 4*n-4))
    Y_new = np.zeros(4*n-4)

    for i in range(M.shape[0]-1):
        M_new[i+1] = M[i]
        Y_new[i+1] = Y[i]

    # insert natural boundary conditions (first and last row)
    first = np.zeros(4*n-4)
    first[0] = 6*x[0]
    first[1] = 2
    M_new[0] = first

    last = np.zeros(4*n-4)
    last[-4] = 6*x[-1]
    last[-3] = 2
    M_new[-1] = last

    # solve linear system for the coefficients of the spline
    coeffs = np.linalg.solve(M_new, Y_new)

    # extract local interpolation coefficients from solution
    spline = []
    offset_param = 0

    for i in range(n-1):
        a = coeffs[0+offset_param]
        b = coeffs[1+offset_param]
        c = coeffs[2+offset_param]
        d = coeffs[3+offset_param]
        spline.append(np.poly1d([a, b, c, d]))
        offset_param += 4

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)

    # construct linear system with periodic boundary conditions
    M, Y, n = construct_linear_system(x, y)

    first = np.zeros(4*n-4)
    first[0] = 3*x[0]**2
    first[1] = 2*x[0]
    first[2] = 1
    first[-4] = -(3*x[-1]**2)
    first[-3] = -(2*x[-1])
    first[-2] = -1
    M[-2] = first

    last = np.zeros(4*n-4)
    last[0] = 6*x[0]
    last[1] = 2
    last[-4] = -(6*x[-1])
    last[-3] = -2
    M[-1] = last

    # solve linear system for the coefficients of the spline
    coeffs = np.linalg.solve(M, Y)

    # extract local interpolation coefficients from solution
    spline = []
    offset_param = 0

    for i in range(n-1):
        a = coeffs[0+offset_param]
        b = coeffs[1+offset_param]
        c = coeffs[2+offset_param]
        d = coeffs[3+offset_param]
        spline.append(np.poly1d([a, b, c, d]))
        offset_param += 4

    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
