import numpy as np


####################################################################################################
# Exercise 1: Function Roots

def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0, n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    # set meaningful minimal interval size if not given as parameter, e.g. 10 * eps
    if ival_size == -1.0:
        ival_size = 10 * np.finfo(float).eps

    # intialize iteration
    fl = f(lival)
    fr = f(rival)

    # make sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))

    # loop until final interval is found, stop if max iterations are reached
    n_iterations = 0

    while ((np.abs(rival-lival) >= ival_size) and (n_iterations < n_iters_max)):
        x = (lival + rival) / 2
        if f(lival) * f(x) > 0:
            lival = x
        else:
            rival = x
        n_iterations += 1

    # calculate final approximation to root
    root = x

    return root


def func_f(x):
    return x**3 - 2*x + 2 # -1.76929235423863

def deri_f(x):
    return 3 * x**2 - 2

def func_g(x):
    return 6*x/(x**2 + 1)

def deri_g(x):
    return 6 * (1 - x**2) / (x**2 + 1)**2

def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert(n_iters_max > 0)

    # Initialize root with start value
    root = start

    # chose meaningful convergence criterion eps, e.g 10 * eps
    eps = 10 * np.finfo(float).eps

    # Initialize iteration
    fc = f(root)
    dfc = df(root)
    n_iterations = 0

    # loop until convergence criterion eps is met
    while np.abs(f(root)) >= eps:
        # return root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid divergence)
        if (np.abs(df(root)) < eps) or (np.abs(root) > 1e5):
            return root, n_iters_max+1

        # update root value and function/dfunction values
        root = root - (f(root)/df(root))

        # avoid infinite loops and return (root, n_iters_max+1)
        if n_iterations >= n_iters_max:
            return root, n_iters_max+1

        n_iterations += 1

    return root, n_iterations


####################################################################################################
# Exercise 2: Newton Fractal

def generate_newton_fractal(f: object, df: object, roots: np.ndarray, sampling: np.ndarray, n_iters_max: int=20) -> np.ndarray:
    """
    Generates a Newton fractal for a given function and sampling data.

    Arguments:
    f: function (handle)
    df: derivative of function (handle)
    roots: array of the roots of the function f
    sampling: sampling of complex plane as 2d array
    n_iters_max: maxium number of iterations the newton method can calculate to find a root

    Return:
    result: 3d array that contains for each sample in sampling the index of the associated root and the number of iterations performed to reach it.
    """

    result = np.zeros((sampling.shape[0], sampling.shape[1], 2), dtype=int)

    # iterate over sampling grid
    for i in range(sampling.shape[0]):
        for j in range(sampling.shape[1]):

            # run Newton iteration to find a root and the iterations for the sample (in maximum n_iters_max iterations)
            root, n_iters = find_root_newton(f, df, sampling[i][j], n_iters_max)

            # determine the index of the closest root from the roots array. The functions np.argmin and np.tile could be helpful.
            index = 0
            for k in range(roots.size):
                if np.abs(roots[k] - root) < np.abs(roots[index] - root):
                    index = k

            # write the index and the number of needed iterations to the result
            # result[i, j] = np.array([index, n_iters_max+1])
            result[i][j] = np.array([index, n_iters])

    return result


####################################################################################################
# Exercise 3: Minimal Surfaces

def surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """
    Calculate the area of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    area: the total surface area
    """

    # initialize area
    area = 0.0

    # iterate over all triangles and sum up their area
    for i in range(f.shape[0]):
        A = v[f[i][0]]
        B = v[f[i][1]]
        C = v[f[i][2]]
        a = np.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2 + (B[2]-C[2])**2)
        b = np.sqrt((A[0]-C[0])**2 + (A[1]-C[1])**2 + (A[2]-C[2])**2)
        c = np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2)
        p = a + b + c
        s = p/2
        T = np.sqrt(s*(s-a)*(s-b)*(s-c))
        area += T

    return area


def surface_area_gradient(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the area gradient of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    gradient: the surface area gradient of all vertices in v
    """

    # intialize the gradient
    gradient = np.zeros(v.shape)

    # iterate over all triangles and sum up the vertices gradients
    for i in range(v.shape[0]): # Berechne Gradienten der i-ten Ecke =grad
        grad = 0
        for j in range(f.shape[0]): # Berechne Gradienten der Dreiecke die mit der i-ten Ecke verbunden sind =tmp , diese werden auf grad addiert
            tmp = 0
            if i in f[j]:
                if i == f[j][0]:
                    A = v[f[j, 0]]
                    B = v[f[j, 1]]
                    C = v[f[j, 2]]
                elif i == f[j][1]:
                    A = v[f[j, 1]]
                    B = v[f[j, 0]]
                    C = v[f[j, 2]]
                else:
                    A = v[f[j, 2]]
                    B = v[f[j, 0]]
                    C = v[f[j, 1]]
                n = np.cross(B-C, A-C) / np.linalg.norm(np.cross(B-C, A-C))
                tmp = np.cross(B-C, n)
                grad += tmp
        gradient[i] = grad

    return gradient


def gradient_descent_step(v: np.ndarray, f: np.ndarray, c: np.ndarray, epsilon: float=1e-6, ste=1.0, fac=0.5) -> (bool, float, np.ndarray, np.ndarray):
    """
    Calculate the minimal area surface for the given triangles in v/f and boundary representation in c.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i
    c: list of vertex indices which are fixed and can't be moved
    epsilon: difference tolerance between old area and new area

    Return:
    converged: flag that determines whether the function converged
    area: new surface area after the gradient descent step
    updated_v: vertices with changed positions
    gradient: calculated gradient
    """

    # calculate gradient and area before changing the surface
    gradient = surface_area_gradient(v, f)
    area = surface_area(v, f)

    # calculate indices of vertices whose position can be changed
    indices = np.zeros(v.shape[0])

    for i in range(indices.size):
        if i not in c:
            indices[i] = 1

    # find suitable step size so that area can be decreased
    step = 10
    conv = False

    while not conv:
        v_tmp = v.copy()
        v_tmp[indices==1] = v_tmp[indices==1] + step * gradient[indices==1]
        new = surface_area(v_tmp, f)
        if np.abs(area - new) < epsilon:
            return True, new, v_tmp, gradient
        elif new < area:
            conv = True
        else:
            step *= fac

    return False, new, v_tmp, gradient


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
