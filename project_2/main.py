
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    A = A.astype(np.float64)
    b = b.astype(np.float64)

    # Test if shape of matrix and vector is compatible and raise ValueError if not
    m1, n1 = A.shape
    m2 = b.shape[0]

    if m1 != m2:
        raise ValueError("Matrix and vector sizes are incompatible!")
    elif m1 != n1:
        raise ValueError("Matrix is not square!")

    # Perform gaussian elimination
    for i in range(m1):
        if use_pivoting:
            pivotelement = A[i][i]
            for k in range(i+1, m1):
                if abs(A[k][i]) > abs(pivotelement):
                    pivotelement = A[k][i]
                    A[[i, k]] = A[[k, i]]
                    b[[i, k]] = b[[k, i]]
        else:
            if A[i][i] == 0:
                raise ValueError("Pivoting is disabled but necessary!")
            else:
                pivotelement = A[i][i]
        for j in range(i, m1-1):
            factor = -(A[j+1][i]/pivotelement)
            A[j+1] += A[i] * factor
            b[j+1] += b[i] * factor

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # Test if shape of matrix and vector is compatible and raise ValueError if not
    m1, n1 = A.shape
    m2 = b.shape[0]

    if m1 != m2:
        raise ValueError("Matrix and vector sizes are incompatible!")
    elif m1 != n1:
        raise ValueError("Matrix is not square!")

    # Initialize solution vector with proper size
    x = np.zeros(n1)

    # Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    if A[m1-1][m1-1] == 0:
        raise ValueError("No OR infinite solutions exist!")

    for i in reversed(range(m1)):
        for j in reversed(range(i, m1)):
            if j != i:
                x[i] -= A[i][j] * x[j]
            else:
                x[i] += b[i]
                x[i] /= A[i][i]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L : Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # check for symmetry and raise an exception of type ValueError
    (m, n) = M.shape

    if m != n:
        raise ValueError("Matrix is not square!")

    if not np.allclose(M, M.T):
        raise ValueError("Matrix is not symmetric!")

    # build the factorization and raise a ValueError in case of a non-positive definite input matrix
    L = np.zeros((m, m))

    for j in range(n):
        for i in range(j, m):
            if i == j:
                sum = 0
                for k in range(j):
                    sum += L[i][k]**2
                if M[i][j]-sum >= 0:
                    L[i][j] = np.sqrt(M[i][j]-sum)
                else:
                    raise ValueError("Matrix is not positive definite!")
            else:
                sum = 0
                for k in range(j):
                    sum += L[i][k]*L[j][k]
                L[i][j] = 1/L[j][j] * (M[i][j]-sum)

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # Check the input for validity, raising a ValueError if this is not the case
    m, n = L.shape
    m1 = b.shape[0]

    if m != n:
        raise ValueError("Matrix is not square!")

    if not np.allclose(L, np.tril(L)):
        raise ValueError("Matrix is not a lower triangular matrix!")

    if m != m1:
        raise ValueError("Matrix and vector sizes are incompatible!")

    # Solve the system L * y = b by forward substitution
    y = np.zeros(n)

    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j]*y[j]
        y[i] = 1/L[i][i] * (b[i]-sum)

    # Solve the system L^T * x = y by backward substitution
    x = np.zeros(n)
    L_T = np.transpose(L)

    x = back_substitution(L_T, y)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # Initialize system matrix with proper size
    n_equations = n_shots * n_rays
    n_unknowns = n_grid * n_grid
    L = np.zeros((n_equations, n_unknowns))

    # Initialize intensity vector with proper size
    n_measurements = n_shots * n_rays
    g = np.zeros(n_measurements)

    # Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = np.deg2rad(np.linspace(0, 180, n_shots, endpoint=False))

    row = 0
    for i in range(len(theta)):
        # Take a measurement with the tomograph from direction r_theta.
        # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
        # ray_indices: indices of rays that intersect a cell
        # isect_indices: indices of intersected cells
        # lengths: lengths of segments in intersected cells
        # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta[i])
        for k in range(len(ray_indices)):
            L[row + ray_indices[k]][isect_indices[k]] = lengths[k]
            g[row + ray_indices[k]] = intensities[ray_indices[k]]
        row += n_rays

    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [A, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # Solve for tomographic image using Cholesky
    L = np.linalg.cholesky(A.transpose() @ A)
    x = solve_cholesky(L, A.transpose() @ g)

    # Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))
    tim = np.reshape(x, (n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
