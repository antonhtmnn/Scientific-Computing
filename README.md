# Scientific-Computing

This repository contains a collection of projects related to scientific computing. Each project focuses on different computational techniques such as numerical methods, linear algebra, interpolation, optimization, Fourier transforms, and more. The projects are implemented in Python, using various mathematical and numerical algorithms.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Projects](#projects)
  - [Project 1: Efficiency of Calculations in NumPy](#project-1-efficiency-of-calculations-in-numpy)
  - [Project 2: Gaussian Elimination and Cholesky Decomposition](#project-2-gaussian-elimination-and-cholesky-decomposition)
  - [Project 3: Eigenfaces for Face Recognition](#project-3-eigenfaces-for-face-recognition)
  - [Project 4: Interpolation and Keyframe Animation](#project-4-interpolation-and-keyframe-animation)
  - [Project 5: Fourier Transforms](#project-5-fourier-transforms)
  - [Project 6: Root Finding and Optimization](#project-6-root-finding-and-optimization)

## Setup and Installation

To run the projects in this repository, you need to set up a Python environment. Follow these steps:

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/antonhtmnn/Scientific-Computing.git
cd Scientific-Computing
```

### Step 2: Set Up a Python Virtual Environment

It's recommended to use a virtual environment to manage dependencies. To create a virtual environment, run:

```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

Activate the virtual environment:

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Step 5: Running the Projects

Navigate to the desired project folder and execute the main Python file. For example, to run the first project:

```bash
cd project_1
python main.py
```

## Projects

### Project 1: Efficiency of Calculations in NumPy

This project compares the performance of NumPy with a simple Python implementation for matrix multiplication. It also explores floating-point precision and rotation matrices in 2D space.

- **Key Functions**:
  - `matrix_multiplication()`: Implements matrix multiplication without NumPy.
  - `compare_multiplication()`: Compares the performance of custom and NumPy-based matrix multiplication.
  - `machine_epsilon()`: Calculates machine epsilon for different floating-point formats.
  - `rotation_matrix()`: Generates a rotation matrix for a given angle.
  - `inverse_rotation()`: Computes the inverse of a given rotation matrix.

### Project 2: Gaussian Elimination and Cholesky Decomposition

This project focuses on solving linear systems using Gaussian elimination and Cholesky decomposition.

- **Key Functions**:
  - `gaussian_elimination()`: Implements Gaussian elimination with and without pivoting.
  - `back_substitution()`: Solves linear equations using back substitution.
  - `compute_cholesky()`: Computes the Cholesky decomposition of a matrix.
  - `solve_cholesky()`: Solves a linear system using Cholesky decomposition.

### Project 3: Eigenfaces for Face Recognition

This project implements the Eigenface algorithm for face recognition using principal component analysis (PCA).

- **Key Functions**:
  - `load_images()`: Loads images from a directory.
  - `setup_data_matrix()`: Constructs the data matrix for PCA.
  - `calculate_pca()`: Computes the principal components.
  - `accumulated_energy()`: Determines the number of components required to capture a specified percentage of variance.
  - `project_faces()`: Projects images onto the eigenbasis.
  - `identify_faces()`: Identifies faces by comparing test images to training images.

### Project 4: Interpolation and Keyframe Animation

This project deals with polynomial interpolation and spline interpolation, particularly in the context of animating keyframes.

- **Key Functions**:
  - `lagrange_interpolation()`: Implements Lagrange polynomial interpolation.
  - `hermite_cubic_interpolation()`: Implements Hermite cubic interpolation.
  - `natural_cubic_interpolation()`: Implements natural cubic spline interpolation.
  - `periodic_cubic_interpolation()`: Implements periodic cubic spline interpolation.

### Project 5: Fourier Transforms

This project involves implementing the discrete Fourier transform (DFT) and the fast Fourier transform (FFT).

- **Key Functions**:
  - `dft_matrix()`: Constructs the DFT matrix.
  - `is_unitary()`: Checks if a matrix is unitary.
  - `create_harmonics()`: Computes the DFT of delta impulses.
  - `shuffle_bit_reversed_order()`: Reorders data for FFT using bit-reversal.
  - `fft()`: Implements the Cooley-Tukey FFT algorithm.

### Project 6: Root Finding and Optimization

This project focuses on root-finding algorithms and optimization techniques such as gradient descent.

- **Key Functions**:
  - `find_root_bisection()`: Implements the bisection method for root finding.
  - `find_root_newton()`: Implements Newton's method for root finding.
  - `generate_newton_fractal()`: Generates fractals using Newton's method.
  - `surface_area()`: Calculates the surface area of a mesh.
  - `surface_area_gradient()`: Computes the gradient of the surface area.
  - `gradient_descent_step()`: Executes a step of gradient descent for surface minimization.
