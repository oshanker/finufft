import finufft
import numpy as np
import time
from scipy.sparse.linalg import cg, LinearOperator

# 1. Setup exact problem sizes (N = 120,000 points, K = 120,000 target modes)
N = 1200000
K_modes = 1200000  # Total frequency terms

# Generate dummy scattered data scaled to [-pi, pi] for NUFFT compatibility
np.random.seed(42)
x_scattered = np.random.uniform(-np.pi, np.pi, size=N)

# True complex signal + some random white noise
true_signal = np.sin(x_scattered) + 0.5 * np.cos(5 * x_scattered)
y_scattered = (true_signal ).astype(
    np.complex128
)


# 2. Define the Implicit Forward and Adjoint Operators
# Forward operation: Given complex Fourier coefficients, evaluate at scattered points x (NUFFT Type 2)
def forward_nufft(coeffs):
  # finufft expects the input coefficients to match the number of modes
  return finufft.nufft1d2(x_scattered, coeffs)


# Adjoint operation: Map errors from scattered space back to frequency space (NUFFT Type 1)
def adjoint_nufft(grid_values):
  return finufft.nufft1d1(x_scattered, grid_values, K_modes)


# 3. Wrap into a Scipy LinearOperator for Conjugate Gradient (Least-Squares equivalent)
# The normal equations we want to solve iteratively: (A^H * A) * c = A^H * y
def normal_equation_matvec(coeffs):
  # Forward projection to scattered space
  fwd = forward_nufft(coeffs)
  # Adjoint projection back to Fourier space
  return adjoint_nufft(fwd)

# Record the start time
start_time = time.perf_counter()


# Create the structural linear operator mapping Fourier Space -> Fourier Space
A_matrix_operator = LinearOperator(
    shape=(K_modes, K_modes), matvec=normal_equation_matvec, dtype=np.complex128
)

# Compute the Right-Hand Side vector for the normal equations: b = A^H * y
b_vector = adjoint_nufft(y_scattered)

# 4. Iteratively solve for the Fourier coefficients
print("Starting Iterative Conjugate Gradient solver...")
# 'tol' controls precision, 'maxiter' keeps execution time bounded
fourier_coefficients, info = cg(A_matrix_operator, b_vector, \
                                rtol=1e-6, maxiter=1500)

if info == 0:
  print("Convergence successful!")
else:
  print(f"Solver stopped. Info code: {info} (Max iterations reached or stalled)")

end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

# 5. Result Verification
# Reconstruct the fitted signal at our original scattered positions to verify accuracy
fitted_y_scattered = forward_nufft(fourier_coefficients)
residual_norm = np.linalg.norm(y_scattered - fitted_y_scattered) / np.linalg.norm(
    y_scattered
)
print(f"Relative Least-Squares Error: {residual_norm:.5f}")
mad = np.mean(np.abs((y_scattered - fitted_y_scattered)))
print(f"Mean Absolute Deviation: {mad}")
# 1. Generate the uniform x-coordinate grid matching your domain
# We use endpoint=False because periodic grids wrap around \
# (the point at +pi is identical to -pi)
x_grid = np.linspace(-np.pi, np.pi, num=K_modes, endpoint=False)

# 2. Transform the Fourier coefficients into physical y-values on that grid
# (Using the IFFT method shown previously)
y_grid_complex = np.fft.ifft(np.fft.ifftshift(fourier_coefficients)) * K_modes

# 3. Extract the real part (since your physical function values are real)
y_grid = y_grid_complex.real

# Now you have perfect pairs: (x_grid[i], y_grid[i])
print("First 5 grid coordinates (x):", x_grid[:5])
print("First 5 fitted grid values (y):", y_grid[:5])

