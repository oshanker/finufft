import numpy as np
import finufft
from scipy.sparse.linalg import LinearOperator, lsqr
import matplotlib.pyplot as plt

# 1. Generate synthetic nonuniform data
M = 250  # Number of irregular points
np.random.seed(101)
x_raw = np.random.uniform(0, 10, size=M)

# Define the TRUE underlying curve
def true_signal(x):
    return np.sin(2 * np.pi * x / 3.0) + 0.5 * np.cos(2 * np.pi * x / 1.5)

# Nonuniform samples (No noise, to demonstrate exact tracking)
y_nonuniform = true_signal(x_raw) + 0J

real_array = np.array(y_nonuniform).real

combined_data = np.column_stack((x_raw, real_array))

# 4. Save to file with clean decimal formatting
np.savetxt(
    "../out/input_data.txt", 
    combined_data, 
    fmt= "%.6f",  # 4 decimals for Time, 6 decimals for Value
    header="Time\tValue", 
    delimiter="\t",
    comments=""            )


# 2. Setup uniform target grid configuration
N_grid = 128  # Number of Fourier modes / uniform output points
scale_factor = 2 * np.pi / 10.0
x_scaled = x_raw * scale_factor - np.pi

# 3. Define the Operators (Exact Mathematical Duals)
def forward_pass(u_grid):
    # Maps Fourier Modes -> Nonuniform Points
    return finufft.nufft1d2(x_scaled, u_grid)

def adjoint_pass(nu_points):
    # Maps Nonuniform Points -> Fourier Modes
    # NO DIVISION BY M. finufft.nufft1d1 is natively the exact adjoint of nufft1d2.
    return finufft.nufft1d1(x_scaled, nu_points, N_grid)

# 4. Wrap into a SciPy LinearOperator
A = LinearOperator(
    shape=(M, N_grid),
    matvec=forward_pass,    
    rmatvec=adjoint_pass   
)

# 5. Solve for the true underlying Fourier modes via LSQR
lsqr_results = lsqr(A, y_nonuniform, atol=1e-10, btol=1e-10, iter_lim=100)
modes_reconstructed = lsqr_results[0]
iters = lsqr_results[2]

# 6. FOOLPROOF SPATIAL RECONSTRUCTION:
# Create a uniform grid of target coordinates scaled to [-pi, pi)
x_uniform = np.linspace(0, 10, N_grid, endpoint=False)
x_uniform_scaled = x_uniform * scale_factor - np.pi

# Use Type 2 NUFFT to directly evaluate the modes onto the uniform spatial grid
u_spatial_grid = finufft.nufft1d2(x_uniform_scaled, modes_reconstructed)

# 7. Visualize the perfect mathematical alignment
plt.figure(figsize=(10, 5))

# Plot 1: True underlying clean signal
x_dense = np.linspace(0, 10, 500)
plt.plot(x_dense, true_signal(x_dense), color='black', linestyle='--', alpha=0.7, label='True Underlying Curve')

# Plot 2: Nonuniform sampled points 
plt.scatter(x_raw, y_nonuniform.real, color='red', alpha=0.6, edgecolors='none', s=40, label='Nonuniform Samples')

# Plot 3: Corrected Least-squares inversion result
plt.plot(x_uniform, u_spatial_grid.real, color='teal', linewidth=2.5, label='Perfectly Matched Inversion Grid')

combined_data = np.column_stack((x_uniform,  u_spatial_grid.real))

# 3. Save to a text file
np.savetxt(
    "../out/output_data.txt", 
    combined_data, 
    fmt="%.7f",       # Scientific notation (keeps maximum precision)
    header="Time\tValue", # Column names at the top
    delimiter="\t"     # Separated by tabs
)


plt.title(f"Rigorous NUFFT Inversion: Exact Match via Type 2 Spatial Evaluation ({iters} Iters)")
plt.xlabel("Spatial Coordinate (x)")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
