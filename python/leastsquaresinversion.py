import numpy as np
import finufft
from scipy.sparse.linalg import LinearOperator, lsqr
import matplotlib.pyplot as plt

# =====================================================================
# 1. PARAMETERS: Define your time axis range and grid size here
# =====================================================================
xmin = 0.0       # Start of your time/spatial axis (can be negative or positive)
xmax = 150.125       # End of your time/spatial axis
domain_length = xmax - xmin

M = 250000           # Number of irregular, nonuniform data points
N_grid = 150000      # Number of regular uniform output points / Fourier modes

np.random.seed(101)

variation_scale = 1.0
# Define an underlying test curve relative to the chosen domain
def true_signal(x):
    # Scales frequencies dynamically based on your chosen domain length
    return np.sin( 3 * (x) / variation_scale) + \
           0.5 * np.cos( 6 * (x) / variation_scale)


# =====================================================================
# 2. GENERATE NONUNIFORM DATA WITHIN YOUR DEFINED RANGE
# =====================================================================
# Generate raw irregular coordinates strictly inside [xmin, xmax]
x_raw = np.random.uniform(xmin, xmax, size=M)
print(f" x_raw[0] {x_raw[0]} {true_signal(x_raw[0])} ")


# Generate nonuniform samples (No noise, to verify exact tracking)
y_nonuniform = true_signal(x_raw) + 0J

combined_data = np.column_stack((x_raw, y_nonuniform.real))

# 4. Save to file with clean decimal formatting
np.savetxt(
    "../out/input_data.txt", 
    combined_data, 
    fmt= "%.6f",  # 4 decimals for Time, 6 decimals for Value
    header="Time\tValue", 
    delimiter="\t",
    comments=""            )


# =====================================================================
# 3. DYNAMIC SCALING FOR FINUFFT MAPS BOUNDS TO [-pi, pi]
# =====================================================================
# Formula: scaled = ((x - xmin) / domain_length) * 2*pi - pi
x_scaled = ((x_raw - xmin) / domain_length) * 2.0 * np.pi - np.pi

# =====================================================================
# 4. DEFINE MATHEMATICAL DUAL OPERATORS
# =====================================================================
def forward_pass(u_grid):
    return finufft.nufft1d2(x_scaled, u_grid)

def adjoint_pass(nu_points):
    return finufft.nufft1d1(x_scaled, nu_points, N_grid)

A = LinearOperator(
    shape=(M, N_grid),
    matvec=forward_pass,    
    rmatvec=adjoint_pass   
)

# =====================================================================
# 5. SOLVE UNIFORM COEFFICIENTS VIA LSQR
# =====================================================================
#lsqr_results = lsqr(A, y_nonuniform, atol=1e-10, btol=1e-10, iter_lim=100)
# SOLVE WITH DAMPENING (THE CRITICAL FIX)
# =====================================================================
# damp=1e-2 acts as an L2 penalty to suppress wild, unstable high-frequency modes
# which naturally occur when N_grid >= M with nonuniform spacing.
iters = 5000
lsqr_results = lsqr(A, y_nonuniform, atol=1e-15, btol=1e-15, damp=1.0e-5,
                    iter_lim=iters)


modes_reconstructed = lsqr_results[0]
iters = lsqr_results[2]

# =====================================================================
# 6. FOOLPROOF UNIFORM GRID EVALUATION
# =====================================================================
# Generate regular target coordinates within the user defined range

umin = xmin+0.125       # Start of your time/spatial axis (can be negative or positive)
umax = xmax       # End of your time/spatial axis
u_domain_length = umax - umin

print(f" umin {umin} {true_signal(umin)} ")

x_uniform = np.linspace(umin, umax, N_grid, endpoint=False)
print(f" x_uniform[1] {x_uniform[1]} {true_signal(x_uniform[1])} ")

# Scale uniform targets exactly like the source coordinates into [-pi, pi]
x_uniform_scaled = ((x_uniform - xmin) / domain_length) * 2.0 * np.pi - np.pi

# Map optimized Fourier coefficients directly to the uniform spatial targets
u_spatial_grid = finufft.nufft1d2(x_uniform_scaled, modes_reconstructed)

y_test = true_signal(x_uniform)
mae = np.mean(np.abs(u_spatial_grid.real -  y_test))
print(f"Mean Absolute Deviation: {mae}")

mae1 = np.mean(np.abs(u_spatial_grid.real[100:-100] -  y_test[100:-100]))
print(f"Mean Absolute Deviation: {mae1}")

combined_data = np.column_stack((x_uniform,  u_spatial_grid.real, y_test,
                                 u_spatial_grid.real -  y_test))

# 3. Save to a text file
np.savetxt(
    "../out/output_data.txt", 
    combined_data, 
    fmt="%.7f",       # Scientific notation (keeps maximum precision)
    header="Time\tValue", # Column names at the top
    delimiter="\t"     # Separated by tabs
)


plt.figure(figsize=(10, 5))

x_dense = np.linspace(xmin, xmax, 500)
plt.plot(x_dense, true_signal(x_dense), color='black', linestyle='--', alpha=0.7, label='True Underlying Curve')

plt.scatter(x_raw, y_nonuniform.real, color='red', alpha=0.6, edgecolors='none', s=40, label='Nonuniform Samples')

plt.scatter(x_uniform, u_spatial_grid.real, color='teal', marker='+',
         linewidth=2.5, label='Regularized Inversion Grid')



plt.title(f"Parameterized NUFFT Inversion on Bound [{xmin}, {xmax}] ({iters} Iters)")
plt.xlabel("Time / Coordinate Axis")
plt.ylabel("Signal Value")
plt.xlim(xmin, xmax)  # Forces plot view to match parameters exactly
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
