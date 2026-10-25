import numpy as np
import finufft
import matplotlib.pyplot as plt

# 1. Generate synthetic 1D irregularly spaced coordinates
M = 200  # Number of nonuniform source points
np.random.seed(42)

# Raw coordinates and their source strengths (true underlying curve)
x_raw = np.random.uniform(0, 10, size=M)
strengths = np.sin(2 * np.pi * x_raw / 3.0) + 0J

# 2. Define your desired uniform spatial output grid
N_out = 100
x_uniform = np.linspace(0, 10, N_out)

# 3. Preprocessing: Rescale both grids into [-pi, pi] for FINUFFT
scale_factor = 2 * np.pi / 10.0
x_src_scaled = x_raw * scale_factor - np.pi
x_tgt_scaled = x_uniform * scale_factor - np.pi

# 4. Execute Type 3 NUFFT for the signal
spatial_output = finufft.nufft1d3(x_src_scaled, strengths, x_tgt_scaled)

# 5. NORMALIZATION STEP: Compute local point density to fix the scaling discrepancy
# We run a second NUFFT where all source strengths are exactly 1.0
ones_strengths = np.ones(M, dtype=complex)
local_density = finufft.nufft1d3(x_src_scaled, ones_strengths, x_tgt_scaled)

# Divide the signal output by the local density to restore original curve scale
# Avoid division by zero by adding a tiny epsilon if needed
normalized_output = spatial_output / np.real(local_density)

# 6. Visualize the properly scaled uniform spatial output
plt.figure(figsize=(10, 5))
plt.scatter(x_raw, strengths.real, color='red', alpha=0.4, label='Original Nonuniform Data')
plt.plot(x_uniform, normalized_output.real, color='blue', linewidth=2, label='Normalized Uniform Grid Output')
plt.title("1D NUFFT Type 3: Normalized to Match Original Values")
plt.xlabel("Spatial Coordinate (x)")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
