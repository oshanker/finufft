import numpy as np
import finufft

# 1. Generate 1 million mock real-world timestamps (e.g., irregular seconds)
n_points = 1_000_000
t_min, t_max = 120.5, 840.2  # Real-world start and end times (seconds)

t_real = np.random.uniform(t_min, t_max, n_points)
t_real.sort() # Sorting is optional, but common for time-series data
values_irregular = np.sin(0.05 * t_real) + 0.2 * np.random.randn(n_points)

# 2. Add padding to prevent periodic wrap-around boundary artifacts
duration = t_max - t_min
padding = duration * 0.05  # 5% padding on each side
t_start = t_min - padding
t_end = t_max + padding
padded_duration = t_end - t_start

# 3. Rescale real timestamps to the [-pi, pi] window
# Formula: t_scaled = -pi + 2 * pi * (t_real - t_start) / padded_duration
t_scaled = -np.pi + 2 * np.pi * (t_real - t_start) / padded_duration

# 4. Run the Type 1 NUFFT to get Fourier modes
uniform_grid_size = 500_000  # Number of output points you want
fourier_modes = finufft.nufft1d1(t_scaled,
            values_irregular.astype(np.complex128), uniform_grid_size, isign=1, eps=1e-9)

# 5. Convert Fourier modes back to Uniform Time Domain using standard IFFT
# Note: Shift the modes back to standard ordering before running ifft
uniform_time_padded = np.fft.ifft(np.fft.ifftshift(fourier_modes))

# 6. Reconstruct the real-world uniform time vector
# Create an evenly spaced grid across our padded real-world timeline
t_uniform_real_padded = np.linspace(t_start, t_end, uniform_grid_size, endpoint=False)

# 7. Unpad the results to match your original data limits
valid_indices = (t_uniform_real_padded >= t_min) & (t_uniform_real_padded <= t_max)
t_uniform_final = t_uniform_real_padded[valid_indices]
values_uniform_final = np.real(uniform_time_padded[valid_indices])

# 't_uniform_final' and 'values_uniform_final' are now your perfectly spaced
# data points!
import numpy as np

# 1. Extract your data
t_uniform_final = t_uniform_real_padded[valid_indices]
values_uniform_final = np.real(uniform_time_padded[valid_indices])

# 2. Combine columns side-by-side
combined_data = np.column_stack((t_uniform_final, values_uniform_final))

# 3. Save to a text file
np.savetxt(
    "../out/output_data.txt", 
    combined_data, 
    fmt="%.18e",       # Scientific notation (keeps maximum precision)
    header="Time\tValue", # Column names at the top
    delimiter="\t"     # Separated by tabs
)

print(f"Interpolated {len(t_uniform_final)} uniform points matching your original time range.")
