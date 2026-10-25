import numpy as np
import finufft

# 1. Generate 1 million mock real-world timestamps (e.g., irregular seconds)
n_points = 1_000
exact_target_size = 1_000
t_min, t_max = 120.00000000, 840.00000000  # Real-world start and end times (seconds)
dt = (t_max - t_min) / (exact_target_size - 1)

# 3. Add integer point padding to the boundaries (e.g., 50,000 points on each side)
pad_points = 50
uniform_grid_size = exact_target_size + (2 * pad_points)

#t_real = np.random.uniform(t_min, t_max, n_points)
#t_real.sort() # Sorting is optional, but common for time-series data
t_real = np.linspace(t_min, t_max, n_points)
#print(f"dt {dt}")
values_irregular = np.sin(0.05 * t_real) + 0.2 * np.sin(0.1 * t_real)

# 3. Combine them into two parallel columns
combined_data = np.column_stack((t_real, values_irregular))

# 4. Save to file with clean decimal formatting
np.savetxt(
    "../out/input_data.txt", 
    combined_data, 
    fmt=["%.4f", "%.6f"],  # 4 decimals for Time, 6 decimals for Value
    header="Time\tValue", 
    delimiter="\t",
    comments=""            # Removes the default '#' character from the header line
)


# Establish the exact padded time boundaries
t_start = t_min - (pad_points * dt)
t_end = t_max + (pad_points * dt)
padded_duration = t_end - t_start

# 4. Rescale real timestamps to the [-pi, pi] window
t_scaled = -np.pi + 2 * np.pi * (t_real - t_start) / padded_duration

# 5. Run the Type 1 NUFFT
fourier_modes = finufft.nufft1d1(t_scaled, values_irregular.astype(np.complex128), uniform_grid_size,
                                 isign=1, eps=1e-9)


# . Rescale real timestamps to the [-pi, pi] window
# Formula: t_scaled = -pi + 2 * pi * (t_real - t_start) / padded_duration
t_scaled = -np.pi + 2 * np.pi * (t_real - t_start) / padded_duration


# 5. Convert Fourier modes back to Uniform Time Domain using standard IFFT
# Note: Shift the modes back to standard ordering before running ifft
uniform_time_padded = np.fft.ifft(np.fft.ifftshift(fourier_modes))

#
# 1. Extract your data
# 7. Slice out the exact 1,000,000 point window
# Because we padded by exactly 'pad_points' on each side, we slice them off cleanly
values_uniform_final = np.real(uniform_time_padded[pad_points : pad_points + exact_target_size])

# Reconstruct the exact matching real-world uniform time vector
t_uniform_final = np.linspace(t_min, t_max, exact_target_size)

# Verify the dimensions match perfectly
print(f"Final unpadded array size: {len(values_uniform_final):,}")  # Will output: 1,000,000
print(f"Start time: {t_uniform_final[0]}, End time: {t_uniform_final[-1]}")


# 2. Combine columns side-by-side
combined_data = np.column_stack((t_uniform_final, values_uniform_final))

# 3. Save to a text file
np.savetxt(
    "../out/output_data.txt", 
    combined_data, 
    fmt="%.7f",       # Scientific notation (keeps maximum precision)
    header="Time\tValue", # Column names at the top
    delimiter="\t"     # Separated by tabs
)

print(f"Interpolated {len(t_uniform_final)} uniform points matching your original time range.")
