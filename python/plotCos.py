import numpy as np
import matplotlib.pyplot as plt

# Define the constants given in the prompt
T = 0.2435990469039924426798
x_start = 243.53396107975615984
x_end = x_start + 10 * T

# Generate 1,000 evenly spaced points over the specified range
x = np.linspace(x_start, x_end, 1000)

# Evaluate your exact formula: cos(pi * (x - x_start) / T)
y = np.cos(np.pi * (x - x_start) / T)

peak_indices = np.arange(0, 11, 2)
x_peaks = x_start + peak_indices * T
y_peaks = np.ones_like(x_peaks)  # Peaks are always at y = 1


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$\cos\left(\frac{\pi(x - x_{start})}{T}\right)$', color='b', linewidth=2)
plt.scatter(x_peaks, y_peaks, color='red', marker='x',
            s=60, zorder=3, label='Peaks')

# Formatting the plot
plt.title('Plot of $\cos(\pi \cdot (x - x_{start}) / T)$', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.xlim(x_start-0.01, x_end+0.01)
plt.ylim(-1.1, 1.1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=11)

# Display the chart
plt.show()
