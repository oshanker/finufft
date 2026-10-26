import numpy as np
import matplotlib.pyplot as plt

# Define the constants given in the prompt
gincr = 0.2435990469039924426798
x_start = 243.53396107975615984
x_end = x_start + 60000 * gincr

# Generate 1,000 evenly spaced points over the specified range
x = np.linspace(x_start + gincr/3, x_end - gincr/3, 180000)

# Evaluate your exact formula: cos(pi * (x - x_start) / T)
y = np.cos(np.pi * (x - x_start) / gincr)


combined_data = np.column_stack((x, y))

# 4. Save to file with clean decimal formatting
np.savetxt(
    "../out/input.csv", 
    combined_data, 
    fmt= "%.6f",  # 4 decimals for Time, 6 decimals for Value
    header="x,c", 
    delimiter=",",
    comments=""            )
