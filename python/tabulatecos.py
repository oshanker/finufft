import numpy as np
import matplotlib.pyplot as plt

# Define the constants given in the prompt
gincr = 0.2435990469039924426798
x_start = 243.53396107975615984
x_end = x_start + 60000 * gincr
N = 120000
ENDPOINT = False  # False = standard FFT grid, True = include X_MAX exactly

X_uniform = np.linspace(x_start, x_end, N, endpoint=ENDPOINT)

# Generate 1,000 evenly spaced points over the specified range
x = np.linspace(x_start + gincr/3, x_end - gincr/3, 170000)
def true_signal(x):
    # Scales frequencies dynamically based on your chosen domain length
    ret = np.cos(np.pi * (x - x_start) / gincr) + \
           np.cos(np.pi * (x - x_start) / (2*gincr))
    return ret



# Evaluate your exact formula: cos(pi * (x - x_start) / T)
y = true_signal(x)


combined_data = np.column_stack((x, y))

# 4. Save to file with clean decimal formatting
np.savetxt(
    "../out/input.csv", 
    combined_data, 
    fmt= "%.6f",  # 4 decimals for Time, 6 decimals for Value
    header="x,c", 
    delimiter=",",
    comments=""            )

X_uniform = np.linspace(x_start, x_end, N, endpoint=ENDPOINT)
ytest = true_signal(X_uniform)


combined_data = np.column_stack((X_uniform, ytest))

# 4. Save to file with clean decimal formatting
np.savetxt(
    "../out/test_output.csv", 
    combined_data, 
    fmt= "%.18e",  # 4 decimals for Time, 6 decimals for Value
    header="x,c", 
    delimiter=",",
    comments=""            )
