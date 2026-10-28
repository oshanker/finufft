"""
Plain FINUFFT regrid - drop extreme values, no Akima
- Drops points with extreme c values (statistical outlier removal, no fixed threshold)
- Does plain FINUFFT Type1 estimate

Input: 6KE12posmax.csv (id,x,c) or input.csv (x,c)
Output: uniform_120k_finufft_plain.csv

Requires: pip install finufft
"""

import numpy as np
import sys

INPUT_CSV = "../out/6KE12posmax.csv"
OUTPUT_CSV = "../out/uniform_120k_finufft_plain.csv"
X_MIN = 243.53396107975615984
X_MAX = 14859.4767721
L = X_MAX - X_MIN
N = 120000
EPS = 1e-9

# How to drop extremes - statistical, no fixed SPIKE_THRESHOLD
# Options: "iqr", "zscore", "percentile"
EXTREME_METHOD = "iqr"  # iqr is most robust for your data with spikes
IQR_FACTOR = 0.5  # drop points outside Q1 - 3*IQR, Q3 + 3*IQR

ZSCORE_THRESH = 4.0
PERCENTILE_LOW = 0.1
PERCENTILE_HIGH = 99.9

def load_csv_smart(path):
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except:
        data = np.loadtxt(path, delimiter=',')
    if data.shape[1] == 2:
        x, c = data[:,0], data[:,1]
    elif data.shape[1] >= 3:
        x, c = data[:,1], data[:,2]
        print(f"3-col file detected, using col1=x, col2=c")
    else:
        print(f"ERROR: Need at least 2 cols")
        sys.exit(1)
    return x, c

print(f"Loading {INPUT_CSV}...")
x_phys, c_real = load_csv_smart(INPUT_CSV)
idx = np.argsort(x_phys)
x_phys = x_phys[idx]
c_real = c_real[idx]
print(f"Original: M={len(x_phys)} x=[{x_phys[0]:.2f},{x_phys[-1]:.2f}] c=[{c_real.min():.2f},{c_real.max():.2f}] mean={c_real.mean():.3f} std={c_real.std():.3f}")

# --- Drop extreme values (no fixed threshold) ---
if EXTREME_METHOD == "iqr":
    q1, q3 = np.percentile(c_real, [0.9, 99.1])
    iqr = q3 - q1
    lo = q1 - IQR_FACTOR * iqr
    hi = q3 + IQR_FACTOR * iqr
    mask = (c_real >= lo) & (c_real <= hi)
    print(f"IQR method: Q1={q1:.3f} Q3={q3:.3f} IQR={iqr:.3f} keep=[{lo:.3f},{hi:.3f}]")
elif EXTREME_METHOD == "zscore":
    mean, std = c_real.mean(), c_real.std()
    lo = mean - ZSCORE_THRESH * std
    hi = mean + ZSCORE_THRESH * std
    mask = (c_real >= lo) & (c_real <= hi)
    print(f"Z-score method: mean={mean:.3f} std={std:.3f} keep=[{lo:.3f},{hi:.3f}]")
elif EXTREME_METHOD == "percentile":
    lo, hi = np.percentile(c_real, [PERCENTILE_LOW, PERCENTILE_HIGH])
    mask = (c_real >= lo) & (c_real <= hi)
    print(f"Percentile method: keep [{PERCENTILE_LOW}%, {PERCENTILE_HIGH}%] = [{lo:.3f},{hi:.3f}]")
else:
    mask = np.ones(len(c_real), dtype=bool)

x_filt = x_phys[mask]
c_filt = c_real[mask]
print(f"After dropping extremes: M={len(x_filt)} (dropped {len(x_phys)-len(x_filt)} points)")
print(f"Filtered c=[{c_filt.min():.3f},{c_filt.max():.3f}]")

X_uniform = np.linspace(X_MIN, X_MAX, N, endpoint=False)

# --- Plain FINUFFT estimate ---
print("Running plain FINUFFT Type1...")
try:
    import finufft
except ImportError:
    print("ERROR: finufft not installed. Install with: pip install finufft")
    print("This script has no Akima fallback by design (as requested).")
    sys.exit(1)

# Density weights for nonuniform sampling
# w_j = local gap, sum(w) ~ L
w = np.empty_like(x_filt)
w[1:-1] = (x_filt[2:] - x_filt[:-2]) / 2.0
w[0] = x_filt[1] - x_filt[0]
w[-1] = x_filt[-1] - x_filt[-2]
w_scaled = w * (2*np.pi / L)  # scale to [-pi,pi) domain, sum ~2pi

x_scaled = (x_filt - X_MIN) * (2*np.pi / L) - np.pi
c_weighted = (c_filt * w_scaled).astype(np.complex128)

# Plain FINUFFT Type1: f_k = sum c_j*w_j*exp(i k x_j)
f_centered = finufft.nufft1d1(x_scaled.astype(np.float64), c_weighted, n_modes=N, eps=EPS, isign=1)

# Plain inverse: c(X) = N/(2pi) * ifft(f)  [correct normalization, no Tukey]
f_fft = np.fft.ifftshift(f_centered)
u_uniform = (np.fft.ifft(f_fft) * N / (2*np.pi)).real
SPIKE_THRESHOLD = 15.0

print(f"FINUFFT done: u min max=[{u_uniform.min():.3f},{u_uniform.max():.3f}] \
        spikes={(np.abs(u_uniform)>SPIKE_THRESHOLD).sum()}" \
      )

# Save
np.savetxt(OUTPUT_CSV, np.column_stack((X_uniform, u_uniform)), delimiter=',', header='x_uniform,c_uniform', comments='')
print(f"Saved {OUTPUT_CSV} N={N} dx={L/N:.6f}")
