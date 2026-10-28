"""
Plain FINUFFT regrid - drop extreme values, no Akima
- Drops points with extreme c values (statistical outlier removal, no fixed threshold)
- Does plain FINUFFT Type1 estimate with Tukey taper

Input: 6KE12posmax.csv (id,x,c) or input.csv (x,c)
Output: uniform_120k_finufft_plain.csv

Requires: pip install finufft scipy
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
SPIKE_THRESHOLD = 50.0
TUKEY_ALPHA = 0.01  # 5% taper at each edge to kill edge discontinuity
SAFE = 0.15

# How to drop extremes - statistical, no fixed SPIKE_THRESHOLD
EXTREME_METHOD = "value"  # iqr is most robust for your data with spikes
IQR_FACTOR = 3.0
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
print(f"Original: M={len(x_phys)} x=[{x_phys[0]:.2f},{x_phys[-1]:.2f}] \
  c min max =[{c_real.min():.2f},{c_real.max():.2f}] mean={c_real.mean():.3f} \
  std= {c_real.std():.3f}")

# --- Drop extreme values (no fixed threshold) ---
if EXTREME_METHOD == "iqr":
    q1, q3 = np.percentile(c_real, [25, 75])
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
elif EXTREME_METHOD == "value":
    lo, hi = -SPIKE_THRESHOLD, SPIKE_THRESHOLD
    mask = (c_real >= lo) & (c_real <= hi)
    print(f"value method: keep [LOW, HIGH] = [{lo:.3f},{hi:.3f}]")
else:
    mask = np.ones(len(c_real), dtype=bool)

x_filt = x_phys[mask]
c_filt = c_real[mask]
print(f"After dropping extremes: M={len(x_filt)} (dropped {len(x_phys)-len(x_filt)} points)")
print(f"Filtered c=[{c_filt.min():.3f},{c_filt.max():.3f}]")

X_uniform = np.linspace(X_MIN, X_MAX, N, endpoint=False)

# --- Plain FINUFFT estimate with Tukey taper ---
print(f"Running plain FINUFFT Type1 with Tukey taper... \
  TUKEY_ALPHA {TUKEY_ALPHA} SAFE {SAFE}")
try:
    import finufft
    from scipy.signal.windows import tukey
except ImportError as e:
    print(f"ERROR: {e}. Install with: pip install finufft scipy")
    print("This script has no Akima fallback by design (as requested).")
    sys.exit(1)

# Density weights for nonuniform sampling
w = np.empty_like(x_filt)
w[1:-1] = (x_filt[2:] - x_filt[:-2]) / 2.0
w[0] = x_filt[1] - x_filt[0]
w[-1] = x_filt[-1] - x_filt[-2]
w_scaled = w * (2*np.pi / L)

x_scaled = (x_filt - X_MIN) * (2*np.pi / L) - np.pi

# Tukey window to kill edge discontinuity (periodic assumption)
tukey_win = tukey(len(x_filt), alpha=TUKEY_ALPHA)
c_weighted = (c_filt * w_scaled * tukey_win).astype(np.complex128)

# FINUFFT Type1
f_centered = finufft.nufft1d1(x_scaled.astype(np.float64), c_weighted, n_modes=N, eps=EPS, isign=1)

# Inverse with window compensation
f_fft = np.fft.ifftshift(f_centered)
u_raw = (np.fft.ifft(f_fft) * N / (2*np.pi)).real

# Undo Tukey: where window > 0.1, divide to recover amplitude
#            where window <= 0.1, keep tapered value (smooth to 0) to avoid blow-up
uniform_win = tukey(N, alpha=TUKEY_ALPHA)
safe = uniform_win > SAFE
u_uniform = np.zeros_like(u_raw)
u_uniform[safe] = u_raw[safe] / uniform_win[safe]  # center 90%: recover
u_uniform[~safe] = u_raw[~safe]  # edges 5% each side: keep tapered, no blow-up

print(f"FINUFFT done: u min max=[{u_uniform.min():.3f},{u_uniform.max():.3f}] \
        spikes={(np.abs(u_uniform)>SPIKE_THRESHOLD).sum()} \
        mean={u_uniform.mean():.3f} \
  std= {u_uniform.std():.3f}" \
      )

# Save
np.savetxt(OUTPUT_CSV, np.column_stack((X_uniform, u_uniform)), \
          fmt="%.6f", delimiter=',', header='x_uniform,c_uniform', comments='')
print(f"Saved {OUTPUT_CSV} N={N} dx={L/N:.6f}")
