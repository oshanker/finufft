"""
Plain FINUFFT regrid with zero-padding instead of Tukey
- Drops extreme values statistically (IQR, no fixed threshold)
- Handles periodicity by padding 10 intervals with zeros on each side
- No Tukey, no Akima

X_MAX - X_MIN = 120000 * dx_fixed (120000 intervals) - CORRECTED
Padding: 10*dx on each side with zeros

Input: 6KE12posmax.csv (id,x,c)
Output: uniform_120k_padded.csv (interior) + uniform_120020_padded_full.csv (with padding)

Requires: pip install finufft
"""

import numpy as np
import sys

INPUT_CSV = "../out/input.csv"
OUTPUT_CSV = "../out/uniform_120k_padded.csv"
OUTPUT_CSV_PADDED = "../out/uniform_120020_padded_full.csv"

X_MIN_ORIG = 243.53396107975615984
X_MAX_ORIG = 14859.4767721
L_ORIG = X_MAX_ORIG - X_MIN_ORIG

# --- Requirement: X_MAX - X_MIN = 120000 * fixed_interval ---
N = 120000  # 120000 intervals as requested (corrected)
DX_FIXED = L_ORIG / N  # = 0.1217995234251687
X_MIN = X_MIN_ORIG
X_MAX = X_MAX_ORIG
L = X_MAX - X_MIN  # = N * DX_FIXED = 14615.94

PAD_INTERVALS = 1  # 10 intervals each side, padded with zeros
EPS = 1e-9
EXTREME_METHOD = "iqr"
IQR_FACTOR = 3.0

def load_csv_smart(path):
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except:
        data = np.loadtxt(path, delimiter=',')
    if data.shape[1] == 2:
        x, c = data[:,0], data[:,1]
    else:
        x, c = data[:,1], data[:,2]
    return x, c

def test_data():
    path =   "../out/test_output.csv"

    _, y_test = load_csv_smart(path)

    print(f"y_test First  rows:")
    print(y_test[:4])
    print(f"y_test Last  rows:")
    print(y_test[-4:])

print(f"Loading {INPUT_CSV}...")
x_phys, c_real = load_csv_smart(INPUT_CSV)
test_data()

print(f"Original: M={len(x_phys)} c=[{c_real.min():.2f},{c_real.max():.2f}]")

# Drop extremes via IQR (no fixed threshold)
q1, q3 = np.percentile(c_real, [25, 75])
iqr = q3 - q1
lo = q1 - IQR_FACTOR * iqr
hi = q3 + IQR_FACTOR * iqr
mask = (c_real >= lo) & (c_real <= hi)
x_filt = x_phys[mask]
c_filt = c_real[mask]
print(f"IQR keep=[{lo:.3f},{hi:.3f}] -> M={len(x_filt)} dropped {len(x_phys)-len(x_filt)}")

# --- Zero-padding setup ---
DX = L / N
print(f"Fixed interval DX = {DX:.12f}")
print(f"X_MAX - X_MIN = {L:.3f} = {N} * DX (120000 * dx)")

PAD = PAD_INTERVALS
N_PADDED = N + 2*PAD  # 120000 + 20 = 120020
L_PADDED = L + 2*PAD*DX
X_MIN_PADDED = X_MIN - PAD*DX
X_MAX_PADDED = X_MAX + PAD*DX

print(f"Padded: {PAD} intervals each side with zeros")
print(f"Padded domain: [{X_MIN_PADDED:.3f}, {X_MAX_PADDED:.3f}) L_padded={L_PADDED:.3f} N_padded={N_PADDED}")

endpoint=True
X_uniform_padded = np.linspace(X_MIN_PADDED, X_MAX_PADDED, N_PADDED, endpoint=endpoint)
X_uniform = np.linspace(X_MIN, X_MAX, N, endpoint=endpoint)

# --- FINUFFT on padded domain (handles periodicity) ---
print("Running FINUFFT Type1 on padded domain (no Tukey)...")
try:
    import finufft
except ImportError:
    print("ERROR: finufft not installed - pip install finufft")
    print("No Akima fallback by design")
    sys.exit(1)

# Density weights
w = np.empty_like(x_filt)
w[1:-1] = (x_filt[2:] - x_filt[:-2]) / 2.0
w[0] = x_filt[1] - x_filt[0]
w[-1] = x_filt[-1] - x_filt[-2]
w_scaled = w * (2*np.pi / L_PADDED)

x_scaled = (x_filt - X_MIN_PADDED) * (2*np.pi / L_PADDED) - np.pi
c_weighted = (c_filt * w_scaled).astype(np.complex128)

f_centered = finufft.nufft1d1(x_scaled.astype(np.float64), c_weighted, n_modes=N_PADDED, eps=EPS, isign=1)
f_fft = np.fft.ifftshift(f_centered)
u_padded = (np.fft.ifft(f_fft) * N_PADDED / (2*np.pi)).real

print(f"FINUFFT padded done: u_padded=[{u_padded.min():.3f},{u_padded.max():.3f}]")

# Extract interior (remove padding)
if PAD == 0:
   u_interior = u_padded  # 120000 points from X_MIN to X_MAX
else:
   u_interior = u_padded[PAD:-PAD]  # 120000 points from X_MIN to X_MAX

print(f"Interior (after removing padding): u=[{u_interior.min():.3f},{u_interior.max():.3f}]")

# Save both
np.savetxt(OUTPUT_CSV_PADDED, np.column_stack((X_uniform_padded, u_padded)), \
         fmt= "%.6f",   delimiter=',', header='x_uniform_padded,c_uniform', comments='')
print(f"Saved padded full (with zeros): {OUTPUT_CSV_PADDED} N={N_PADDED}")

np.savetxt(OUTPUT_CSV, np.column_stack((X_uniform, u_interior)), \
           fmt= "%.6f", delimiter=',', header='x_uniform,c_uniform', comments='') 

print(f"u_padded First  rows:")
print(u_padded[:4])
print(f"u_padded Last  rows:")
print(u_padded[-4:])
print(f"Saved interior: {OUTPUT_CSV} N={N} dx={DX:.12f}")
print(f"Check: X_MAX - X_MIN = {X_MAX-X_MIN:.6f} = {N} * {DX:.6f}")
