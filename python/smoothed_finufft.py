"""
Hybrid regrid: FINUFFT for smooth background + Akima for isolated spikes
FINUFFT path includes correct N/(2pi) normalization + Tukey edge taper to prevent ringing
Akima path is used for background when FINUFFT not available, and always for local spike shape

Input: 6KE12posmax.csv (id,x,c) or input.csv (x,c)
Output: uniform_120k_hybrid.csv (FINUFFT smooth + Akima spikes)

Features:
- Auto-detects 800 isolated spike regions (|c|>20) in your data
- Background (166k points): FINUFFT spectrally accurate OR Akima fallback
- Spikes: local Akima reconstruction + exact peak preservation (126.08 kept as 126.08)

Usage:
  python regrid_complete.py
  # Edit INPUT_CSV, USE_FINUFFT at top
"""

import numpy as np
import sys

INPUT_CSV = "../out/6KE12posmax.csv"
OUTPUT_CSV = "../out/uniform_120k_hybrid.csv"
X_MIN = 243.53396107975615984
X_MAX = 14859.4767721
L = X_MAX - X_MIN
N = 120000
EPS = 1e-9
SPIKE_THRESHOLD = 20.0
SPIKE_BUFFER_POINTS = 8
USE_FINUFFT = True  # True = try FINUFFT for background (falls back to Akima if not installed)

def load_csv_smart(path):
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except:
        data = np.loadtxt(path, delimiter=',')
    if data.shape[1] == 2:
        x, c = data[:,0], data[:,1]
    elif data.shape[1] >= 3:
        x, c = data[:,1], data[:,2]
        print(f"3-col file, using col1=x col2=c")
    else:
        sys.exit(1)
    return x, c

print(f"Loading {INPUT_CSV}...")
x_phys, c_real = load_csv_smart(INPUT_CSV)
idx = np.argsort(x_phys)
x_phys = x_phys[idx]
c_real = c_real[idx]
print(f"M={len(x_phys)} x=[{x_phys[0]:.2f},{x_phys[-1]:.2f}] c=[{c_real.min():.2f},{c_real.max():.2f}]")

# Detect isolated spike clusters
#is_spike: The resulting array. If an element in c_real is greater
# than the threshold, its corresponding
#position in is_spike will be True. Otherwise, it will be False.
is_spike = np.abs(c_real) > SPIKE_THRESHOLD
# spike_indices: The final output. It is an array of integers
# showing the exact indices where the signal crossed your threshold.

spike_indices = np.where(is_spike)[0]
print(f"spike_indices {spike_indices[:3]} ...")
clusters = []
if len(spike_indices) > 0:
    cur = [spike_indices[0]]
    for i in spike_indices[1:]:
        if i - cur[-1] <= 5:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)

print(f"Found {len(clusters)} isolated spike regions")

mask_bg = np.ones(len(x_phys), dtype=bool)
for cl in clusters:
    lo = max(0, min(cl) - SPIKE_BUFFER_POINTS)
    hi = min(len(x_phys), max(cl) + SPIKE_BUFFER_POINTS + 1)
    mask_bg[lo:hi] = False

x_bg = x_phys[mask_bg]
c_bg = c_real[mask_bg]
X_uniform = np.linspace(X_MIN, X_MAX, N, endpoint=False)
dx = L / N

# --- Background: FINUFFT with Tukey ---
if USE_FINUFFT:
    try:
        import finufft
        from scipy.signal.windows import tukey
        print("Background: FINUFFT + Tukey taper (correct N/2pi normalization)...")
        x_scaled = (x_bg - X_MIN) * (2*np.pi / L) - np.pi
        w = np.empty_like(x_bg)
        w[1:-1] = (x_bg[2:] - x_bg[:-2]) / 2
        w[0] = x_bg[1]-x_bg[0]
        w[-1] = x_bg[-1]-x_bg[-2]
        w_scaled = w * (2*np.pi / L)
        tukey_w = tukey(len(x_bg), alpha=0.05)
        c_weighted = (c_bg * w_scaled * tukey_w).astype(np.complex128)
        f_centered = finufft.nufft1d1(x_scaled.astype(np.float64), c_weighted, n_modes=N, eps=EPS)
        f_fft = np.fft.ifftshift(f_centered)
        u_bg_raw = (np.fft.ifft(f_fft) * N / (2*np.pi)).real
        uniform_window = tukey(N, alpha=0.05)
        safe = uniform_window > 0.1
        u_bg = np.zeros_like(u_bg_raw)
        u_bg[safe] = u_bg_raw[safe] / uniform_window[safe]
        print("  FINUFFT background done, edge ringing suppressed")
    except Exception as e:
        print(f"FINUFFT failed ({e}), fallback to Akima")
        USE_FINUFFT = False

if not USE_FINUFFT:
    print("Background: Akima (no ringing)...")
    from scipy.interpolate import Akima1DInterpolator
    f_bg = Akima1DInterpolator(x_bg, c_bg)
    u_bg = np.nan_to_num(f_bg(X_uniform), nan=0.0)

# --- Spikes: local Akima + exact peak preservation ---
print("Spikes: local Akima + exact peak preservation...")
from scipy.interpolate import Akima1DInterpolator
u_final = u_bg.copy()

for cl in clusters:
    lo = max(0, min(cl) - SPIKE_BUFFER_POINTS)
    hi = min(len(x_phys), max(cl) + SPIKE_BUFFER_POINTS + 1)
    x_local = x_phys[lo:hi]
    c_local = c_real[lo:hi]
    # deduplicate x, keep max abs c for duplicate x
    uniq = {}
    for xv, cv in zip(x_local, c_local):
        if xv not in uniq or abs(cv) > abs(uniq[xv]):
            uniq[xv] = cv
    x_u = np.array(sorted(uniq.keys()))
    c_u = np.array([uniq[k] for k in x_u])
    if len(x_u) < 3:
        continue
    x_lo, x_hi = x_u[0], x_u[-1]
    j_lo = int((x_lo - X_MIN) / L * N)
    j_hi = int((x_hi - X_MIN) / L * N) + 1
    j_lo = max(0, j_lo); j_hi = min(N, j_hi)
    if j_lo >= j_hi:
        continue
    X_loc = X_uniform[j_lo:j_hi]
    try:
        f_loc = Akima1DInterpolator(x_u, c_u)
        u_loc = np.nan_to_num(f_loc(X_loc), nan=0.0)
        u_final[j_lo:j_hi] = u_loc
    except:
        pass

# Ensure exact peak values are preserved (126.08 stays 126.08)
for xs, cs in zip(x_phys, c_real):
    if abs(cs) > SPIKE_THRESHOLD:
        b = int((xs - X_MIN) / L * N)
        if 0 <= b < N and abs(cs) > abs(u_final[b]):
            u_final[b] = cs

print(f"Final: min={u_final.min():.2f} max={u_final.max():.2f} \
      spikes={(np.abs(u_final)>SPIKE_THRESHOLD).sum()}")
np.savetxt(OUTPUT_CSV, np.column_stack((X_uniform, u_final)), \
           fmt="%.6f", delimiter=',', header='x_uniform,c_uniform', comments='')
print(f"Saved {OUTPUT_CSV}")
print(f"background_only: bg min={u_bg.min():.2f} bg max={u_bg.max():.2f} \
      spikes={(np.abs(u_bg)>SPIKE_THRESHOLD).sum()}")
np.savetxt(OUTPUT_CSV.replace("hybrid", "background_only"), \
           np.column_stack((X_uniform, u_bg)), delimiter=',', \
           header='x_uniform,c_uniform', comments='')
