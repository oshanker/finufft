
"""
Complete regrid for nonuniform 1D data
Input: input.csv with x,c in [243.5339..., 14859.4767...]
Output: uniform_120k.csv with 120k uniform points

Works with:
- FINUFFT if installed (pip install finufft) -> spectrally accurate
- Fallback to cubic interpolation if FINUFFT not installed -> also accurate for your data

Fixes the 14k bug: correct normalization is N/(2pi) * ifft, with density weights.
"""
import numpy as np
import sys

INPUT_CSV = "../out/input.csv"  # or "input.csv" locally
OUTPUT_CSV = "../out/uniform_120k.csv"
X_MIN = 243.53396107975615984
X_MAX = 14859.4767721
L = X_MAX - X_MIN  # 14615.942811020244
N = 120000
EPS = 1e-9
USE_DENSITY_WEIGHTS = True
ENDPOINT = False  # False = standard FFT grid, True = include X_MAX exactly

def load_csv(path):
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except:
        data = np.loadtxt(path, delimiter=',')
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"ERROR: Expected 2 columns x,c, got shape {data.shape}")
        sys.exit(1)
    return data[:,0], data[:,1]

print(f"Loading {INPUT_CSV}...")
x_phys, c_real = load_csv(INPUT_CSV)
print(f"Original M={len(x_phys)} x=[{x_phys.min()}, {x_phys.max()}]")

# Filter to declared range and sort
mask = (x_phys >= X_MIN-1e-6) & (x_phys <= X_MAX+1e-6)
x_phys = x_phys[mask]
c_real = c_real[mask]
idx_sort = np.argsort(x_phys)
x_phys = x_phys[idx_sort]
c_real = c_real[idx_sort]
print(f"Filtered & sorted M={len(x_phys)}")

# --- Try FINUFFT path ---
try:
    import finufft
    print("FINUFFT found, using Type-1 NUFFT with correct normalization...")
    
    # 1. Scale to [-pi, pi)
    x_scaled = (x_phys - X_MIN) * (2*np.pi / L) - np.pi
    x_scaled = x_scaled.astype(np.float64)

    # 2. Density weights
    if USE_DENSITY_WEIGHTS:
        sx = x_phys
        w = np.empty_like(sx)
        w[1:-1] = (sx[2:] - sx[:-2]) / 2.0
        w[0] = sx[1] - sx[0]
        w[-1] = sx[-1] - sx[-2]
        # w_phys sum = L
        w_scaled = w * (2*np.pi / L)  # sum = 2pi
        c_weighted = (c_real * w_scaled).astype(np.complex128)
        print(f"Weights: mean_phys={w.mean():.6f} (expected {L/len(w):.6f})")
    else:
        # uniform weight approximation
        w_scaled = np.full_like(x_phys, 2*np.pi / len(x_phys))
        c_weighted = (c_real * w_scaled).astype(np.complex128)

    # 3. Type 1: f_k = sum c_j*w_j*exp(i k x_j) ~ integral
    f_centered = finufft.nufft1d1(x_scaled, c_weighted, n_modes=N, eps=EPS, isign=1)
    
    # 4. Correct inverse: c(X) = 1/(2pi) * sum_k f_k exp(-ikX)
    # ifft(f) = (1/N) sum_k f_k exp(ikX), so sum = N*ifft
    # => c = N/(2pi) * ifft(f)
    f_fft = np.fft.ifftshift(f_centered)  # to FFT order
    u_uniform = np.fft.ifft(f_fft) * N / (2*np.pi)
    u_uniform = u_uniform.real

    X_uniform = np.linspace(X_MIN, X_MAX, N, endpoint=ENDPOINT)
    print("FINUFFT path done.")

except ImportError:
    print("FINUFFT not installed, falling back to cubic interpolation (scipy)...")
    print("To use FINUFFT locally: pip install finufft")
    from scipy.interpolate import interp1d
    f_interp = interp1d(x_phys, c_real, kind='cubic', bounds_error=False, fill_value='extrapolate')
    X_uniform = np.linspace(X_MIN, X_MAX, N, endpoint=ENDPOINT)
    u_uniform = f_interp(X_uniform)

# Save
out = np.column_stack((X_uniform, u_uniform))
np.savetxt(OUTPUT_CSV, out, delimiter=',', header='x_uniform,c_uniform', comments='')
print(f"Saved {N} points to {OUTPUT_CSV}")
print(f"dx = {L/N if not ENDPOINT else L/(N-1):.12f}")
print(f"First 3 rows:")
print(out[:3])
print(f"Last 3 rows:")
print(out[-3:])
print(f"c range: min={u_uniform.min():.6f} max={u_uniform.max():.6f} (should be ~[-1,1] for your data)")
