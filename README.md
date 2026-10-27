# few isolated regions with large variations

That's where both methods break — and it's exactly why you got 14616 before with the wrong normalization. With isolated large variations, it's worse.

If you have a few isolated regions with spikes / jumps / high frequency bursts in an otherwise smooth signal like yours, here's what happens:

1. With FINUFFT (global Fourier method) — what you were trying
FINUFFT is global. It assumes your data is periodic and smooth.

If you have 1% of the domain with a jump from 0 -> 1000 -> 0:

Gibbs ringing: That spike has energy at all Fourier modes. When you do Type1 -> ifft to a uniform grid, the reconstruction will ring everywhere, not just near the spike. You'll see ∼9% overshoot that decays as 1/|x - x_spike|.
Your uniform output will look like: clean sine wave plus a low-amplitude sine wave that wiggles across the whole 243 -> 14859 range.
With density weights: It gets worse if the spike region is also densely sampled — the weights w_j become small there, but c_j*w_j is still large, so f_k is dominated by the spike.
For N=120k, a single isolated spike of amplitude A at x0 gives f_k ≈ A * w * e^{ikx0} — flat spectrum. Then u_uniform[X] ≈ A * Dirichlet_kernel(X-x0) — which is A * sin(N*(X-x0)/2) / sin((X-x0)/2) — ringing everywhere.

2. With cubic interpolation (what I used for your fallback)
Interpolation is local, so it's better:

Near the spike, cubic will overshoot. For your data [-1,1], a jump to 1000 will make cubic interpolate to ∼1100 or -100 just before/after.
Away from the spike (>5-10 points), it stays perfect.
What to do if you really have those regions
Don't use one global NUFFT. Split:

Python
import numpy as np
from scipy.interpolate import Akima1DInterpolator
Detect large variation regions
Gradient of c

22 lines hidden
Akima instead of cubic is key — it doesn't overshoot at isolated jumps. That's what you want for input.csv if it has bursts.

For FINUFFT if you must stay spectral:

Window/taper the spike: c_tapered = c * tukey_window around spike
Increase N: Need N > 2 * (L / width_of_spike) to resolve spike — for a spike width 0.1 in your L=14615 domain, you'd need N > 292k, not 120k
Iterative reconstruction: Use finufft inside a conjugate-gradient solve: solve A f = c where A = Type2, instead of direct Type1. This is what pynufft calls density compensation iteration.

# finufft


To convert your nonuniformly sampled data into a perfectly aligned regular spatial grid, we moved away from simple approximations and implemented a rigorous least-squares inversion framework.
Here is the step-by-step summary of the approach that successfully resolved the alignment and scaling issues:

1. Formulating the Inverse Problem

Instead of trying to interpolate the irregular points directly, we treated the Nonuniform FFT (NUFFT) as a forward linear system matrix equation:

\(\mathbf{Ac}=\mathbf{y}\)

• \(\mathbf{y}\): Your known, irregularly spaced data samples.

• \(\mathbf{A}\): The Type 2 NUFFT operator, which acts as a matrix mapping uniform Fourier frequencies to irregular spatial locations.

• \(\mathbf{c}\): The unknown underlying Fourier modes we need to find.

2. Setting Up Exact Mathematical Duality

Iterative matrix solvers require a mathematically flawless relationship between the forward operation (\(\mathbf{A}\)) and its conjugate transpose adjoint (\(\mathbf{A}^{H}\)).
• Forward Pass (\(\mathbf{A}\)): Modeled using finufft.nufft1d2 (Uniform Grid Modes \(\rightarrow \) Nonuniform Points).
• Adjoint Pass (\(\mathbf{A}^{H}\)): Modeled using finufft.nufft1d1 (Nonuniform Points \(\rightarrow \) Uniform Grid Modes).
• Crucial Fix: Leaving FINUFFT’s native scaling intact without arbitrary modifiers guaranteed that the solver could mathematically converge rather than flattening out into a zero line.

3. Iterative Optimization via LSQR

Because the explicit system matrix \(\mathbf{A}\) is too massive to compute directly, we wrapped our forward and adjoint passes inside a SciPy LinearOperator. We passed this to the LSQR algorithm, which iteratively adjusted the Fourier coefficients (\(\mathbf{c}\)) until the difference between the reconstructed model and your original nonuniform points (\(\Vert{}\mathbf{A}\mathbf{c} - \mathbf{y}\Vert{}^2\)) was minimized to a near-zero tolerance (\(10^{-10}\)).

4. Foolproof Spatial Evaluation

To completely bypass the common formatting errors, phase scrambles, and indexing mismatches associated with shifting frequencies into standard NumPy fft utilities, we stayed entirely within the NUFFT ecosystem for the final step:
• We generated a target list of perfectly uniform spatial coordinates.
• We ran the optimized Fourier coefficients (\(\mathbf{c}\)) through a final Type 2 NUFFT targeted precisely at those uniform coordinates.
This directly yielded the correct, continuous spatial values perfectly matching the amplitude and path of your underlying curve.



Instead of using np.fft.ifft and risking indexing errors, use Type 2 NUFFT (nufft1d2) to generate the final uniform spatial grid. Since Type 2 evaluates Fourier modes at any target coordinates, passing a regular grid of coordinates to nufft1d2 produces the exact spatial curve directly, completely bypassing ifft and any shifting utilities.

