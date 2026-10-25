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

