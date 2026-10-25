# finufft

Instead of using np.fft.ifft and risking indexing errors, use Type 2 NUFFT (nufft1d2) to generate the final uniform spatial grid. Since Type 2 evaluates Fourier modes at any target coordinates, passing a regular grid of coordinates to nufft1d2 produces the exact spatial curve directly, completely bypassing ifft and any shifting utilities.
Here 
