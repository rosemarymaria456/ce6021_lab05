# Lab Report — Lab 05: Harris Corner Detector

*Answer each question briefly and factually — a number, a name, or a short phrase is enough.
Each question is marked on whether the required information is present and correct, not on how
it's written, so there is no benefit to writing full sentences or extra explanation.*

**Q1.** In one phrase, what does `det_coords()` compute?

The row, col coordinates of the detected corners.

**Q2.** Which library function did you use in Step 2 (`compute_gradients`) to convolve the image
with the Sobel kernels?

signal.convolve2d

**Q3.** In Step 3 (`gradient_products`), what are the three products computed from `Ix` and `Iy`?

Ix², Iy², and Ix * Iy

**Q4.** What does Step 4 (`smooth_products`) use to weight each gradient product before summing
it over a local window?

A 2D Gaussian window 

**Q5.** State the formula used in Step 5 (`det_and_trace`) to compute the determinant from `A`,
`B` and `C`.

Det(M) = A * B - C²

**Q6.** State the formula used in Step 6 (`harris_response`) to compute the response `R` from the
determinant and trace.

R = Det(M) - alpha * Trace(M)²

**Q7.** What is the default value of `self.alpha` in the constructor?

0.05

**Q8.** What is the default value of `self.threshold_val` in the constructor?

0.001

**Q9.** Which `skimage` function does Step 8 (`find_peaks`) use to collapse a cluster of
high-response pixels down to a single coordinate?

peak_local_max

**Q10.** In Step 9 (`det_coords`), what coordinate order — `(row, col)` or `(col, row)` — does the
returned array use?

row,col