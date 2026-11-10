# Lab 05 — Harris Corner Detector

## Introduction

<p align="center">
  <img src="https://github.com/tonyscan6003/etivities/blob/main/10x10_checkered_board.png?raw=true" alt="A 10 by 10 black and white checkerboard" width="320">
  <br><em>The checkerboard used in this lab — 9 × 9 = 81 internal corners.</em>
</p>

Where are the corners in this image? Corners are among the most useful local features in an image: their position is pinned down in both directions, so the same corner can be found again in a second photo of the same scene — the basis of image matching, stitching and stereo. An edge, by contrast, looks the same all along its length.

The **Harris Corner Detector** finds corners by examining how the image gradient behaves in a small window around each pixel. At a corner the intensity changes strongly in *every* direction; at an edge it changes in one direction only; in a flat region it hardly changes at all. These three cases are told apart by a single number per pixel, the **Harris response**, computed from the gradients. In this lab the detector is built from first principles — `cv2.cornerHarris` is not used.

The lab is split into nine small steps. Each step is one short method — usually a single NumPy or SciPy operation — and each one can be checked before the next is started.

## Dataset / Test Images

| Resource | Details |
|----------|---------|
| Checkerboard image | A 10 × 10 checkerboard, downloaded automatically from `CHECKERBOARD_IMAGE_URL` and cached in `Data/`. Cropped and rescaled to 512 × 512 pixels, converted to greyscale and normalised to the range 0–1. Its 81 internal corners lie on a regular grid, 51 pixels apart starting at (51, 51), so the detected corners can be checked against a known answer. |

## How This Lab Works

1. **One step** is implemented in `lab05.py`.
2. `python evaluate_lab05.py` is run. Every finished step is run, a figure for each one is added to `lab05_results.html`, and the script stops at the first step that has not yet been implemented.
3. `lab05_results.html` is opened and the new figure is checked before the next step is started.

`pytest test_lab05.py` can also be run at any point — the tests are grouped by step, so the step that needs attention is indicated by the failing tests.

> **Image coordinates:** NumPy arrays are indexed `image[row, col]`, with row 0 at the **top**. The corners are returned as `(row, col)` pairs. Points are passed to OpenCV drawing functions such as `cv2.circle` as `(x, y)` = `(col, row)` — the other way round.

## The Pipeline at a Glance

```
greyscale image
  → 1. gaussian_kernel      a 2D Gaussian weighting window, sized from sigma
  → 2. compute_gradients    Sobel gradients Ix, Iy
  → 3. gradient_products    Ix², Iy², Ix·Iy
  → 4. smooth_products      each product Gaussian-weighted over a local window: A, B, C
  → 5. det_and_trace        Det(M) and Trace(M), where M = [[A, C], [C, B]]
  → 6. harris_response      R = Det(M) − alpha · Trace(M)²
  → 7. threshold_response   weak responses set to zero
  → 8. find_peaks           one (row, col) per cluster of strong response
  → 9. det_coords           runs steps 2–8 in order
```

## Background

### Corners, edges and flat regions
An **edge** is a point where the intensity changes strongly in *one* direction (across the edge) but not the other (along it), so any point along an edge looks much like its neighbours. A **corner** changes strongly in *every* direction, so its location is fixed exactly. A **flat** region hardly changes in any direction.

### Image gradients
The gradient of an image measures how fast the intensity changes. It is computed by convolution with the **Sobel** kernels: `Ix` responds to changes from left to right (vertical edges) and `Iy` to changes from top to bottom (horizontal edges).

### The second-moment matrix
At each pixel, the gradients in a small window are summarised by the **second-moment matrix** `M = [[A, C], [C, B]]`, where `A`, `B` and `C` are the Gaussian-weighted sums of `Ix²`, `Iy²` and `Ix·Iy` (Steps 3–4). The two eigenvalues of `M` describe how much the intensity changes along the two main directions of the window:
- **both eigenvalues small** → a flat region,
- **one large, one small** → an edge,
- **both large** → a corner.

### Gaussian weighting
The Gaussian kernel of Step 1 is built in the same way as the 2D window in Lab 00B: a 1D Gaussian combined with itself by an outer product. It weights nearby pixels more heavily than distant ones, so `M` describes a local neighbourhood rather than a single noisy pixel. A larger `sigma` gives a larger window.

### The Harris response, without eigenvalues
Computing the eigenvalues at every pixel is avoided. Instead the **determinant** of `M` (`Det(M) = AB − C²`, the *product* of the eigenvalues) and its **trace** (`Trace(M) = A + B`, their *sum*) are combined into one response value:

```
R = Det(M) − alpha · Trace(M)²
```

`R` is large and positive at a corner (both eigenvalues large), negative at an edge (one eigenvalue much larger than the other), and close to zero in flat regions. `alpha` is an empirical constant, typically between 0.04 and 0.25.

### Thresholding and non-maximum suppression
A real corner produces a small *cluster* of pixels with a high response, not a single pixel. The weak responses are first removed by a threshold (Step 7). Each remaining cluster is then reduced to one point by `skimage.feature.peak_local_max`, which returns the local maxima that are at least `min_distance` pixels apart (Step 8). This is known as **non-maximum suppression**.

## Your Task

### Class: `HarrisCorner`

**Constructor:** `HarrisCorner(**kwargs)` — all settings have defaults, so `HarrisCorner()` works with no arguments. The class is instantiated in `evaluate_lab05.py`.

The constructor is **already written**. Every setting is stored on `self`, and these attributes (e.g. `self.alpha`) should be used in the methods rather than numbers typed in directly — this is checked by the tests, which change the settings.

| Setting | Default | Used in | Meaning |
|---------|---------|---------|---------|
| `self.sigma` | 2 | Steps 1, 4 | Standard deviation of the Gaussian weighting window. |
| `self.alpha` | 0.05 | Step 6 | Harris sensitivity constant in `R = Det(M) − alpha · Trace(M)²`. |
| `self.threshold_val` | 0.001 | Step 7 | Responses not above this value are set to 0. |
| `self.min_distance` | 5 | Step 8 | Minimum distance (px) between two detected corners. |

### Methods to Implement

| Step | Method | Input → Output |
|------|--------|----------------|
| 1 | `gaussian_kernel()` | — → `(k, k)` Gaussian window |
| 2 | `compute_gradients(image)` | greyscale image → `(Ix, Iy)` |
| 3 | `gradient_products(Ix, Iy)` | gradients → `(Ixx, Iyy, Ixy)` |
| 4 | `smooth_products(Ixx, Iyy, Ixy)` | products → `(A, B, C)` |
| 5 | `det_and_trace(A, B, C)` | `(A, B, C)` → `(det, trace)` |
| 6 | `harris_response(det, trace)` | `(det, trace)` → response map |
| 7 | `threshold_response(response)` | response map → thresholded response map |
| 8 | `find_peaks(thresh_response)` | thresholded response map → `(N, 2)` array of `(row, col)` |
| 9 | `det_coords(image)` | greyscale image → `(N, 2)` array of `(row, col)` |

## Step-by-Step Walkthrough

Each step below gives the goal, the details that the result depends on, and a hint — a function to look up, or an earlier lab where the same thing was done. The code itself is to be written from these.

### Step 1 — `gaussian_kernel()`

**Goal:** a 2D Gaussian window sized from `self.sigma`.

- **Size:** `k = int(6σ + 1)`, with `half = (k − 1) // 2`. The 1D positions run from `−half` to `+half`.
- **1D Gaussian:** `g(x) = 1 / (σ√(2π)) · exp(−x² / (2σ²))`
- **2D window:** the outer product of `g` with itself.

**Hint:** `gaussian_kernel_2d` in Lab 00B follows the same pattern, using `np.arange` and `np.outer`. There are two differences here: the size rule above is used, and the kernel is **not** divided by its sum.

**Check:** a small, round blob, brightest at the centre, is shown in the Step 1 figure — a 13 × 13 kernel for `sigma = 2`.

### Step 2 — `compute_gradients(image)`

**Goal:** the gradient of the image in x and y.

The image is convolved with these two Sobel kernels, which are to be used exactly as shown:

```
kernel_x            kernel_y
[ -1   0   1 ]      [  1   2   1 ]
[ -2   0   2 ]      [  0   0   0 ]
[ -1   0   1 ]      [ -1  -2  -1 ]
```

**Hint:** `signal.convolve2d` is used, as in `convolve` in Lab 00B, with the same settings: `mode='same'` keeps the output the same size as the image, and `boundary='symm'` mirrors the image at its border.

**Check:** the vertical edges of the checkerboard are highlighted in `Ix` and the horizontal edges in `Iy`.

### Step 3 — `gradient_products(Ix, Iy)`

**Goal:** the three element-wise products needed for the second-moment matrix.

The products are returned in the order `Ix²`, `Iy²`, `Ix·Iy`.

**Hint:** the `*` operator multiplies NumPy arrays element by element, which is what is needed here. `@` and `np.dot` are matrix products and would give the wrong result.

**Check:** in the Step 3 figure `Ix²` and `Iy²` follow the edges, while `Ix·Iy` is non-zero only where both gradients are present — at the corners.

### Step 4 — `smooth_products(Ixx, Iyy, Ixy)`

**Goal:** each product Gaussian-weighted over a local window.

`A`, `B` and `C` are `Ixx`, `Iyy` and `Ixy`, each convolved with the Gaussian window.

**Hint:** the window comes from the method written in Step 1, and the convolution is done with the same function and settings as in Step 2.

**Check:** the Step 4 figure looks like a blurred version of Step 3.

### Step 5 — `det_and_trace(A, B, C)`

**Goal:** the determinant and trace of `M = [[A, C], [C, B]]` at every pixel.

The formulas are given in the Background section: `Det(M) = AB − C²` and `Trace(M) = A + B`.

**Hint:** NumPy arithmetic works element by element, so the formulas can be written directly on the arrays and are computed for every pixel at once. No loops are needed.

**Check:** in the Step 5 figure `Det(M)` is large only at the corners, while `Trace(M)` is large along every edge.

### Step 6 — `harris_response(det, trace)`

**Goal:** the Harris response at every pixel.

The formula is `R = Det(M) − alpha · Trace(M)²`.

**Hint:** this is one line of array arithmetic, as in Step 5. `self.alpha` is used rather than a typed-in number.

**Check:** a bright spot is shown at each internal corner in the Step 6 heatmap. The values can be read by hovering — they are negative along the edges.

### Step 7 — `threshold_response(response)`

**Goal:** the weak responses removed.

Every value that is not above `self.threshold_val` is replaced by 0, and the other values are kept unchanged. A value exactly equal to the threshold is therefore set to 0.

**Hint:** `np.where(condition, value_if_true, value_if_false)` chooses between two values at every pixel.

**Check:** a small white cluster of pixels is shown at each corner in the Step 7 figure.

### Step 8 — `find_peaks(thresh_response)`

**Goal:** one coordinate per cluster.

The local maxima of the thresholded response are returned as an `(N, 2)` array of `(row, col)` coordinates, at least `self.min_distance` pixels apart.

**Hint:** `peak_local_max` was used in Step 6 of Lab 04 (`orange_markers`), and it already returns this array. Lab 04 also passed `threshold_abs`; that is not needed here, because the weak responses were removed in Step 7. Only `min_distance` is set.

**Check:** one red dot is shown on each corner in the Step 8 figure — not a cluster of several dots.

### Step 9 — `det_coords(image)`

**Goal:** the whole pipeline run. Steps 2–8 are called in order, each result is stored on `self` so that it can be displayed afterwards, and `self.coords` is returned. Step 1 is not called here, because it is already called inside Step 4.

| Step | Stored on `self` as |
|------|---------------------|
| 2 | `self.grad_x`, `self.grad_y` |
| 3 | `self.grad_products` (all three products, as one tuple) |
| 4 | `self.smoothed` (`A`, `B` and `C`, as one tuple) |
| 5 | `self.det`, `self.trace` |
| 6 | `self.corner_response` |
| 7 | `self.thresh_response` |
| 8 | `self.coords` |

**Hint:** `segment` in Lab 04 (Step 10) has the same structure. A stored tuple can be passed on with `*`, which unpacks it into separate arguments: `f(*t)` is the same as `f(t[0], t[1], t[2])` for a tuple of three.

**Check:** 81 corners are reported by the evaluate script, and the Step 9 figure matches Step 8.

## What the Pytests Check

The tests are run with:

```bash
pytest test_lab05.py
```

- **Step 1** — matches the Gaussian-kernel formula exactly; a different `sigma` gives the expected kernel size and values.
- **Step 2** — matches `scipy.signal.convolve2d` with the Sobel kernels exactly; a vertical edge gives an x-gradient but no y-gradient.
- **Step 3** — matches the element-wise products on a hand-built example.
- **Step 4** — matches `scipy.signal.convolve2d` with the Gaussian window, for the default and a non-default `sigma`.
- **Step 5** — matches `AB − C²` and `A + B` on hand-built values.
- **Step 6** — matches `Det(M) − alpha · Trace(M)²`; `alpha` is respected.
- **Step 7** — values not above the threshold become 0 and the others are kept unchanged; `threshold_val` is respected.
- **Step 8** — matches `peak_local_max`, returning `(row, col)`; `min_distance` is respected.
- **Step 9** — the detected corners match the checkerboard's 9 × 9 grid (at most 3 extra or missing); every intermediate is stored on `self`; a very high `threshold_val` leaves no corners.

## Evaluation Script

In `evaluate_lab05.py` the checkerboard image is downloaded, each step is run in turn, and one figure per completed step is saved to `lab05_results.html`:

- the input image;
- **Step 1** — the Gaussian kernel as a heatmap;
- **Steps 2–5** — the gradients, their products, the smoothed sums, and the determinant and trace (shown as images — only the pattern matters here);
- **Step 6** — the Harris response as a heatmap (values shown on hover);
- **Step 7** — the pixels that survive the threshold;
- **Steps 8–9** — the detected corners on the checkerboard.

The script is run with:

```bash
python evaluate_lab05.py
```

`lab05_results.html` is then opened — in VS Code / Codespaces, by right-clicking the file → **Open with Live Server**, or by running `python -m http.server 5000` and opening the forwarded port.

## Report

`report_lab05.md` is to be filled in.
