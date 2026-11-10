# -*- coding: utf-8 -*-
"""Tests for Lab 05 — Harris Corner Detector.

Grouped by step — a failing test names the step that needs attention.
"""
import numpy as np
import pytest
from scipy import signal
from scipy.spatial import distance_matrix
from skimage.feature import peak_local_max
from contextlib import redirect_stdout
from os import devnull
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

with redirect_stdout(open(devnull, 'w')):
    import cv2
    from lab05 import HarrisCorner, CHECKERBOARD_IMAGE_URL
    from helpers.dataloader import DATA_DIR, download_image, norm_img, rescale_crop

@pytest.fixture(scope="module")
def checkerboard():
    """The checkerboard, cropped to 512x512 greyscale and normalised to [0, 1]
    (exactly as evaluate_lab05.py prepares it)."""
    image_rgb = download_image(CHECKERBOARD_IMAGE_URL, save_path=DATA_DIR)
    crop_image = rescale_crop(image_rgb, target_size=512)
    return norm_img(cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY))

def reference_kernel(sigma):
    k_size = int(6 * sigma + 1)
    rng = (k_size - 1) // 2
    x = np.arange(-rng, rng + 1)
    g = np.exp((-x ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return np.outer(g, g)

def test_constructor_defaults():
    hc = HarrisCorner()
    assert (hc.sigma, hc.alpha, hc.threshold_val, hc.min_distance) == (2, 0.05, 0.001, 5)

# ── Step 1: gaussian_kernel ─────────────────────────────────────────────────

def test_gaussian_kernel_matches_reference():
    assert np.allclose(HarrisCorner().gaussian_kernel(), reference_kernel(2))

def test_gaussian_kernel_respects_sigma():
    kernel = HarrisCorner(sigma=3).gaussian_kernel()
    assert kernel.shape == (19, 19)
    assert np.allclose(kernel, reference_kernel(3))

# ── Step 2: compute_gradients ───────────────────────────────────────────────

def test_compute_gradients_matches_sobel_reference(checkerboard):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1, -2, -1]])
    expected_x = signal.convolve2d(checkerboard, kernel_x, boundary='symm', mode='same')
    expected_y = signal.convolve2d(checkerboard, kernel_y, boundary='symm', mode='same')
    Ix, Iy = HarrisCorner().compute_gradients(checkerboard)
    assert Ix.shape == checkerboard.shape and Iy.shape == checkerboard.shape
    assert np.allclose(Ix, expected_x)
    assert np.allclose(Iy, expected_y)

def test_compute_gradients_directions():
    """A vertical edge (dark left, bright right) has an x-gradient but no y-gradient."""
    img = np.zeros((10, 10))
    img[:, 5:] = 1.0
    Ix, Iy = HarrisCorner().compute_gradients(img)
    assert abs(Ix[5, 5]) > 0
    assert np.allclose(Iy, 0)

# ── Step 3: gradient_products ───────────────────────────────────────────────

def test_gradient_products_matches_elementwise_formula():
    rng = np.random.default_rng(0)
    Ix = rng.random((5, 5))
    Iy = rng.random((5, 5))
    Ixx, Iyy, Ixy = HarrisCorner().gradient_products(Ix, Iy)
    assert np.allclose(Ixx, Ix * Ix)
    assert np.allclose(Iyy, Iy * Iy)
    assert np.allclose(Ixy, Ix * Iy)

# ── Step 4: smooth_products ─────────────────────────────────────────────────

@pytest.mark.parametrize("sigma", [2, 3])
def test_smooth_products_matches_convolve_reference(sigma):
    rng = np.random.default_rng(1)
    Ixx, Iyy, Ixy = (rng.random((40, 40)) for _ in range(3))
    window = reference_kernel(sigma)
    A, B, C = HarrisCorner(sigma=sigma).smooth_products(Ixx, Iyy, Ixy)
    assert np.allclose(A, signal.convolve2d(Ixx, window, boundary='symm', mode='same'))
    assert np.allclose(B, signal.convolve2d(Iyy, window, boundary='symm', mode='same'))
    assert np.allclose(C, signal.convolve2d(Ixy, window, boundary='symm', mode='same'))

# ── Step 5: det_and_trace ───────────────────────────────────────────────────

def test_det_and_trace_matches_formula():
    A = np.array([[4.0, 2.0], [1.0, 3.0]])
    B = np.array([[5.0, 1.0], [2.0, 6.0]])
    C = np.array([[1.0, 0.5], [0.0, 2.0]])
    det, trace = HarrisCorner().det_and_trace(A, B, C)
    assert np.allclose(det, A * B - C ** 2)
    assert np.allclose(trace, A + B)

# ── Step 6: harris_response ─────────────────────────────────────────────────

def test_harris_response_matches_formula():
    det   = np.array([[19.0, 1.75], [2.0, 14.0]])
    trace = np.array([[9.0, 3.0], [3.0, 9.0]])
    assert np.allclose(HarrisCorner().harris_response(det, trace), det - 0.05 * trace ** 2)

def test_harris_response_respects_alpha():
    det   = np.full((3, 3), 15.0)
    trace = np.full((3, 3), 8.0)
    assert np.allclose(HarrisCorner(alpha=0.2).harris_response(det, trace), 15.0 - 0.2 * 64)

# ── Step 7: threshold_response ──────────────────────────────────────────────

def test_threshold_response_keeps_values_above_threshold():
    response = np.array([[0.0005, 0.002], [-1.0, 0.5]])
    expected = np.array([[0.0,    0.002], [ 0.0, 0.5]])
    assert np.allclose(HarrisCorner().threshold_response(response), expected)

def test_threshold_response_respects_threshold_val():
    response = np.array([[50.0, 150.0], [80.0, 100.0]])
    expected = np.array([[0.0,  150.0], [0.0,  0.0]])
    assert np.allclose(HarrisCorner(threshold_val=100).threshold_response(response), expected)

# ── Step 8: find_peaks ──────────────────────────────────────────────────────

def two_peaks():
    response = np.zeros((40, 40))
    response[10, 10] = 100
    response[10, 14] = 90       # 4 px from the first peak
    response[30, 25] = 80
    return response

def test_find_peaks_matches_peak_local_max():
    response = two_peaks()
    expected = peak_local_max(response, min_distance=5)
    result = HarrisCorner().find_peaks(response)
    assert result.shape == (2, 2)
    assert np.array_equal(result, expected)

def test_find_peaks_respects_min_distance():
    found = HarrisCorner(min_distance=2).find_peaks(two_peaks())
    assert len(found) == 3
    assert {tuple(p) for p in found} == {(10, 10), (10, 14), (30, 25)}

# ── Step 9: det_coords (end-to-end) ─────────────────────────────────────────

def generate_grid(start=51, spacing=51, size=9):
    """Expected 9x9 regular grid of internal checkerboard corners, as (row, col)."""
    r = np.arange(start, start + spacing * size, spacing)
    return np.array([[ri, ci] for ri in r for ci in r])

def test_det_coords_matches_checkerboard_grid(checkerboard):
    coords = HarrisCorner().det_coords(checkerboard)
    assert coords.ndim == 2 and coords.shape[1] == 2
    dist = distance_matrix(coords, generate_grid())
    unmatched = int(np.sum(dist.min(axis=1) > 10))       # detections far from any corner
    missed    = int(np.sum(dist.min(axis=0) > 10))       # corners with no detection
    assert unmatched <= 3, f"{unmatched} detected points are not near a checkerboard corner"
    assert missed <= 3, f"{missed} checkerboard corners were not detected"

def test_det_coords_stores_intermediates_on_self(checkerboard):
    hc = HarrisCorner()
    coords = hc.det_coords(checkerboard)
    for name in ["grad_x", "grad_y", "grad_products", "smoothed", "det", "trace",
                 "corner_response", "thresh_response", "coords"]:
        assert getattr(hc, name) is not None, f"self.{name} was not set"
    assert hc.corner_response.shape == checkerboard.shape
    assert np.array_equal(hc.coords, coords)

def test_det_coords_uses_settings(checkerboard):
    """A threshold above every response leaves no corners."""
    assert len(HarrisCorner(threshold_val=1e6).det_coords(checkerboard)) == 0
