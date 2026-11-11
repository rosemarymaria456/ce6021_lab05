# -*- coding: utf-8 -*-
"""Lab 05 — Harris Corner Detector.

Detect corner points in an image from scratch — no cv2.cornerHarris. The
work is split into small steps, each one a method below that takes an array
in and returns the result out:

  1. gaussian_kernel()                    a 2D Gaussian weighting window
  2. compute_gradients(image)             Sobel gradients (Ix, Iy)
  3. gradient_products(Ix, Iy)            Ix^2, Iy^2, Ix*Iy
  4. smooth_products(Ixx, Iyy, Ixy)       Gaussian-weighted sums (A, B, C)
  5. det_and_trace(A, B, C)               Det(M) and Trace(M) at every pixel
  6. harris_response(det, trace)          Harris response R per pixel
  7. threshold_response(response)         weak responses set to zero
  8. find_peaks(thresh_response)          one (row, col) per corner
  9. det_coords(image)                    put the steps together

Run evaluate_lab05.py after each step to see your progress.
"""
import numpy as np
from scipy import signal
import os
import sys
from skimage.feature import peak_local_max

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import DATA_DIR, download_image, norm_img, rescale_crop  # noqa: F401

CHECKERBOARD_IMAGE_URL = (
    "https://github.com/tonyscan6003/etivities/blob/main/"
    "10x10_checkered_board.png?raw=true"
)

class HarrisCorner:
    """Harris Corner Detector, built from the second-moment matrix of the
    image gradients.

    All settings have defaults, so no arguments are needed:

        hc     = HarrisCorner()
        coords = hc.det_coords(image)

    Any setting can be changed with a keyword argument, e.g.:

        hc = HarrisCorner(sigma=3, min_distance=10)

    Use these settings (self.sigma, self.alpha, ...) inside your methods
    rather than typing the numbers in directly.
    """

    def __init__(self, **kwargs):
        self.sigma         = kwargs.get('sigma',         2)
        self.alpha         = kwargs.get('alpha',         0.05)
        self.threshold_val = kwargs.get('threshold_val', 0.001)
        self.min_distance  = kwargs.get('min_distance',  5)

        # Intermediate results, filled in by det_coords() (step 9)
        self.grad_x          = None
        self.grad_y          = None
        self.grad_products   = None
        self.smoothed        = None
        self.det             = None
        self.trace           = None
        self.corner_response = None
        self.thresh_response = None
        self.coords          = None

    def gaussian_kernel(self):
        """Step 1 — a 2D Gaussian weighting window sized from self.sigma.

        Returns:
            np.ndarray: (k, k) 2D Gaussian kernel, k = int(6 * sigma + 1).
        """
        k = int(6 * self.sigma + 1)
        half = (k - 1) // 2
        x = np.arange(-half, half + 1)
        g = (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * self.sigma ** 2))
        return np.outer(g, g)

    def compute_gradients(self, image):
        """Step 2 — Sobel gradients of the image in x and y.

        Args:
            image (np.ndarray): 2D greyscale image.

        Returns:
            (Ix, Iy): each a 2D array the same shape as image.
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = signal.convolve2d(image, kernel_x, mode='same', boundary='symm')
        Iy = signal.convolve2d(image, kernel_y, mode='same', boundary='symm')
        return Ix, Iy

    def gradient_products(self, Ix, Iy):
        """Step 3 — element-wise products of the gradients.

        Args:
            Ix, Iy (np.ndarray): gradients from compute_gradients().

        Returns:
            (Ixx, Iyy, Ixy): Ix*Ix, Iy*Iy, Ix*Iy — each the same shape as Ix.
        """
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        return Ixx, Iyy, Ixy

    def smooth_products(self, Ixx, Iyy, Ixy):
        """Step 4 — Gaussian-weighted sum of each gradient product over a
        local window (the entries of the second-moment matrix M).

        Args:
            Ixx, Iyy, Ixy (np.ndarray): gradient products from gradient_products().

        Returns:
            (A, B, C): the Gaussian-weighted Ixx, Iyy, Ixy — same shape as Ixx.
        """
        window = self.gaussian_kernel()
        A = signal.convolve2d(Ixx, window, mode='same', boundary='symm')
        B = signal.convolve2d(Iyy, window, mode='same', boundary='symm')
        C = signal.convolve2d(Ixy, window, mode='same', boundary='symm')
        return A, B, C

    def det_and_trace(self, A, B, C):
        """Step 5 — the determinant and trace of M = [[A, C], [C, B]] at
        every pixel.

        Args:
            A, B, C (np.ndarray): the smoothed products from smooth_products().

        Returns:
            (det, trace): Det(M) = A*B - C^2 and Trace(M) = A + B, same shape as A.
        """
        det = A * B - C ** 2
        trace = A + B
        return det, trace

    def harris_response(self, det, trace):
        """Step 6 — the Harris response R = Det(M) - alpha * Trace(M)^2.

        Args:
            det, trace (np.ndarray): from det_and_trace().

        Returns:
            np.ndarray: the response map, same shape as det.
        """
        return det - self.alpha * (trace ** 2)

    def threshold_response(self, response):
        """Step 7 — keep only the strong responses: every value not above
        self.threshold_val is set to 0.

        Args:
            response (np.ndarray): response map from harris_response().

        Returns:
            np.ndarray: the thresholded response map, same shape as response.
        """
        return np.where(response > self.threshold_val, response, 0)

    def find_peaks(self, thresh_response):
        """Step 8 — one coordinate per corner: the local maxima of the
        thresholded response, at least self.min_distance pixels apart.

        Args:
            thresh_response (np.ndarray): from threshold_response().

        Returns:
            np.ndarray: shape (N, 2) array of (row, col) coordinates.
        """
        return peak_local_max(thresh_response, min_distance=self.min_distance)

    def det_coords(self, image):
        """Step 9 — run the whole pipeline using the methods above.

        Stores each intermediate result on self so they can be displayed
        afterwards.

        Args:
            image (np.ndarray): 2D greyscale image.

        Returns:
            np.ndarray: shape (N, 2) array of (row, col) corner coordinates.
        """
        self.grad_x, self.grad_y = self.compute_gradients(image)
        self.grad_products = self.gradient_products(self.grad_x, self.grad_y)
        self.smoothed = self.smooth_products(*self.grad_products)
        self.det, self.trace = self.det_and_trace(*self.smoothed)
        self.corner_response = self.harris_response(self.det, self.trace)
        self.thresh_response = self.threshold_response(self.corner_response)
        self.coords = self.find_peaks(self.thresh_response)
        return self.coords
