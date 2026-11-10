# -*- coding: utf-8 -*-
"""Evaluate Lab 05 — Harris Corner Detector.

Run from the repo root:
    python Assignments/lab05/evaluate_lab05.py

Runs each step of HarrisCorner in turn and adds a figure for every step that
works. If a step isn't implemented yet, it stops there and still saves the
figures made so far — so you can run this after finishing each step.
Results are written to a single HTML file next to this script.
"""
import os
import sys
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab05 import HarrisCorner, CHECKERBOARD_IMAGE_URL
from helpers.dataloader import DATA_DIR, download_image, norm_img, rescale_crop
from helpers.plotting import save_html, png_source

HOVER = "x: %{x}<br>y: %{y}<extra></extra>"

def plot_image(img, title):
    """A single-panel image figure, embedded as a compressed PNG."""
    fig = go.Figure(go.Image(source=png_source(img), hovertemplate=HOVER))
    fig.update_layout(title_text=title, height=420)
    return fig

def plot_heatmaps(panels, title):
    """panels: list of (array, subtitle) pairs, each shown as a Jet heatmap.

    Only use this for small arrays (e.g. the Gaussian kernel) or where the
    actual numeric hover values matter (e.g. the final response map) — a
    full-resolution heatmap stores every pixel as uncompressed JSON text,
    which is much larger than an embedded PNG.
    """
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels])
    for col, (arr, _) in enumerate(panels, start=1):
        fig.add_trace(go.Heatmap(z=arr.astype(np.float32), colorscale="Jet", showscale=False),
                     row=1, col=col)
        # Square pixels, like the image figures (otherwise the heatmap is
        # stretched to fill the panel).
        fig.update_xaxes(constrain="domain", row=1, col=col)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{col if col > 1 else ''}",
                         constrain="domain", row=1, col=col)
    fig.update_layout(title_text=title, height=420)
    return fig

def plot_arrays_as_images(panels, title):
    """panels: list of (array, subtitle) pairs, each normalised to [0, 255]
    and shown as a compressed-PNG image — for diagnostic arrays (gradients,
    products) where only the visual pattern matters, not exact values."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels])
    for col, (arr, _) in enumerate(panels, start=1):
        display = norm_img(arr) * 255
        fig.add_trace(go.Image(source=png_source(display), hovertemplate=HOVER), row=1, col=col)
    fig.update_layout(title_text=title, height=360)
    return fig

def draw_corners(image_rgb, coords):
    """A red dot at each (row, col) corner."""
    overlay = image_rgb.copy()
    for r, c in coords:
        cv2.circle(overlay, (int(c), int(r)), radius=4, color=(255, 0, 0), thickness=-1)
    return overlay

def get_checkerboard_image():
    """Download the checkerboard image to Data/ if not cached; return processed arrays."""
    image_rgb = download_image(CHECKERBOARD_IMAGE_URL, save_path=DATA_DIR)
    image_rgb = rescale_crop(image_rgb, target_size=512)
    gray      = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_norm = norm_img(gray)
    return image_rgb, gray_norm

def evaluate():
    image_rgb, gray_norm = get_checkerboard_image()

    hc = HarrisCorner()
    figs = [plot_image(image_rgb, "Input image — checkerboard (81 internal corners)")]
    step = ""

    try:
        step = "Step 1 (gaussian_kernel)"
        window = hc.gaussian_kernel()
        figs.append(plot_heatmaps([(window, "Gaussian weighting window")],
                                  f"Step 1: {window.shape[0]}x{window.shape[0]} Gaussian kernel "
                                  f"(sigma={hc.sigma}) — a round blob, brightest at the centre"))
        print(f"{step} done — kernel shape {window.shape}.")

        step = "Step 2 (compute_gradients)"
        Ix, Iy = hc.compute_gradients(gray_norm)
        figs.append(plot_arrays_as_images([(Ix, "Ix"), (Iy, "Iy")],
                                          "Step 2: Sobel gradients — Ix shows vertical edges, Iy horizontal edges"))
        print(f"{step} done.")

        step = "Step 3 (gradient_products)"
        Ixx, Iyy, Ixy = hc.gradient_products(Ix, Iy)
        figs.append(plot_arrays_as_images(
            [(Ixx, "Ix^2"), (Iyy, "Iy^2"), (Ixy, "Ix*Iy")],
            "Step 3: Gradient products — Ix*Iy is non-zero only where both gradients are, at the corners"))
        print(f"{step} done.")

        step = "Step 4 (smooth_products)"
        A, B, C = hc.smooth_products(Ixx, Iyy, Ixy)
        figs.append(plot_arrays_as_images([(A, "A"), (B, "B"), (C, "C")],
                                          "Step 4: Gaussian-weighted sums A, B, C — a blurred version of Step 3"))
        print(f"{step} done.")

        step = "Step 5 (det_and_trace)"
        det, trace = hc.det_and_trace(A, B, C)
        figs.append(plot_arrays_as_images([(det, "Det(M)"), (trace, "Trace(M)")],
                                          "Step 5: Det(M) is large only at corners; Trace(M) is large along every edge"))
        print(f"{step} done.")

        step = "Step 6 (harris_response)"
        response = hc.harris_response(det, trace)
        figs.append(plot_heatmaps([(response, "Harris response R")],
                                  f"Step 6: Harris response (alpha={hc.alpha}) — bright spots at the "
                                  "corners, negative along edges (hover to read values)"))
        print(f"{step} done — largest response {response.max():.3f}.")

        step = "Step 7 (threshold_response)"
        thresh_response = hc.threshold_response(response)
        kept = (thresh_response > 0).astype(np.uint8) * 255
        figs.append(plot_image(kept, f"Step 7: Responses above threshold_val={hc.threshold_val} "
                                     "(white) — a small cluster of pixels at each corner"))
        print(f"{step} done — {int(np.count_nonzero(kept))} pixels kept.")

        step = "Step 8 (find_peaks)"
        coords = hc.find_peaks(thresh_response)
        figs.append(plot_image(draw_corners(image_rgb, coords),
                               f"Step 8: {len(coords)} peaks (min_distance={hc.min_distance}) "
                               "— one red dot per corner, not a cluster"))
        print(f"{step} done — {len(coords)} corners.")

        step = "Step 9 (det_coords)"
        coords = hc.det_coords(gray_norm)
        figs.append(plot_image(draw_corners(image_rgb, coords),
                               f"Step 9: Full pipeline — {len(coords)} corners detected (81 expected)"))
        print(f"{step} done — {len(coords)} corners detected.")

    except NotImplementedError:
        print(f"\n{step} is not implemented yet — stopping here.")
        print("Figures for the earlier steps are still saved below.")

    save_html(*figs, output_path=os.path.join(LAB_DIR, "lab05_results.html"))

if __name__ == "__main__":
    evaluate()
