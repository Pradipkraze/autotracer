# -*- coding: utf-8 -*-
"""
Core image processing pipeline for raster-to-vector conversion.

Pipeline steps:
  1. Binarization  – convert to 1-bit black/white
  2. Thresholding  – adaptive or global threshold
  3. Smoothing     – median or Gaussian noise removal
  4. Skeletonization – thin lines to single-pixel width
  5. Vectorization – trace raster lines → vector polylines
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte


# ── Step 1 & 2 ──────────────────────────────────────────────────────────────

def binarize(image_array: np.ndarray,
             threshold_method: str = "otsu",
             manual_threshold: int = 128,
             invert: bool = False) -> np.ndarray:
    """
    Convert a grayscale/colour array to a binary (0/255) uint8 array.

    Parameters
    ----------
    image_array      : H×W or H×W×C uint8 ndarray
    threshold_method : 'otsu' | 'adaptive' | 'manual'
    manual_threshold : value used when method == 'manual'
    invert           : if True, swap black and white after thresholding
                       (needed when map lines are lighter than background)

    Returns
    -------
    binary : H×W uint8 ndarray (0 = background, 255 = foreground/lines)
    """
    # Ensure grayscale
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()

    if threshold_method == "otsu":
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=5
        )
    else:  # manual
        _, binary = cv2.threshold(
            gray, manual_threshold, 255, cv2.THRESH_BINARY
        )

    if invert:
        binary = cv2.bitwise_not(binary)

    return binary


# ── Step 3 ───────────────────────────────────────────────────────────────────

def smooth(binary: np.ndarray,
           method: str = "median",
           kernel_size: int = 3) -> np.ndarray:
    """
    Remove salt-and-pepper noise from a binary image.

    Parameters
    ----------
    binary      : H×W uint8 binary image
    method      : 'median' | 'gaussian'
    kernel_size : filter kernel size (must be odd, ≥ 3)

    Returns
    -------
    smoothed : H×W uint8 binary image
    """
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if method == "median":
        smoothed = cv2.medianBlur(binary, ksize)
    else:  # gaussian
        sigma = ksize / 6.0
        blurred = cv2.GaussianBlur(binary, (ksize, ksize), sigma)
        _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return smoothed


# ── Step 4 ───────────────────────────────────────────────────────────────────

def skeletonize_image(binary: np.ndarray) -> np.ndarray:
    """
    Thin all foreground lines to single-pixel width (skeletonization).

    Parameters
    ----------
    binary : H×W uint8 image (255 = lines, 0 = background)

    Returns
    -------
    skeleton : H×W uint8 image with thinned lines
    """
    # skimage expects bool; lines = True
    bool_img = binary > 127
    skeleton_bool = skeletonize(bool_img)
    return img_as_ubyte(skeleton_bool)


# ── Step 5 ───────────────────────────────────────────────────────────────────

_N8 = [(-1, -1), (-1, 0), (-1, 1),
       ( 0, -1),           ( 0, 1),
       ( 1, -1), ( 1, 0), ( 1, 1)]


def vectorize(skeleton: np.ndarray,
              smooth_lines: bool = True,
              dp_epsilon: float = 1.5,
              min_contour_length: int = 10) -> list:
    """
    Convert a skeletonized image into clean, non-overlapping polylines.

    Topological graph approach
    --------------------------
    Nodes  = endpoint pixels (degree 1) + junction pixels (degree >= 3).
    Branches = interior pixel chains (degree 2) connecting two nodes.

    Key insight: NODE pixels are NEVER added to the visited set during
    branch walks. Only interior (degree-2) pixels are consumed. This means
    a junction can be the shared endpoint of multiple branches without any
    branch "stealing" it from the others.

    Each directed edge (node → neighbour) is walked at most once via
    a visited_edges set, guaranteeing no duplicate polylines.
    """
    skel = (skeleton > 0)
    rows, cols = np.where(skel)
    on_set = set(zip(rows.tolist(), cols.tolist()))
    if not on_set:
        return []

    # ── Build neighbour lookup ───────────────────────────────────────────────
    nbr_map = {}
    for r, c in on_set:
        nbr_map[(r, c)] = [(r + dr, c + dc) for dr, dc in _N8
                           if (r + dr, c + dc) in on_set]

    degree = {px: len(nbrs) for px, nbrs in nbr_map.items()}

    # ── Classify pixels ──────────────────────────────────────────────────────
    # Nodes: anything that is NOT a simple interior pixel (degree == 2)
    # Isolated pixels (degree 0) and endpoints (degree 1) are also nodes.
    nodes = {px for px, d in degree.items() if d != 2}

    # If skeleton is one big loop with no endpoints/junctions, treat any
    # pixel as a node so we still get one polyline out.
    if not nodes:
        nodes = {next(iter(on_set))}

    # ── Walk each branch exactly once ────────────────────────────────────────
    visited_interior = set()   # degree-2 pixels already assigned to a branch
    visited_edges    = set()   # (node_px, neighbour_px) directed edges already walked

    polylines = []

    for node in nodes:
        for first_step in nbr_map[node]:
            edge_key = (node, first_step)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            # Walk: from node through first_step until we hit another node
            path = [node, first_step]
            visited_interior.add(first_step)   # mark interior pixel consumed

            prev, cur = node, first_step
            while cur not in nodes:
                # Find next pixel: any neighbour that isn't where we came from
                # and hasn't been consumed as interior of another branch
                candidates = [n for n in nbr_map[cur]
                              if n != prev and n not in visited_interior]

                if not candidates:
                    # Dead end in interior — shouldn't happen on clean skeleton
                    break

                # If multiple candidates exist cur is actually a junction;
                # treat it as a node (stop here).
                if len(candidates) > 1:
                    break

                nxt = candidates[0]
                visited_interior.add(nxt)
                # Mark reverse direction so we don't re-walk this edge
                visited_edges.add((nxt, prev))
                path.append(nxt)
                prev, cur = cur, nxt

            # Mark the terminal edge so the other node doesn't re-walk it
            visited_edges.add((cur, prev))

            if len(path) < min_contour_length:
                continue

            # (row, col) → (col, row) = (x, y) for QGIS
            pts = np.array([[c, r] for r, c in path], dtype=np.float32)

            if smooth_lines and len(pts) >= 3:
                pts_dp = cv2.approxPolyDP(
                    pts.reshape(-1, 1, 2),
                    epsilon=dp_epsilon,
                    closed=False
                ).squeeze().astype(np.float32)
                if pts_dp.ndim == 2 and len(pts_dp) >= 2:
                    pts = pts_dp

            if len(pts) >= 2:
                polylines.append(pts)

    return polylines


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(image_array: np.ndarray,
                 threshold_method: str = "otsu",
                 manual_threshold: int = 128,
                 invert: bool = False,
                 smooth_method: str = "median",
                 smooth_kernel: int = 3,
                 smooth_lines: bool = True,
                 dp_epsilon: float = 1.5,
                 min_contour_length: int = 10,
                 progress_callback=None) -> tuple:
    """
    Run the complete 5-step pipeline.

    Returns
    -------
    (polylines, intermediates)
    polylines     : list of Nx2 float32 ndarrays (pixel coords, x=col, y=row)
    intermediates : dict with keys 'binary', 'smoothed', 'skeleton'
    """
    def _progress(step, msg):
        if progress_callback:
            progress_callback(step, msg)

    _progress(10, "Step 1 & 2: Binarization & Thresholding…")
    binary = binarize(image_array, threshold_method, manual_threshold, invert)

    _progress(35, "Step 3: Smoothing (noise removal)…")
    smoothed = smooth(binary, smooth_method, smooth_kernel)

    _progress(55, "Step 4: Skeletonization…")
    skeleton = skeletonize_image(smoothed)

    _progress(80, "Step 5: Vectorization…")
    polylines = vectorize(skeleton, smooth_lines, dp_epsilon, min_contour_length)

    _progress(100, f"Done — {len(polylines)} line features created.")

    intermediates = {
        "binary": binary,
        "smoothed": smoothed,
        "skeleton": skeleton,
    }
    return polylines, intermediates
