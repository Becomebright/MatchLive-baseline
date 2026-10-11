"""Shot-cut detection from colour-histogram change between consecutive frames.

Every hard cut starts a new shot: the trackers are reset there and track keys carry the
shot id.
"""

from __future__ import annotations

import cv2
import numpy as np


def shot_histogram(frame: np.ndarray) -> np.ndarray:
    """Hue-saturation histograms of a 2x2 grid over the frame, each normalized to sum 1.

    Per-quadrant histograms (instead of one for the whole frame) catch cuts between two
    shots with similar overall colours, e.g. two wide views of the same pitch from
    different cameras, where the grass/crowd layout moves even if the global mix does not.
    """
    # Downscale first: the histogram only needs the colour layout, and this makes the
    # result independent of the input resolution.
    small = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hists = []
    for rows in (slice(0, 36), slice(36, 72)):
        for cols in (slice(0, 64), slice(64, 128)):
            h = cv2.calcHist([hsv[rows, cols]], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hists.append(h / max(float(h.sum()), 1.0))
    return np.stack(hists).astype(np.float32)


def shot_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Mean Bhattacharyya distance over the four quadrants: 0 = identical colour
    # distributions, 1 = no overlap at all.
    return float(np.mean([cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b)]))


class ShotDetector:
    """Synchronous: call update() once per frame in order; it returns the frame's shot id."""

    def __init__(self, cut_threshold: float):
        self.cut_threshold = float(cut_threshold)
        self.shot_id = 0
        self.prev_hist: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> int:
        hist = shot_histogram(frame)
        if self.prev_hist is not None and shot_distance(self.prev_hist, hist) > self.cut_threshold:
            self.shot_id += 1
        self.prev_hist = hist
        return self.shot_id
