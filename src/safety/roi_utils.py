"""Shared ROI depth utilities for safety components.

Used by both the simulation deploy loop (src/deployment/deploy.py) and
the ROS2 safety monitor (ros_ws/src/safety_monitor/safety_monitor/safety_node.py)
to ensure consistent proximity-braking behaviour across both paths.
"""
from __future__ import annotations

import numpy as np


def centre_roi_min_depth(img: np.ndarray, roi_frac: float = 0.3) -> float:
    """Return minimum depth (metres) in the centre ROI of a depth image.

    Parameters
    ----------
    img:
        Depth image array of shape ``(H, W)`` or ``(H, W, C)`` with values
        in metres.  Only the spatial centre ``roi_frac`` of rows and columns
        is examined.
    roi_frac:
        Fraction of the image height and width that defines the centre ROI.
        Default ``0.3`` → centre 30 % of the image.

    Returns
    -------
    float
        Minimum depth in the ROI, or ``float('inf')`` when the ROI is empty.
    """
    h, w = img.shape[:2]
    r0 = int(h * (1 - roi_frac) / 2)
    r1 = int(h * (1 + roi_frac) / 2)
    c0 = int(w * (1 - roi_frac) / 2)
    c1 = int(w * (1 + roi_frac) / 2)
    roi = img[r0:r1, c0:c1]
    return float(roi.min()) if roi.size > 0 else float("inf")
