#!/usr/bin/env python3

"""
Crop a rotated rectangle from an image using OpenCV.
Based on: https://stackoverflow.com/a/11627903
"""

import cv2
import numpy as np


def crop_rotated_rectangle(image, rect):
    """
    Crop a rotated rectangle from an image.
    @param image - 2D or 3D numpy array (the source image).
    @param rect  - OpenCV rotated rectangle tuple: ((cx, cy), (w, h), angle_degrees)
    @return Cropped and rotated image region, or None if the region is outside bounds.
    """
    # Get the rotation matrix for the rectangle angle.
    center, size, angle = rect
    center = tuple(map(float, center))
    size = tuple(map(float, size))

    # Ensure OpenCV warpAffine/remap gets a supported contiguous dtype.
    image = np.nan_to_num(np.asarray(image), nan=0.0, posinf=1.0, neginf=0.0)
    image = np.ascontiguousarray(image, dtype=np.float32)

    # Ensure the rectangle is within the image bounds.
    rows, cols = image.shape[:2]

    # Get the four corner points of the rotated rectangle.
    box = cv2.boxPoints(rect)

    # Check if any corner is outside the image — return None if so.
    for point in box:
        if point[0] < 0 or point[1] < 0 or point[0] >= cols or point[1] >= rows:
            return None

    # Get rotation matrix.
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the entire image.
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Crop the rectangle out of the rotated image.
    # After rotation, the rectangle is axis-aligned, so we can crop directly.
    crop_w = int(size[0])
    crop_h = int(size[1])

    x1 = int(center[0] - crop_w / 2)
    y1 = int(center[1] - crop_h / 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp to image bounds.
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(cols, x2)
    y2 = min(rows, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    cropped = rotated[y1:y2, x1:x2]
    return cropped