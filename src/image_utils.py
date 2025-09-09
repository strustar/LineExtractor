"""
Image processing utilities for CAD line extraction system.
"""

from typing import Tuple, Optional
import logging

import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale.

    Args:
        image_bgr: BGR image array

    Returns:
        Grayscale image array
    """
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def deskew_by_hough(gray: np.ndarray, angle_search_deg: float = 7.0) -> Tuple[np.ndarray, float]:
    """Deskew image using Hough line detection.

    Args:
        gray: Grayscale image
        angle_search_deg: Maximum angle to search for skew

    Returns:
        Deskewed image and rotation angle
    """
    try:
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=200)

        if lines is None or len(lines) == 0:
            logger.debug("No lines detected for deskewing")
            return gray, 0.0

        angles = []
        for rho, theta in lines[:, 0, :]:
            deg = (theta * 180.0 / np.pi) - 90.0
            if -angle_search_deg <= deg <= angle_search_deg:
                angles.append(deg)

        if not angles:
            logger.debug("No angles within search range")
            return gray, 0.0

        angle = float(np.median(angles))
        logger.debug(f"Deskewing by {angle:.2f} degrees")

        # Rotate image
        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rot, angle

    except Exception as e:
        logger.warning(f"Deskewing failed: {e}")
        return gray, 0.0


def binarize_sauvola(gray: np.ndarray, window_size: int = 41, k: float = 0.25) -> np.ndarray:
    """Apply Sauvola binarization to grayscale image.

    Args:
        gray: Grayscale image
        window_size: Window size for local thresholding
        k: Sauvola parameter

    Returns:
        Binary image
    """
    # Sauvola via scikit-image formula replicated with OpenCV primitives
    mean = cv2.boxFilter(gray.astype(np.float32), ddepth=-1, ksize=(window_size, window_size), normalize=True)
    sq = cv2.sqrIntegral(gray)[1:, 1:]  # integral square
    # compute local variance via integral images
    h, w = gray.shape
    hw = window_size // 2
    # pad for simplicity
    pad = cv2.copyMakeBorder(gray, hw, hw, hw, hw, cv2.BORDER_REPLICATE)
    II = cv2.integral(pad)
    SS = cv2.integral(cv2.multiply(pad, pad))
    win = window_size * window_size
    var = np.zeros_like(gray, dtype=np.float32)
    for y in range(h):
        y0, y1 = y, y + window_size
        for x in range(w):
            x0, x1 = x, x + window_size
            s = II[y1, x1] - II[y0, x1] - II[y1, x0] + II[y0, x0]
            ss = SS[y1, x1] - SS[y0, x1] - SS[y1, x0] + II[y0, x0]
            mu = s / win
            var[y, x] = max(ss / win - mu * mu, 0.0)
    std = np.sqrt(var)
    R = 128.0
    th = mean * (1 + k * ((std / R) - 1))
    bin_img = (gray.astype(np.float32) > th).astype(np.uint8) * 255
    return bin_img


def simple_binarize(gray: np.ndarray) -> np.ndarray:
    """Apply simple adaptive thresholding.

    Args:
        gray: Grayscale image

    Returns:
        Binary image
    """
    # Fallback to adaptive threshold if Sauvola is expensive
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 10)


def rotate_rgb(image_rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate RGB image by given angle.

    Args:
        image_rgb: RGB image
        angle: Rotation angle in degrees

    Returns:
        Rotated RGB image
    """
    h, w = image_rgb.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def validate_image(image: np.ndarray) -> bool:
    """Validate image array.

    Args:
        image: Image array to validate

    Returns:
        True if image is valid, False otherwise
    """
    if not isinstance(image, np.ndarray):
        return False
    if image.size == 0:
        return False
    if len(image.shape) not in [2, 3]:
        return False
    return True


def get_image_info(image: np.ndarray) -> dict:
    """Get image information.

    Args:
        image: Image array

    Returns:
        Dictionary containing image metadata
    """
    if not validate_image(image):
        raise ImageProcessingError("Invalid image array")

    info = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "size": image.size,
        "channels": 1 if len(image.shape) == 2 else image.shape[2],
        "memory_mb": image.nbytes / (1024 * 1024)
    }
    return info
