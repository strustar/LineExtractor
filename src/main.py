import os
import sys
from pathlib import Path
import math
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import cv2
import fitz  # PyMuPDF

# Import configuration (robust: avoid picking up pip 'config' package)
ROOT_DIR = Path(__file__).resolve().parent.parent

def _load_user_config():
    import importlib.util
    cfg_path = ROOT_DIR / "config.py"
    if cfg_path.exists():
        spec = importlib.util.spec_from_file_location("user_config", str(cfg_path))
        if spec and spec.loader:  # type: ignore
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(mod)  # type: ignore
            return mod
    return None

_user_cfg = _load_user_config()

class _DefaultConfig:
    ROOT_DIR = ROOT_DIR
    OUTPUT_DIR = ROOT_DIR / "outputs" / "lines"
    WEB_DIR = ROOT_DIR
    DEFAULT_PDF_PATH = next((p for p in [ROOT_DIR / "샘플도면.pdf", ROOT_DIR / "풀도면.pdf"] if p.exists()), ROOT_DIR / "샘플도면.pdf")
    PDF_CONFIG = {"default_dpi": 600, "fallback_dpi": 300, "parallel_processing": False, "max_workers": 1}
    IMAGE_CONFIG = {"deskew_angle_search": 7.0, "canny_low_threshold": 50, "canny_high_threshold": 150, "min_absolute_line_length": 10}
    ALGORITHM_CONFIG = {}
    LOGGING_CONFIG = {"level": "INFO", "format": "%(asctime)s %(levelname)s %(message)s"}
    PERFORMANCE_CONFIG = {"memory_limit_mb": 2048, "processing_timeout_seconds": 300, "max_image_dimension": 6000}

def _default_get_pdf_files():
    candidates = [(_DefaultConfig.ROOT_DIR / "샘플도면.pdf"), (_DefaultConfig.ROOT_DIR / "풀도면.pdf")]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return existing
    return list(_DefaultConfig.ROOT_DIR.glob("*.pdf"))

if _user_cfg is not None and all(hasattr(_user_cfg, a) for a in [
    "PDF_CONFIG", "IMAGE_CONFIG", "ALGORITHM_CONFIG", "LOGGING_CONFIG", "PERFORMANCE_CONFIG"
]):
    config_module = _user_cfg  # type: ignore
    get_pdf_files = getattr(_user_cfg, "get_pdf_files", _default_get_pdf_files)  # type: ignore
else:
    config_module = _DefaultConfig
    get_pdf_files = _default_get_pdf_files

PDF_CONFIG = config_module.PDF_CONFIG
IMAGE_CONFIG = config_module.IMAGE_CONFIG
ALGORITHM_CONFIG = config_module.ALGORITHM_CONFIG
LOGGING_CONFIG = config_module.LOGGING_CONFIG
PERFORMANCE_CONFIG = config_module.PERFORMANCE_CONFIG

# Import custom exceptions and utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))

from exceptions import (
    CADProcessingError, PDFProcessingError, ImageProcessingError,
    LineDetectionError, ConfigurationError, ValidationError
)

# Import utilities
from memory_manager import MemoryManager, optimize_image_for_processing, calculate_optimal_workers
from pdf_processor import PDFProcessor
from image_utils import to_gray, deskew_by_hough, simple_binarize
from config_validator import ConfigValidator

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


def ensure_dir(path: Path) -> None:
	"""Create directory if it doesn't exist."""
	try:
		os.makedirs(path, exist_ok=True)
		logger.debug(f"Directory ensured: {path}")
	except OSError as e:
		logger.error(f"Failed to create directory {path}: {e}")
		raise


def render_pdf_to_images(pdf_path: Path, dpi: int = 600) -> List[Tuple[int, np.ndarray]]:
	"""Render PDF pages to high-resolution images.

	Args:
		pdf_path: Path to PDF file
		dpi: Resolution for rendering (default 600)

	Returns:
		List of (page_index, image_array) tuples
	"""
	processor = PDFProcessor()
	return processor.render_pdf_to_images(pdf_path, dpi)


def extract_vector_lines_from_pdf(pdf_path: Path, page_index: int, dpi: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""Extract vector line segments directly from PDF page drawings and map to pixel coords.

	Note: Works only for vector PDFs. For raster-only PDFs returns empty list.
	"""
	segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	try:
		zoom = float(dpi) / 72.0
		doc = fitz.open(pdf_path.as_posix())
		page = doc.load_page(page_index)
		paths = page.get_drawings()
		for path in paths:
			items = path.get("items", [])
			last_pt = None
			for it in items:
				op = it[0]
				pts = it[1]
				if op == "m":  # move to
					x, y = pts
					last_pt = (x * zoom, y * zoom)
				elif op == "l" and last_pt is not None:  # line to
					x, y = pts
					x0, y0 = last_pt
					x1, y1 = (x * zoom, y * zoom)
					segments.append(((int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))))
					last_pt = (x1, y1)
				elif op == "re":  # rectangle: x, y, w, h
					x, y, w, h = pts
					x0 = x * zoom; y0 = y * zoom
					x1 = (x + w) * zoom; y1 = (y + h) * zoom
					segments.append(((int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y0)))))
					segments.append(((int(round(x1)), int(round(y0))), (int(round(x1)), int(round(y1)))))
					segments.append(((int(round(x1)), int(round(y1))), (int(round(x0)), int(round(y1)))))
					segments.append(((int(round(x0)), int(round(y1))), (int(round(x0)), int(round(y0)))))
		doc.close()
		return segments
	except Exception:
		return []


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
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
			ss = SS[y1, x1] - SS[y0, x1] - SS[y1, x0] + SS[y0, x0]
			mu = s / win
			var[y, x] = max(ss / win - mu * mu, 0.0)
	std = np.sqrt(var)
	R = 128.0
	th = mean * (1 + k * ((std / R) - 1))
	bin_img = (gray.astype(np.float32) > th).astype(np.uint8) * 255
	return bin_img


def simple_binarize(gray: np.ndarray) -> np.ndarray:
	# Fallback to adaptive threshold if Sauvola is expensive
	return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 10)


Rect = Tuple[int, int, int, int]


def detect_text_regions_mser(gray: np.ndarray) -> List[Rect]:
	# MSER-based region proposal for text-like blobs
	mser = cv2.MSER_create(delta=5, min_area=60, max_area=20000)
	regions, _ = mser.detectRegions(gray)
	rects: List[Rect] = []
	for pts in regions:
		pts = np.array(pts).reshape(-1, 1, 2)
		x, y, w, h = cv2.boundingRect(pts)
		rects.append((x, y, w, h))

	# Geometry filtering
	H, W = gray.shape[:2]
	filtered: List[Rect] = []
	for x, y, w, h in rects:
		if w < 8 or h < 8:
			continue
		if w / h > 20 or h / w > 10:
			continue
		if (w * h) > (W * H * 0.1):
			continue
		filtered.append((x, y, w, h))
	return filtered


def group_text_rects(rects: List[Rect], image_shape: Tuple[int, int]) -> List[Rect]:
	# Morphological grouping: draw boxes on mask, dilate, then re-extract groups
	H, W = image_shape
	mask = np.zeros((H, W), dtype=np.uint8)
	for x, y, w, h in rects:
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
	# Dilation kernel encourages merging nearby characters into line blocks
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
	dil = cv2.dilate(mask, kernel, iterations=1)
	contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	grouped: List[Rect] = []
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		grouped.append((x, y, w, h))
	return grouped


def classify_text_groups_by_stroke(bin_img: np.ndarray, groups: List[Rect]) -> Tuple[List[Rect], List[Rect]]:
	# bin_img: 0 for text (dark), 255 for background
	if not groups:
		return [], []
	ratios: List[float] = []
	for x, y, w, h in groups:
		roi = bin_img[y : y + h, x : x + w]
		if roi.size == 0:
			ratios.append(1.0)
			continue
		# Invert so text becomes foreground (non-zero)
		roi_text = cv2.bitwise_not(roi)
		# Keep only text pixels (binary mask 0/255)
		_, roi_text = cv2.threshold(roi_text, 0, 255, cv2.THRESH_BINARY)
		# Distance to background approximates half stroke width at skeleton
		dt = cv2.distanceTransform(roi_text, cv2.DIST_L2, 3)
		vals = dt[roi_text > 0]
		if vals.size < 10:
			ratios.append(1.0)
			continue
		stroke_mean = float(np.mean(vals))
		stroke_std = float(np.std(vals))
		ratio = stroke_std / (stroke_mean + 1e-6)
		ratios.append(ratio)
	# Adaptive threshold by median of dataset
	median_ratio = float(np.median(ratios))
	printed: List[Rect] = []
	handwritten: List[Rect] = []
	for rect, ratio in zip(groups, ratios):
		if ratio <= median_ratio:
			printed.append(rect)
		else:
			handwritten.append(rect)
	return printed, handwritten


def draw_text_overlays(image_bgr: np.ndarray, printed: List[Rect], handwritten: List[Rect]) -> np.ndarray:
	over = image_bgr.copy()
	for x, y, w, h in printed:
		cv2.rectangle(over, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green
	for x, y, w, h in handwritten:
		cv2.rectangle(over, (x, y), (x + w, y + h), (0, 165, 255), 2)  # orange
	return over


def make_text_masks(shape: Tuple[int, int], printed: List[Rect], handwritten: List[Rect]) -> Tuple[np.ndarray, np.ndarray]:
	H, W = shape
	printed_mask = np.zeros((H, W), dtype=np.uint8)
	hand_mask = np.zeros((H, W), dtype=np.uint8)
	for x, y, w, h in printed:
		cv2.rectangle(printed_mask, (x, y), (x + w, y + h), 255, -1)
	for x, y, w, h in handwritten:
		cv2.rectangle(hand_mask, (x, y), (x + w, y + h), 255, -1)
	# Light dilation to avoid edge residue
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	printed_mask = cv2.dilate(printed_mask, kernel, iterations=1)
	hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
	return printed_mask, hand_mask


def build_text_mask_fast(gray: np.ndarray, downscale: float = 0.5) -> np.ndarray:
	"""Build a coarse text mask using MSER on downscaled image for speed, then upsample."""
	H, W = gray.shape[:2]
	ss = max(0.25, min(1.0, downscale))
	new_w = max(1, int(round(W * ss)))
	new_h = max(1, int(round(H * ss)))
	small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
	rects = detect_text_regions_mser(small)
	groups = group_text_rects(rects, (new_h, new_w))
	printed_mask_s, hand_mask_s = make_text_masks((new_h, new_w), groups, [])
	text_mask_s = cv2.bitwise_or(printed_mask_s, hand_mask_s)
	text_mask = cv2.resize(text_mask_s, (W, H), interpolation=cv2.INTER_NEAREST)
	return text_mask


def build_handwriting_masks_fast(gray: np.ndarray, downscale: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
	"""Build handwriting and printed masks using downscaled MSER + stroke classification.

	Returns (hand_mask, printed_mask) at full resolution.
	"""
	H, W = gray.shape[:2]
	ss = max(0.25, min(1.0, downscale))
	new_w = max(1, int(round(W * ss)))
	new_h = max(1, int(round(H * ss)))
	small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
	rects = detect_text_regions_mser(small)
	groups = group_text_rects(rects, (new_h, new_w))
	# Binarize small for stroke analysis: text ~ 0, background ~ 255
	_, small_bin = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	printed, handwritten = classify_text_groups_by_stroke(small_bin, groups)
	printed_mask_s, hand_mask_s = make_text_masks((new_h, new_w), printed, handwritten)
	printed_mask = cv2.resize(printed_mask_s, (W, H), interpolation=cv2.INTER_NEAREST)
	hand_mask = cv2.resize(hand_mask_s, (W, H), interpolation=cv2.INTER_NEAREST)
	return hand_mask, printed_mask


def build_handwriting_color_mask(image_bgr: np.ndarray, sat_thr: int = 60, diff_thr: int = 35) -> np.ndarray:
	"""Extract colored handwriting via colorfulness (HSV saturation and channel spread).

	Returns single-channel mask (255=handwriting candidate).
	"""
	try:
		# 파라미터 타입 안전성 보장
		sat_thr = int(float(sat_thr)) if not isinstance(sat_thr, int) else sat_thr
		diff_thr = int(float(diff_thr)) if not isinstance(diff_thr, int) else diff_thr
		
		# HSV-based saturation mask
		hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
		s = hsv[:, :, 1]
		v = hsv[:, :, 2]
		mask_sat = (s > sat_thr).astype(np.uint8) * 255
		# Channel spread mask (non-gray)
		b, g, r = cv2.split(image_bgr)
		mx = cv2.max(cv2.max(r, g), b)
		mn = cv2.min(cv2.min(r, g), b)
		spread = cv2.subtract(mx, mn)
		mask_diff = (spread > diff_thr).astype(np.uint8) * 255
		mask = cv2.bitwise_and(mask_sat, mask_diff)
		# Remove background highlights (very bright)
		mask[v > 245] = 0
		# Clean small noise
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
		mask = cv2.dilate(mask, kernel, iterations=1)
		return mask
	except Exception:
		return np.zeros(image_bgr.shape[:2], dtype=np.uint8)


def build_handwriting_strict_mask(image_bgr: np.ndarray, gray_rot: np.ndarray) -> np.ndarray:
	"""Strict handwriting mask using chroma + saturation + edge gating and HV-line suppression.

	- LAB chroma magnitude threshold (adaptive)
	- HSV saturation & channel spread
	- Edge gating to keep thin strokes
	- Remove horizontal/vertical structural lines
	- Adaptive tightening if mask coverage too large
	"""
	H, W = gray_rot.shape[:2]
	# Chroma magnitude
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	L, a, b = cv2.split(lab)
	chroma = cv2.magnitude(a.astype(np.float32) - 128.0, b.astype(np.float32) - 128.0)
	chroma = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	thr_ch = max(15, int(cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))
	mask_ch = (chroma > thr_ch).astype(np.uint8) * 255
	# HSV saturation + channel spread
	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	mask_sat = (hsv[:, :, 1] > 60).astype(np.uint8) * 255
	bgr_max = cv2.max(cv2.max(image_bgr[:, :, 0], image_bgr[:, :, 1]), image_bgr[:, :, 2])
	bgr_min = cv2.min(cv2.min(image_bgr[:, :, 0], image_bgr[:, :, 1]), image_bgr[:, :, 2])
	mask_spread = (cv2.subtract(bgr_max, bgr_min) > 30).astype(np.uint8) * 255
	# Edge gating
	edges = cv2.Canny(gray_rot, 80, 180, apertureSize=3)
	edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
	mask_color = cv2.bitwise_and(cv2.bitwise_and(mask_ch, mask_sat), mask_spread)
	mask = cv2.bitwise_and(mask_color, edges)
	# Remove HV structural lines
	min_dim = min(H, W)
	k = max(15, int(min_dim * 0.02))
	h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
	v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
	hv_h = cv2.morphologyEx(gray_rot, cv2.MORPH_OPEN, h_kernel, iterations=1)
	hv_v = cv2.morphologyEx(gray_rot, cv2.MORPH_OPEN, v_kernel, iterations=1)
	hv = cv2.max(hv_h, hv_v)
	_, hv_bin = cv2.threshold(hv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	mask[hv_bin > 0] = 0
	# Clean noise and enforce coverage limits
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
	coverage = float(np.count_nonzero(mask)) / float(H * W)
	if coverage > 0.2:
		# Too broad → tighten thresholds
		mask = cv2.bitwise_and(mask, (chroma > thr_ch + 10).astype(np.uint8) * 255)
		mask = cv2.bitwise_and(mask, (hsv[:, :, 1] > 80).astype(np.uint8) * 255)
		mask = cv2.bitwise_and(mask, (cv2.subtract(bgr_max, bgr_min) > 45).astype(np.uint8) * 255)
		mask = cv2.bitwise_and(mask, edges)
		mask[hv_bin > 0] = 0
	return mask


def build_printed_mask(
    image_bgr: np.ndarray,
    gray_rot: np.ndarray,
    sat_max: int = 50,
    chroma_max: int = 25,
    min_area: int = 16,
    max_area_ratio: float = 0.02,
    max_aspect_ratio: float = 7.0,
    min_extent: float = 0.15,
    max_extent: float = 0.85,
    remove_lines: bool = True,
) -> np.ndarray:
	"""Extract printed (mono/low-saturation) text mask.

	- Low saturation (HSV S <= sat_max)
	- Low chroma (LAB chroma <= chroma_max)
	- Edge gating to keep strokes
	- Remove structural lines (H/V/diag, Hough-based)
	- Connected-component geometry filter (문자 형태만 유지)
	"""
	H, W = gray_rot.shape[:2]
	# HSV saturation low
	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	mask_sat_low = (hsv[:, :, 1] <= int(float(sat_max))).astype(np.uint8) * 255
	# LAB chroma low
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	_, a, b = cv2.split(lab)
	chroma = cv2.magnitude(a.astype(np.float32) - 128.0, b.astype(np.float32) - 128.0)
	chroma = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	mask_chroma_low = (chroma <= int(float(chroma_max))).astype(np.uint8) * 255
	# Edge gating for thin strokes
	edges = cv2.Canny(gray_rot, 70, 170, apertureSize=3)
	edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
	mask = cv2.bitwise_and(mask_sat_low, mask_chroma_low)
	mask = cv2.bitwise_and(mask, edges)
	# Remove structural lines (build line mask from bin + Hough + morph diag)
	if remove_lines:
		try:
			bin_img = simple_binarize(gray_rot)
			min_len = max(10.0, min(H, W) * 0.01)
			segs_h = detect_lines_houghp(bin_img, min_len=min_len)
			segs_m = detect_lines_morph_hv_diag(bin_img)
			line_mask = np.zeros((H, W), dtype=np.uint8)
			for (x0, y0), (x1, y1) in (segs_h + segs_m):
				cv2.line(line_mask, (x0, y0), (x1, y1), 255, 3, lineType=cv2.LINE_AA)
			line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
			mask[line_mask > 0] = 0
		except Exception:
			pass
	# Connected component filter to drop long lines/frames; keep character-like
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	keep = np.zeros_like(mask)
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		bbox_area = w * h
		if bbox_area < int(min_area):
			continue
		# Discard huge boxes (tables/frames)
		if bbox_area > (W * H * float(max_area_ratio)):
			continue
		# Aspect ratio filter (avoid long lines)
		ar = max(w, h) / max(1, min(w, h))
		if ar > float(max_aspect_ratio):
			continue
		# Extent (foreground area / bbox area)
		cnt_area = max(1.0, cv2.contourArea(c))
		extent = cnt_area / float(bbox_area)
		if extent < float(min_extent) or extent > float(max_extent):
			continue
		cv2.drawContours(keep, [c], -1, 255, -1)
	# Final cleanup
	keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
	return keep

def detect_lines_lsd(gray_bin: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""Detect lines using Line Segment Detector (LSD) algorithm.
	
	Args:
		gray_bin: Binary or grayscale image
		
	Returns:
		List of line segments as ((x0, y0), (x1, y1)) tuples
	"""
	try:
		# Use default parameters for broad compatibility across OpenCV builds
		lsd = cv2.createLineSegmentDetector()
		lines, _, _, _ = lsd.detect(gray_bin)
		segments = []
		
		if lines is not None:
			for l in lines:
				x0, y0, x1, y1 = map(int, l[0])
				segments.append(((x0, y0), (x1, y1)))
			logger.debug(f"LSD detected {len(segments)} lines")
		else:
			logger.debug("LSD detected no lines")
		
		return segments
		
	except cv2.error as e:
		logger.warning(f"LSD not available (opencv-contrib-python required): {e}")
		return []


def detect_lines_lsd_adv_multiscale(gray_bin: np.ndarray, scales: List[float] = [0.5, 1.0]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""Advanced LSD with refinement and multi-scale merging."""
	all_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	H, W = gray_bin.shape[:2]
	for s in scales:
		if s <= 0:
			continue
		new_w = max(1, int(round(W * s)))
		new_h = max(1, int(round(H * s)))
		scaled = cv2.resize(gray_bin, (new_w, new_h), interpolation=cv2.INTER_AREA)
		try:
			lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
		except Exception:
			lsd = cv2.createLineSegmentDetector()
		lines, _, _, _ = lsd.detect(scaled)
		if lines is not None:
			for l in lines:
				x0, y0, x1, y1 = map(int, l[0])
				# scale back to original coords
				x0 = int(round(x0 / s))
				y0 = int(round(y0 / s))
				x1 = int(round(x1 / s))
				y1 = int(round(y1 / s))
				all_segments.append(((x0, y0), (x1, y1)))
	# NMS-like dedup
	merged: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	for seg in all_segments:
		if not any(_are_segments_similar(seg, m) for m in merged):
			merged.append(seg)
	return merged


def detect_lines_fld(gray_or_bin: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""FastLineDetector (OpenCV ximgproc). Returns empty if not available."""
	try:
		from cv2 import ximgproc  # type: ignore
	except Exception:
		return []
	if gray_or_bin.ndim == 3:
		gray = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
	else:
		gray = gray_or_bin
	try:
		fld = ximgproc.createFastLineDetector()
		lines = fld.detect(gray)
		segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
		if lines is None:
			return []
		for l in lines:
			x0, y0, x1, y1 = map(int, l[0]) if hasattr(l, '__len__') and len(l) == 1 else map(int, l)
			segments.append(((x0, y0), (x1, y1)))
		return segments
	except Exception:
		return []


def filter_lines(segments: List[Tuple[Tuple[int, int], Tuple[int, int]]], min_len: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	filtered = []
	for (x0, y0), (x1, y1) in segments:
		length = math.hypot(x1 - x0, y1 - y0)
		if length >= min_len:
			filtered.append(((x0, y0), (x1, y1)))
	return filtered


def overlay_lines(image_bgr: np.ndarray, segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
	"""Draw red lines on image. Returns BGR image for OpenCV."""
	over = image_bgr.copy()
	for (x0, y0), (x1, y1) in segments:
		# OpenCV uses BGR: (0, 0, 255) = Red in BGR format
		cv2.line(over, (x0, y0), (x1, y1), (0, 0, 255), 2, lineType=cv2.LINE_AA)
	return over


def export_lines_json(segments: list[tuple[tuple[int, int], tuple[int, int]]], out_path: Path) -> None:
	data = [
		{"x0": x0, "y0": y0, "x1": x1, "y1": y1}
		for (x0, y0), (x1, y1) in segments
	]
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump({"lines": data}, f, ensure_ascii=False, indent=2)


def detect_lines_houghp(gray_or_bin: np.ndarray, min_len: float, max_gap: int = 4) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	# Use Canny edges then probabilistic Hough
	if gray_or_bin.ndim == 3:
		gray = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
	else:
		gray = gray_or_bin
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	lines = cv2.HoughLinesP(
		edges,
		rho=1,
		theta=np.pi / 180.0,
		threshold=50,
		minLineLength=int(max(1, round(min_len))),
		maxLineGap=max_gap,
	)
	segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
	if lines is not None:
		for l in lines:
			x0, y0, x1, y1 = map(int, l[0])
			segments.append(((x0, y0), (x1, y1)))
	return segments


def detect_lines_morph_hv(bin_img: np.ndarray, kernel_ratio: float = 0.02) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	# Isolate long horizontal/vertical structures then HoughP
	H, W = bin_img.shape[:2]
	min_dim = min(H, W)
	k = max(15, int(min_dim * kernel_ratio))
	# Binarized image may be white=foreground; invert to get lines prominent
	inv = cv2.bitwise_not(bin_img)
	h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
	v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
	h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
	v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
	line_map = cv2.bitwise_or(h_lines, v_lines)
	segments = detect_lines_houghp(line_map, min_len=max(10, int(min_dim * 0.01)), max_gap=3)
	return segments


def detect_lines_morph_hv_diag(bin_img: np.ndarray, kernel_ratio: float = 0.02) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	# H/V + 45/135° diagonal morphological enhancement then HoughP
	H, W = bin_img.shape[:2]
	min_dim = min(H, W)
	k = max(15, int(min_dim * kernel_ratio))
	inv = cv2.bitwise_not(bin_img)
	h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
	v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
	base = np.zeros((k, k), dtype=np.uint8)
	cv2.line(base, (0, k // 2), (k - 1, k // 2), 255, 1)
	M45 = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), 45.0, 1.0)
	M135 = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), 135.0, 1.0)
	d1_kernel = cv2.warpAffine(base, M45, (k, k))
	d2_kernel = cv2.warpAffine(base, M135, (k, k))
	d1_kernel = cv2.threshold(d1_kernel, 127, 255, cv2.THRESH_BINARY)[1]
	d2_kernel = cv2.threshold(d2_kernel, 127, 255, cv2.THRESH_BINARY)[1]

	h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
	v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
	d1_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, d1_kernel, iterations=1)
	d2_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, d2_kernel, iterations=1)
	line_map = cv2.bitwise_or(cv2.bitwise_or(h_lines, v_lines), cv2.bitwise_or(d1_lines, d2_lines))
	segments = detect_lines_houghp(line_map, min_len=max(10, int(min_dim * 0.01)), max_gap=3)
	return segments
def detect_lines_houghp_with_params(
	gray_or_bin: np.ndarray,
	min_len: float,
	can_low: int,
	can_high: int,
	thr: int,
	max_gap: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""HoughP with explicit parameters."""
	if gray_or_bin.ndim == 3:
		gray = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
	else:
		gray = gray_or_bin
	edges = cv2.Canny(gray, can_low, can_high, apertureSize=3)
	lines = cv2.HoughLinesP(
		edges,
		rho=1,
		theta=np.pi / 180.0,
		threshold=int(thr),
		minLineLength=int(max(1, round(min_len))),
		maxLineGap=int(max_gap),
	)
	segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
	if lines is not None:
		for l in lines:
			x0, y0, x1, y1 = map(int, l[0])
			segments.append(((x0, y0), (x1, y1)))
	return segments


def _scale_segments_uniform(
	segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
	scale_x: float,
	scale_y: float,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	res: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	for (x0, y0), (x1, y1) in segments:
		x0n = int(round(x0 * scale_x))
		y0n = int(round(y0 * scale_y))
		x1n = int(round(x1 * scale_x))
		y1n = int(round(y1 * scale_y))
		res.append(((x0n, y0n), (x1n, y1n)))
	return res


def detect_lines_houghp_multiscale(
	bin_img: np.ndarray,
	min_len: float,
	scales: List[float] = [0.5, 1.0],
	can_low: int = 50,
	can_high: int = 150,
	thr: int = 50,
	max_gap: int = 4,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""Run HoughP at multiple scales and merge results back to original size."""
	H, W = bin_img.shape[:2]
	all_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	for s in scales:
		if s <= 0:
			continue
		new_w = max(1, int(round(W * s)))
		new_h = max(1, int(round(H * s)))
		scaled = cv2.resize(bin_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
		scaled_min_len = max(1.0, min_len * s)
		segs = detect_lines_houghp_with_params(
			scaled, scaled_min_len, can_low, can_high, thr, max_gap
		)
		# scale back
		back = _scale_segments_uniform(segs, 1.0 / s, 1.0 / s)
		all_segments.extend(back)
	# light dedup by NMS-like merge using existing helper
	merged: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	for seg in all_segments:
		if not any(_are_segments_similar(seg, m) for m in merged):
			merged.append(seg)
	return merged


def detect_lines_houghp_autotune(
	bin_img: np.ndarray,
	min_len: float,
	sample_scale: float = 0.5,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""Lightweight grid search on downscaled image to pick HoughP params, then run full."""
	H, W = bin_img.shape[:2]
	# Downscale for quick scoring
	ss = max(0.2, min(1.0, sample_scale))
	new_w = max(1, int(round(W * ss)))
	new_h = max(1, int(round(H * ss)))
	small = cv2.resize(bin_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
	min_len_small = max(1.0, min_len * ss)

	can_lows = [30, 50, 70]
	thr_vals = [30, 50, 80]
	max_gaps = [3, 6]

	best = None  # (score, params)
	for low in can_lows:
		high = min(300, int(low * 3))
		for thr in thr_vals:
			for gap in max_gaps:
				segs = detect_lines_houghp_with_params(small, min_len_small, low, high, thr, gap)
				n = len(segs)
				if n == 0:
					score = 0.0
				else:
					# target around few hundreds on downscaled
					target = 300.0
					closeness = max(0.0, 1.0 - abs(n - target) / target)
					mean_len = float(np.mean([_segment_length(s) for s in segs])) if segs else 0.0
					mean_len_norm = min(1.0, (mean_len / float(min(new_h, new_w) + 1e-6)))
					score = 0.8 * closeness + 0.2 * mean_len_norm
				if best is None or score > best[0]:
					best = (score, (low, high, thr, gap))

	# Run with best params on full image
	if best is None:
		return detect_lines_houghp(bin_img, min_len=min_len)
	low, high, thr, gap = best[1]
	return detect_lines_houghp_with_params(bin_img, min_len, low, high, thr, gap)


def _segment_length(seg: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
	(x0, y0), (x1, y1) = seg
	return float(math.hypot(x1 - x0, y1 - y0))


def _segment_angle_deg(seg: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
	(x0, y0), (x1, y1) = seg
	return float(abs(math.degrees(math.atan2(y1 - y0, x1 - x0))))


def _sample_text_overlap_fraction(seg: Tuple[Tuple[int, int], Tuple[int, int]], text_mask: Optional[np.ndarray], num_samples: int = 25) -> float:
	"""Estimate fraction of points on the line that intersect text mask (255=mask)."""
	if text_mask is None:
		return 0.0
	(x0, y0), (x1, y1) = seg
	H, W = text_mask.shape[:2]
	over = 0
	for i in range(num_samples):
		t = i / max(1, (num_samples - 1))
		x = int(round(x0 + (x1 - x0) * t))
		y = int(round(y0 + (y1 - y0) * t))
		if 0 <= x < W and 0 <= y < H and text_mask[y, x] > 0:
			over += 1
	return over / float(num_samples)


def _are_segments_similar(a: Tuple[Tuple[int, int], Tuple[int, int]], b: Tuple[Tuple[int, int], Tuple[int, int]], angle_thr: float = 5.0, dist_thr: float = 8.0) -> bool:
	"""Check if two segments are essentially the same based on angle and midpoint distance."""
	ang_a = _segment_angle_deg(a)
	ang_b = _segment_angle_deg(b)
	if min(abs(ang_a - ang_b), 180 - abs(ang_a - ang_b)) > angle_thr:
		return False
	(ax0, ay0), (ax1, ay1) = a
	(bx0, by0), (bx1, by1) = b
	ma = ((ax0 + ax1) * 0.5, (ay0 + ay1) * 0.5)
	mb = ((bx0 + bx1) * 0.5, (by0 + by1) * 0.5)
	d = math.hypot(ma[0] - mb[0], ma[1] - mb[1])
	return d <= dist_thr


def fuse_segments(
	segments_by_source: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
	image_shape: Tuple[int, int],
	text_mask: Optional[np.ndarray] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""Fuse segments from multiple detectors using simple scoring and NMS.

	Scoring:
	- Base: normalized length (by min(H, W))
	- Source weight: lsd=1.0, hough=0.95, morph=0.9
	- Text overlap penalty: 0.3 * overlap_fraction
	"""
	source_weight = {"lsd": 1.0, "hough": 0.95, "morph": 0.9, "hough_ms": 0.97, "hough_auto": 0.98}
	H, W = image_shape
	min_dim = max(1.0, float(min(H, W)))
	all_items: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], float]] = []
	for src, segs in segments_by_source.items():
		w = source_weight.get(src, 0.9)
		for seg in segs:
			length_norm = _segment_length(seg) / min_dim
			overlap_frac = _sample_text_overlap_fraction(seg, text_mask) if text_mask is not None else 0.0
			score = 0.6 * length_norm + 0.4 * w - 0.3 * overlap_frac
			all_items.append((seg, score))
	# Sort by score desc
	all_items.sort(key=lambda x: x[1], reverse=True)
	kept: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	for seg, _ in all_items:
		if not any(_are_segments_similar(seg, k) for k in kept):
			kept.append(seg)
	return kept


def _farthest_pair(points: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
	"""Return two points with maximum pairwise distance (O(n^2), small n)."""
	best = (points[0], points[0])
	best_d = -1.0
	for i in range(len(points)):
		for j in range(i + 1, len(points)):
			d = math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
			if d > best_d:
				best = (points[i], points[j])
				best_d = d
	return best


def merge_collinear_segments(
	segments: List[Tuple[Tuple[int, int], Tuple[int, int]]], angle_thr: float = 5.0, dist_thr: float = 8.0
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
	"""Greedy merge of collinear and nearby segments by extending to farthest endpoints."""
	merged: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
	for seg in sorted(segments, key=_segment_length, reverse=True):
		merged_any = False
		for idx, keep in enumerate(merged):
			if _are_segments_similar(seg, keep, angle_thr=angle_thr, dist_thr=dist_thr):
				points = [keep[0], keep[1], seg[0], seg[1]]
				n0, n1 = _farthest_pair(points)
				merged[idx] = (n0, n1)
				merged_any = True
				break
		if not merged_any:
			merged.append(seg)
	return merged


def process_page_lines(page_data: Dict[str, Any]) -> Dict[str, Any]:
	"""Process lines for a single page (for parallel processing).
	
	Args:
		page_data: Dictionary containing page processing data
		
	Returns:
		Dictionary with processing results
	"""
	page_idx = page_data['page_idx']
	img_bgr = page_data['img_bgr']
	page_dir = page_data['page_dir']
	
	try:
		logger.info(f"Processing page {page_idx}")
		
		# Save original
		cv2.imwrite((page_dir / "original.png").as_posix(), img_bgr)
		
		# Convert to grayscale and deskew
		gray = to_gray(img_bgr)
		rot, angle = deskew_by_hough(gray)
		cv2.imwrite((page_dir / "deskew_gray.png").as_posix(), rot)
		
		# Binarize
		try:
			bin_img = simple_binarize(rot)
		except Exception:
			bin_img = cv2.threshold(rot, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		cv2.imwrite((page_dir / "binary.png").as_posix(), bin_img)

		# Printed-text removal for line detection
		try:
			pmask = build_printed_mask(img_bgr, rot)
			text_mask = pmask.copy()
			masked_bin = bin_img.copy()
			masked_bin[pmask > 0] = 255
			cv2.imwrite((page_dir / "binary_printless.png").as_posix(), masked_bin)
		except Exception:
			text_mask = None
			masked_bin = bin_img
		
		# Detect lines with multiple algorithms
		min_len = max(10.0, min(rot.shape[:2]) * 0.01)
		
		# Run detection algorithms
		segments_lsd = filter_lines(detect_lines_lsd(masked_bin), min_len=min_len)
		segments_lsd_adv = filter_lines(detect_lines_lsd_adv_multiscale(masked_bin, scales=[0.5, 1.0]), min_len=min_len)
		segments_hough = filter_lines(detect_lines_houghp(masked_bin, min_len=min_len), min_len=min_len)
		segments_hough_ms = filter_lines(detect_lines_houghp_multiscale(masked_bin, min_len=min_len, scales=[0.5, 1.0]), min_len=min_len)
		segments_hough_auto = filter_lines(detect_lines_houghp_autotune(masked_bin, min_len=min_len), min_len=min_len)
		segments_morph = filter_lines(detect_lines_morph_hv(masked_bin), min_len=min_len)
		segments_morph_diag = filter_lines(detect_lines_morph_hv_diag(masked_bin), min_len=min_len)
		segments_fld = filter_lines(detect_lines_fld(masked_bin), min_len=min_len)

		# Vector PDF lines (if any)
		vector_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
		try:
			from config import DEFAULT_PDF_PATH
			vector_segments = filter_lines(extract_vector_lines_from_pdf(DEFAULT_PDF_PATH, page_idx, dpi=PDF_CONFIG["default_dpi"]), min_len=min_len)
		except Exception:
			vector_segments = []

		# Fuse segments
		segments_fused = fuse_segments(
			{"lsd": segments_lsd, "lsd_adv": segments_lsd_adv, "hough": segments_hough, "hough_ms": segments_hough_ms, "hough_auto": segments_hough_auto, "morph": segments_morph, "morph_diag": segments_morph_diag, "fld": segments_fld, "vector": vector_segments},
			image_shape=rot.shape[:2],
			text_mask=text_mask,
		)
		# Post-merge to consolidate collinear segments
		segments_fused = merge_collinear_segments(segments_fused)
		
		# Save JSON results
		export_lines_json(segments_lsd, page_dir / "lines_lsd.json")
		export_lines_json(segments_hough, page_dir / "lines_hough.json")
		export_lines_json(segments_morph, page_dir / "lines_morph.json")
		export_lines_json(segments_hough_ms, page_dir / "lines_hough_ms.json")
		export_lines_json(segments_hough_auto, page_dir / "lines_hough_auto.json")
		export_lines_json(segments_lsd_adv, page_dir / "lines_lsd_adv.json")
		export_lines_json(segments_fld, page_dir / "lines_fld.json")
		export_lines_json(segments_morph_diag, page_dir / "lines_morph_diag.json")
		export_lines_json(segments_fused, page_dir / "lines_fused.json")
		if vector_segments:
			export_lines_json(vector_segments, page_dir / "lines_vector.json")
		
		# Create overlay visualizations with correct red color
		over_lsd = overlay_lines(img_bgr, segments_lsd)
		over_hough = overlay_lines(img_bgr, segments_hough)
		over_morph = overlay_lines(img_bgr, segments_morph)
		over_hough_ms = overlay_lines(img_bgr, segments_hough_ms)
		over_hough_auto = overlay_lines(img_bgr, segments_hough_auto)
		over_lsd_adv = overlay_lines(img_bgr, segments_lsd_adv)
		over_fld = overlay_lines(img_bgr, segments_fld)
		over_morph_diag = overlay_lines(img_bgr, segments_morph_diag)
		over_fused = overlay_lines(img_bgr, segments_fused)
		if vector_segments:
			over_vector = overlay_lines(img_bgr, vector_segments)
		
		cv2.imwrite((page_dir / "overlay_lines_lsd.png").as_posix(), over_lsd)
		cv2.imwrite((page_dir / "overlay_lines_hough.png").as_posix(), over_hough)
		cv2.imwrite((page_dir / "overlay_lines_morph.png").as_posix(), over_morph)
		cv2.imwrite((page_dir / "overlay_lines_hough_ms.png").as_posix(), over_hough_ms)
		cv2.imwrite((page_dir / "overlay_lines_hough_auto.png").as_posix(), over_hough_auto)
		cv2.imwrite((page_dir / "overlay_lines_lsd_adv.png").as_posix(), over_lsd_adv)
		cv2.imwrite((page_dir / "overlay_lines_fld.png").as_posix(), over_fld)
		cv2.imwrite((page_dir / "overlay_lines_morph_diag.png").as_posix(), over_morph_diag)
		cv2.imwrite((page_dir / "overlay_lines_fused.png").as_posix(), over_fused)
		if vector_segments:
			cv2.imwrite((page_dir / "overlay_lines_vector.png").as_posix(), over_vector)
		
		# Maintain compatibility files
		export_lines_json(segments_lsd, page_dir / "lines.json")
		cv2.imwrite((page_dir / "overlay_lines.png").as_posix(), over_lsd)
		
		logger.info(f"Page {page_idx} complete - LSD: {len(segments_lsd)}, LSDadv: {len(segments_lsd_adv)}, Hough: {len(segments_hough)}, HoughMS: {len(segments_hough_ms)}, HoughAuto: {len(segments_hough_auto)}, Morph: {len(segments_morph)}, MorphDiag: {len(segments_morph_diag)}, FLD: {len(segments_fld)}, Vector: {len(vector_segments)}, Fused: {len(segments_fused)} lines")
		
		return {
			'page_idx': page_idx,
			'success': True,
			'lines_count': {
				'lsd': len(segments_lsd),
				'hough': len(segments_hough),
				'morph': len(segments_morph)
			}
		}
		
	except Exception as e:
		logger.error(f"Failed to process page {page_idx}: {e}")
		return {'page_idx': page_idx, 'success': False, 'error': str(e)}

def process_sample(pdf_path: Path, out_dir: Path, dpi: int = 600, parallel: bool = True) -> None:
	"""Process PDF sample for line extraction.
	
	Args:
		pdf_path: Path to input PDF
		out_dir: Output directory for results
		dpi: Resolution for PDF rendering
		parallel: Use parallel processing for multiple pages
	"""
	logger.info(f"Starting processing: {pdf_path}")
	ensure_dir(out_dir)
	
	try:
		# Render PDF to images
		images = render_pdf_to_images(pdf_path, dpi=dpi)
		
		if not images:
			logger.error("No pages could be rendered from PDF")
			return
		
		# Prepare page data for processing
		page_data_list = []
		for page_idx, img_bgr in images:
			page_dir = out_dir / f"page_{page_idx:03d}"
			ensure_dir(page_dir)
			page_data_list.append({
				'page_idx': page_idx,
				'img_bgr': img_bgr,
				'page_dir': page_dir
			})
		
		# Process pages (parallel or sequential)
		if parallel and len(page_data_list) > 1:
			# 메모리 기반 워커 수 최적화
			memory_manager = MemoryManager(PERFORMANCE_CONFIG["memory_limit_mb"])
			optimal_workers = calculate_optimal_workers(
				PERFORMANCE_CONFIG["memory_limit_mb"],
				estimated_mb_per_worker=800  # 각 워커당 예상 메모리 사용량
			)
			max_workers = min(optimal_workers, len(page_data_list))
			logger.info(f"{len(page_data_list)} 페이지 병렬 처리 중 (최적 워커 수: {max_workers})")

			with ProcessPoolExecutor(max_workers=max_workers) as executor:
				futures = [executor.submit(process_page_lines, page_data) for page_data in page_data_list]
				for future in as_completed(futures):
					result = future.result()
					if result['success']:
						logger.info(f"Page {result['page_idx']} processed successfully")
					else:
						logger.error(f"Page {result['page_idx']} failed")
		else:
			logger.info(f"Processing {len(page_data_list)} pages sequentially")
			for page_data in page_data_list:
				process_page_lines(page_data)
		
		logger.info(f"Processing complete. Results saved to: {out_dir}")
		
	except Exception as e:
		logger.error(f"Processing failed: {e}")
		raise


def main():
	"""Main entry point for the line extraction tool."""
	try:
		from config import DEFAULT_PDF_PATH, OUTPUT_DIR
	except Exception:
		DEFAULT_PDF_PATH = config_module.DEFAULT_PDF_PATH
		OUTPUT_DIR = config_module.OUTPUT_DIR

	# Validate configuration using new validator
	logger.info("설정 검증 중...")
	is_valid, errors, warnings = ConfigValidator.validate_all()

	if warnings:
		for warning in warnings:
			logger.warning(f"설정 경고: {warning}")

	if not is_valid:
		for error in errors:
			logger.error(f"설정 오류: {error}")
		logger.error("설정 검증 실패로 프로그램을 종료합니다")
		sys.exit(1)

	logger.info("✅ 모든 설정 검증을 통과했습니다")
	
	# Find PDF files
	pdf_files = get_pdf_files()
	if not pdf_files:
		logger.error("No PDF files found in the project directory")
		sys.exit(1)
	
	pdf_path = pdf_files[0]  # Use first available PDF
	if pdf_path != DEFAULT_PDF_PATH:
		logger.info(f"Using alternative PDF: {pdf_path}")
	
	try:
		# Process with configuration settings
		process_sample(
			pdf_path, 
			OUTPUT_DIR, 
			dpi=PDF_CONFIG["default_dpi"], 
			parallel=PDF_CONFIG["parallel_processing"]
		)
		logger.info(f"✅ Processing complete! Results in: {OUTPUT_DIR}")
		
	except KeyboardInterrupt:
		logger.info("Processing interrupted by user")
		sys.exit(0)
	except Exception as e:
		logger.error(f"Fatal error: {e}")
		sys.exit(1)

if __name__ == "__main__":
	main()


