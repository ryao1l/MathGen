#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
PROMPT_ID = 2
BLUE_H_LOW, BLUE_H_HIGH = (90, 130)
BLUE_S_MIN = 50
BLUE_V_MIN = 50
RATIO_TARGET = 0.25
RATIO_TOL = 0.2
MIN_BAR_ASPECT = 2.0
MIN_BLUE_PURITY = 0.25
MAX_BLUE_IN_RIGHT = 0.1
CANNY1, CANNY2 = (40, 140)
DILATE_K = 5
MIN_AREA_FRAC = 0.02
INSET_FRAC = 0.05

def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from disk (BGR). Return None on failure."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        composited = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        return composited
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def blue_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of blue pixels in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([BLUE_H_LOW, BLUE_S_MIN, BLUE_V_MIN])
    upper = np.array([BLUE_H_HIGH, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def find_bar_bbox(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find a horizontal bar region using edges + contour analysis.
    Returns (x, y, w, h) or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY1, CANNY2)
    k = DILATE_K if DILATE_K % 2 == 1 else DILATE_K + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edges2 = cv2.dilate(edges, kernel, iterations=2)
    edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best_bbox = None
    best_score = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < MIN_AREA_FRAC * (H * W):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        aspect = w / h
        if aspect < MIN_BAR_ASPECT:
            continue
        score = area * aspect ** 0.5
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)
    return best_bbox

def find_bar_bbox_by_blue(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Fallback: find the bar region by looking at the bounding box of blue pixels
    and expanding it to include the full bar (assuming bar extends rightward).
    """
    bm = blue_mask_hsv(img)
    H, W = img.shape[:2]
    coords = cv2.findNonZero(bm)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_center = y + h // 2
    row_band = gray[max(0, row_center - h // 4):min(H, row_center + h // 4), :]
    mean_intensity = row_band.mean(axis=0)
    bar_right = W
    bg_intensity = float(np.median(gray[:5, :]))
    for col in range(x + w, W):
        if abs(float(mean_intensity[col]) - bg_intensity) < 15:
            bar_right = col
            break
    pad = 5
    bx = max(0, x - pad)
    by = max(0, y - pad)
    bw = min(W - bx, bar_right - bx + pad)
    bh = min(H - by, h + 2 * pad)
    if bw < 20 or bh < 10:
        return None
    return (bx, by, bw, bh)

def inset_region(img: np.ndarray) -> np.ndarray:
    """Remove a small border by insetting the image."""
    H, W = img.shape[:2]
    inset = max(1, int(round(min(H, W) * INSET_FRAC)))
    x0, y0 = (inset, inset)
    x1, y1 = (W - inset, H - inset)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return img
    return img[y0:y1, x0:x1]

def measure_blue_fill_ratio(bar_img: np.ndarray) -> Tuple[float, float, float]:
    """
    Measure the blue fill ratio within the bar image.

    Returns:
      - blue_ratio: fraction of bar width that is blue
      - left_blue_purity: fraction of blue pixels in the left quarter
      - right_blue_purity: fraction of blue pixels in the right 75%
    """
    bar_inset = inset_region(bar_img)
    H, W = bar_inset.shape[:2]
    bm = blue_mask_hsv(bar_inset)
    total_pixels = H * W
    if total_pixels == 0:
        return (0.0, 0.0, 0.0)
    col_blue = bm.mean(axis=0) / 255.0
    blue_threshold = 0.15
    blue_cols = np.where(col_blue > blue_threshold)[0]
    if len(blue_cols) == 0:
        return (0.0, 0.0, 0.0)
    rightmost_blue = blue_cols[-1]
    blue_ratio = (rightmost_blue + 1) / W
    split = int(W * 0.35)
    left_part = bm[:, :split]
    right_part = bm[:, split:]
    left_blue_purity = float(np.sum(left_part > 0)) / float(left_part.size) if left_part.size > 0 else 0.0
    right_blue_purity = float(np.sum(right_part > 0)) / float(right_part.size) if right_part.size > 0 else 0.0
    return (blue_ratio, left_blue_purity, right_blue_purity)

def check_unfilled_region(bar_img: np.ndarray) -> bool:
    """
    Check that the right portion of the bar (unfilled) is gray/white
    (i.e., not blue and not heavily colored).
    """
    bar_inset = inset_region(bar_img)
    H, W = bar_inset.shape[:2]
    split = int(W * 0.4)
    right_portion = bar_inset[:, split:]
    if right_portion.size == 0:
        return False
    bm = blue_mask_hsv(right_portion)
    blue_frac = float(np.sum(bm > 0)) / float(bm.size)
    return blue_frac < MAX_BLUE_IN_RIGHT

def check_labels(img: np.ndarray) -> bool:
    """
    Check for the presence of dark text below the bar (labels like 0%, 25%, 100%).
    Uses a heuristic: look for dark pixel clusters in the bottom portion of the image.
    """
    H, W = img.shape[:2]
    bottom = img[int(H * 0.55):, :]
    if bottom.size == 0:
        return False
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    _, text_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    col_density = text_mask.mean(axis=0) / 255.0
    kernel_size = max(1, W // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(col_density, np.ones(kernel_size) / kernel_size, mode='same')
    threshold = 0.02
    above = smoothed > threshold
    transitions = np.diff(above.astype(int))
    num_clusters = int(np.sum(transitions == 1))
    return num_clusters >= 2

def check_background_is_white(image_path: str, threshold: int=240) -> Tuple[bool, float]:
    """Check if the image background (border regions) is predominantly white and opaque."""
    if not os.path.isfile(image_path):
        return (False, 0.0)
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return (False, 0.0)
    H, W = raw_img.shape[:2]
    border = max(1, int(min(H, W) * 0.08))
    top = raw_img[:border, :]
    bottom = raw_img[H - border:, :]
    left = raw_img[:, :border]
    right = raw_img[:, W - border:]
    if raw_img.ndim == 3 and raw_img.shape[2] == 4:
        alpha = raw_img[:, :, 3]
        top_a = alpha[:border, :]
        bottom_a = alpha[H - border:, :]
        left_a = alpha[:, :border]
        right_a = alpha[:, W - border:]
        alpha_border = np.concatenate([top_a.ravel(), bottom_a.ravel(), left_a.ravel(), right_a.ravel()])
        mean_alpha = float(np.mean(alpha_border))
        if mean_alpha < 200:
            return (False, mean_alpha)
        img_check = raw_img[:, :, :3]
    else:
        img_check = raw_img
    if img_check.ndim == 3:
        gray = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_check
    top_g = gray[:border, :]
    bottom_g = gray[H - border:, :]
    left_g = gray[:, :border]
    right_g = gray[:, W - border:]
    intensity_border = np.concatenate([top_g.ravel(), bottom_g.ravel(), left_g.ravel(), right_g.ravel()])
    median_intensity = float(np.median(intensity_border))
    return (median_intensity >= threshold, median_intensity)

def evaluate(image_path: str, save_debug: bool=False) -> Dict[str, object]:
    """
    Full evaluation pipeline for fraction prompt #2.

    Criteria:
      - image_exists: file exists on disk
      - image_readable: OpenCV can load the image
      - bar_detected: a horizontal bar region was found
      - bar_shape_ok: the bar is horizontally elongated
      - blue_fill_present: blue color is present in the bar's left portion
      - unfilled_region_ok: right portion is not blue (gray/white)
      - fill_ratio_ok: blue fill is ~25% of bar length (within tolerance)
      - labels_present: percentage labels detected below the bar
    """
    ALL_CRITERIA_KEYS = ['image_exists', 'image_readable', 'bar_detected', 'bar_shape_ok', 'blue_fill_present', 'unfilled_region_ok', 'fill_ratio_ok', 'labels_present']
    result = {'criteria': {}, 'passed': False}

    def _fail_at(step: str) -> Dict[str, object]:
        """Build criteria dict that is False from `step` onward."""
        crit = {}
        reached = False
        for k in ALL_CRITERIA_KEYS:
            if k == step:
                reached = True
            crit[k] = False if reached else True
        result['criteria'] = crit
        return result
    if not os.path.isfile(image_path):
        return _fail_at('image_exists')
    img = load_image(image_path)
    if img is None:
        return _fail_at('image_readable')
    bbox = find_bar_bbox(img)
    method = 'edges'
    if bbox is None:
        bbox = find_bar_bbox_by_blue(img)
        method = 'blue_fallback'
    c_bar_detected = bbox is not None
    if not c_bar_detected:
        result['criteria'] = _fail_at('bar_detected')['criteria']
        result['meta'] = {'bbox_method': method}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    x, y, w, h = bbox
    H_img, W_img = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, W_img - x)
    h = min(h, H_img - y)
    bar_img = img[y:y + h, x:x + w].copy()
    bar_h, bar_w = bar_img.shape[:2]
    aspect = bar_w / bar_h if bar_h > 0 else 0
    c_bar_shape = aspect >= MIN_BAR_ASPECT
    blue_ratio, left_purity, right_purity = measure_blue_fill_ratio(bar_img)
    c_blue_present = left_purity >= MIN_BLUE_PURITY
    c_unfilled_ok = check_unfilled_region(bar_img)
    c_ratio_ok = 0.22 <= blue_ratio <= 0.45 and left_purity >= 0.45 and (right_purity <= 0.15)
    c_labels = check_labels(img)
    c_all = all([c_bar_detected, c_bar_shape, c_blue_present, c_unfilled_ok, c_ratio_ok])
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'bar_detected': c_bar_detected, 'bar_shape_ok': bool(c_bar_shape), 'blue_fill_present': bool(c_blue_present), 'unfilled_region_ok': bool(c_unfilled_ok), 'fill_ratio_ok': bool(c_ratio_ok), 'labels_present': bool(c_labels)}
    result['passed'] = bool(c_all and c_bg_white)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'bbox_method': method, 'bbox': list(bbox), 'bar_aspect': round(aspect, 2), 'blue_fill_ratio': round(blue_ratio, 4), 'left_blue_purity': round(left_purity, 4), 'right_blue_purity': round(right_purity, 4)}
    if save_debug:
        base, _ = os.path.splitext(image_path)
        bar_path = base + '_bar.png'
        debug_path = base + '_debug.png'
        cv2.imwrite(bar_path, bar_img)
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        split_x = x + int(round(w * blue_ratio))
        cv2.line(vis, (split_x, y), (split_x, y + h), (255, 0, 0), 3)
        cv2.imwrite(debug_path, vis)
        result['debug'] = {'bar_path': bar_path, 'debug_path': debug_path}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 2.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
