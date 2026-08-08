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
PROMPT_ID = 1
GREEN_H_LOW1, GREEN_H_HIGH1 = (30, 90)
GREEN_S_MIN = 40
GREEN_V_MIN = 40
RATIO_TARGET = 0.5
RATIO_TOL = 0.1
MIN_BAR_ASPECT = 2.0
MIN_GREEN_PURITY = 0.3
MAX_GREEN_IN_RIGHT = 0.15
CANNY1, CANNY2 = (40, 140)
DILATE_K = 5
MIN_AREA_FRAC = 0.02
INSET_FRAC = 0.05

def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from disk (BGR). Return None on failure."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path)
    return img

def green_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of green pixels in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([GREEN_H_LOW1, GREEN_S_MIN, GREEN_V_MIN])
    upper = np.array([GREEN_H_HIGH1, 255, 255])
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

def find_bar_bbox_by_green(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Fallback: find the bar region by looking at the bounding box of green pixels
    and expanding it to include the full bar (assuming bar extends rightward).
    """
    gm = green_mask_hsv(img)
    H, W = img.shape[:2]
    coords = cv2.findNonZero(gm)
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

def measure_green_fill_ratio(bar_img: np.ndarray) -> Tuple[float, float, float]:
    """
    Measure the green fill ratio within the bar image.

    Returns:
      - green_ratio: fraction of bar width that is green
      - left_green_purity: fraction of green pixels in the left half
      - right_green_purity: fraction of green pixels in the right half
    """
    bar_inset = inset_region(bar_img)
    H, W = bar_inset.shape[:2]
    gm = green_mask_hsv(bar_inset)
    total_pixels = H * W
    if total_pixels == 0:
        return (0.0, 0.0, 0.0)
    col_green = gm.mean(axis=0) / 255.0
    green_threshold = 0.15
    green_cols = np.where(col_green > green_threshold)[0]
    if len(green_cols) == 0:
        return (0.0, 0.0, 0.0)
    rightmost_green = green_cols[-1]
    green_ratio = (rightmost_green + 1) / W
    mid = W // 2
    left_half = gm[:, :mid]
    right_half = gm[:, mid:]
    left_green_purity = float(np.sum(left_half > 0)) / float(left_half.size) if left_half.size > 0 else 0.0
    right_green_purity = float(np.sum(right_half > 0)) / float(right_half.size) if right_half.size > 0 else 0.0
    return (green_ratio, left_green_purity, right_green_purity)

def check_unfilled_region(bar_img: np.ndarray) -> bool:
    """
    Check that the right portion of the bar (unfilled) is gray/white
    (i.e., not green and not heavily colored).
    """
    bar_inset = inset_region(bar_img)
    H, W = bar_inset.shape[:2]
    mid = W // 2
    right_half = bar_inset[:, mid:]
    if right_half.size == 0:
        return False
    gm = green_mask_hsv(right_half)
    green_frac = float(np.sum(gm > 0)) / float(gm.size)
    return green_frac < MAX_GREEN_IN_RIGHT

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
    Full evaluation pipeline for fraction prompt #1.

    Criteria:
      - image_exists: file exists on disk
      - image_readable: OpenCV can load the image
      - bar_detected: a horizontal bar region was found
      - bar_shape_ok: the bar is horizontally elongated
      - green_fill_present: green color is present in the bar
      - unfilled_region_ok: right portion is not green (gray/white)
      - fill_ratio_ok: green fill is ~50% of bar length (within tolerance)
      - all_criteria_passed: all above criteria pass
    """
    result = {'criteria': {}, 'passed': False}
    c_exists = os.path.isfile(image_path)
    if not c_exists:
        result['criteria'] = {'image_exists': False, 'image_readable': False, 'bar_detected': False, 'bar_shape_ok': False, 'green_fill_present': False, 'unfilled_region_ok': False, 'fill_ratio_ok': False, 'all_criteria_passed': False}
        return result
    img = load_image(image_path)
    c_readable = img is not None
    if not c_readable:
        result['criteria'] = {'image_exists': True, 'image_readable': False, 'bar_detected': False, 'bar_shape_ok': False, 'green_fill_present': False, 'unfilled_region_ok': False, 'fill_ratio_ok': False, 'all_criteria_passed': False}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    bbox = find_bar_bbox(img)
    method = 'edges'
    if bbox is None:
        bbox = find_bar_bbox_by_green(img)
        method = 'green_fallback'
    c_bar_detected = bbox is not None
    if not c_bar_detected:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'bar_detected': False, 'bar_shape_ok': False, 'green_fill_present': False, 'unfilled_region_ok': False, 'fill_ratio_ok': False, 'all_criteria_passed': False}
        result['meta'] = {'bbox_method': method}
        return result
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
    green_ratio, left_purity, right_purity = measure_green_fill_ratio(bar_img)
    c_green_present = left_purity >= MIN_GREEN_PURITY
    c_unfilled_ok = check_unfilled_region(bar_img)
    c_ratio_ok = abs(green_ratio - RATIO_TARGET) <= RATIO_TOL
    c_all = all([c_bar_detected, c_bar_shape, c_green_present, c_unfilled_ok, c_ratio_ok, c_bg_white])
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'bar_detected': c_bar_detected, 'bar_shape_ok': bool(c_bar_shape), 'green_fill_present': bool(c_green_present), 'unfilled_region_ok': bool(c_unfilled_ok), 'fill_ratio_ok': bool(c_ratio_ok), 'all_criteria_passed': bool(c_all)}
    result['passed'] = bool(c_all)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'bbox_method': method, 'bbox': list(bbox), 'bar_aspect': round(aspect, 2), 'green_fill_ratio': round(green_ratio, 4), 'left_green_purity': round(left_purity, 4), 'right_green_purity': round(right_purity, 4)}
    if save_debug:
        base, _ = os.path.splitext(image_path)
        bar_path = base + '_bar.png'
        debug_path = base + '_debug.png'
        cv2.imwrite(bar_path, bar_img)
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        split_x = x + int(round(w * green_ratio))
        cv2.line(vis, (split_x, y), (split_x, y + h), (0, 255, 0), 3)
        cv2.imwrite(debug_path, vis)
        result['debug'] = {'bar_path': bar_path, 'debug_path': debug_path}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 1.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
