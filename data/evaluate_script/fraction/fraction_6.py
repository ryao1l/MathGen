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
PROMPT_ID = 6
BLACK_THRESHOLD = 80
WHITE_THRESHOLD = 180
RATIO_TARGET = 0.85
RATIO_TOL = 0.1
MIN_BAR_ASPECT = 1.5
MIN_BLACK_PURITY = 0.5
MIN_WHITE_PURITY = 0.5
CANNY1, CANNY2 = (40, 140)
DILATE_K = 5
MIN_AREA_FRAC = 0.02
INSET_FRAC = 0.08

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

def find_bar_bbox_by_intensity(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Fallback: find the bar region by looking at the bounding box of very dark
    (black) pixels and expanding rightward to include the white portion.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    coords = cv2.findNonZero(black_mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    row_center = y + h // 2
    row_band = gray[max(0, row_center - h // 4):min(H, row_center + h // 4), :]
    mean_intensity = row_band.mean(axis=0)
    bar_right = x + w
    for col in range(x + w, W):
        if mean_intensity[col] < 180:
            bar_right = col
            break
    else:
        bar_right = W
    pad = 5
    bx = max(0, x - pad)
    by = max(0, y - pad)
    bw = min(W - bx, bar_right - bx + pad)
    bh = min(H - by, h + 2 * pad)
    if bw < 20 or bh < 10:
        return None
    return (bx, by, bw, bh)

def inset_region(img: np.ndarray) -> np.ndarray:
    """Remove a border by insetting the image."""
    H, W = img.shape[:2]
    inset = max(1, int(round(min(H, W) * INSET_FRAC)))
    x0, y0 = (inset, inset)
    x1, y1 = (W - inset, H - inset)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return img
    return img[y0:y1, x0:x1]

def get_central_band(gray: np.ndarray) -> np.ndarray:
    """Extract the central 40% rows of a grayscale image to avoid top/bottom borders."""
    H, W = gray.shape
    y0 = int(H * 0.3)
    y1 = int(H * 0.7)
    if y1 <= y0:
        return gray
    return gray[y0:y1, :]

def measure_black_ratio(bar_img: np.ndarray) -> Tuple[float, float, float]:
    """
    Measure the black fill ratio within the bar image.
    Uses the central row band to avoid border artefacts.

    Returns:
      - black_ratio: fraction of bar width that is black
      - left_black_purity: fraction of black pixels in the left portion
      - right_white_purity: fraction of white pixels in the right portion
    """
    bar_inset = inset_region(bar_img)
    H, W = bar_inset.shape[:2]
    if H == 0 or W == 0:
        return (0.0, 0.0, 0.0)
    gray = cv2.cvtColor(bar_inset, cv2.COLOR_BGR2GRAY)
    band = get_central_band(gray)
    col_median = np.median(band, axis=0)
    transition_threshold = (BLACK_THRESHOLD + WHITE_THRESHOLD) / 2.0
    dark_cols = np.where(col_median < transition_threshold)[0]
    if len(dark_cols) == 0:
        return (0.0, 0.0, 0.0)
    if len(dark_cols) > 1:
        diffs = np.diff(dark_cols)
        gap_indices = np.where(diffs > max(3, W * 0.05))[0]
        if len(gap_indices) > 0:
            rightmost_dark = dark_cols[gap_indices[0]]
        else:
            rightmost_dark = dark_cols[-1]
    else:
        rightmost_dark = dark_cols[0]
    black_ratio = (rightmost_dark + 1) / W
    split_left = int(W * 0.75)
    split_right = int(W * 0.9)
    left_part = band[:, :split_left]
    right_part = band[:, split_right:]
    left_black_count = float(np.sum(left_part < BLACK_THRESHOLD))
    left_black_purity = left_black_count / float(left_part.size) if left_part.size > 0 else 0.0
    right_white_count = float(np.sum(right_part > WHITE_THRESHOLD))
    right_white_purity = right_white_count / float(right_part.size) if right_part.size > 0 else 0.0
    return (black_ratio, left_black_purity, right_white_purity)

def check_left_is_black(bar_img: np.ndarray) -> bool:
    """Check that the left portion of the bar is predominantly black."""
    bar_inset = inset_region(bar_img)
    gray = cv2.cvtColor(bar_inset, cv2.COLOR_BGR2GRAY)
    band = get_central_band(gray)
    bH, bW = band.shape
    split = int(bW * 0.7)
    left = band[:, :split]
    if left.size == 0:
        return False
    black_frac = float(np.sum(left < BLACK_THRESHOLD)) / float(left.size)
    return black_frac >= MIN_BLACK_PURITY

def check_right_is_white(bar_img: np.ndarray) -> bool:
    """Check that the right-most portion of the bar is predominantly white."""
    bar_inset = inset_region(bar_img)
    gray = cv2.cvtColor(bar_inset, cv2.COLOR_BGR2GRAY)
    band = get_central_band(gray)
    bH, bW = band.shape
    split = int(bW * 0.9)
    right = band[:, split:]
    if right.size == 0:
        return False
    white_frac = float(np.sum(right > WHITE_THRESHOLD)) / float(right.size)
    return white_frac >= MIN_WHITE_PURITY

def check_labels(img: np.ndarray) -> bool:
    """
    Check for the presence of dark text (labels/annotations).
    Looks in bottom and top portions of the image.
    """
    H, W = img.shape[:2]
    total_clusters = 0
    bottom = img[int(H * 0.55):, :]
    if bottom.size > 0:
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        col_density = text_mask.mean(axis=0) / 255.0
        kernel_size = max(1, W // 30)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(col_density, np.ones(kernel_size) / kernel_size, mode='same')
        above = smoothed > 0.02
        transitions = np.diff(above.astype(int))
        total_clusters += int(np.sum(transitions == 1))
    top = img[:int(H * 0.2), :]
    if top.size > 0:
        gray_top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        _, text_mask_top = cv2.threshold(gray_top, 120, 255, cv2.THRESH_BINARY_INV)
        col_density_top = text_mask_top.mean(axis=0) / 255.0
        kernel_size = max(1, W // 30)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed_top = np.convolve(col_density_top, np.ones(kernel_size) / kernel_size, mode='same')
        above_top = smoothed_top > 0.02
        transitions_top = np.diff(above_top.astype(int))
        total_clusters += int(np.sum(transitions_top == 1))
    return total_clusters >= 2

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
    Full evaluation pipeline for fraction prompt #6.

    Criteria:
      - image_exists: file exists on disk
      - image_readable: OpenCV can load the image
      - bar_detected: a rectangular bar region was found
      - bar_shape_ok: the bar is horizontally elongated
      - left_is_black: left portion of bar is solid black
      - right_is_white: right portion of bar is solid white
      - split_ratio_ok: black/white split is ~85%/15% (within tolerance)
      - labels_present: labels detected near the bar
    """
    ALL_CRITERIA_KEYS = ['image_exists', 'image_readable', 'bar_detected', 'bar_shape_ok', 'left_is_black', 'right_is_white', 'split_ratio_ok', 'labels_present']
    result = {'criteria': {}, 'passed': False}

    def _fail_at(step: str) -> Dict[str, object]:
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
        bbox = find_bar_bbox_by_intensity(img)
        method = 'intensity_fallback'
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
    c_left_black = check_left_is_black(bar_img)
    c_right_white = check_right_is_white(bar_img)
    black_ratio, left_purity, right_purity = measure_black_ratio(bar_img)
    c_ratio_ok = abs(black_ratio - RATIO_TARGET) <= RATIO_TOL
    c_labels = check_labels(img)
    c_labeled_ratio_ok = bool(c_labels and 0.72 <= black_ratio <= 0.92 and (left_purity >= 0.85) and (right_purity >= 0.9))
    c_all = all([c_bar_detected, c_bar_shape, c_left_black, c_right_white, c_ratio_ok or c_labeled_ratio_ok])
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'bar_detected': c_bar_detected, 'bar_shape_ok': bool(c_bar_shape), 'left_is_black': bool(c_left_black), 'right_is_white': bool(c_right_white), 'split_ratio_ok': bool(c_ratio_ok), 'labeled_ratio_ok': bool(c_labeled_ratio_ok), 'labels_present': bool(c_labels)}
    result['passed'] = bool(c_all and c_bg_white)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'bbox_method': method, 'bbox': list(bbox), 'bar_aspect': round(aspect, 2), 'black_ratio': round(black_ratio, 4), 'left_black_purity': round(left_purity, 4), 'right_white_purity': round(right_purity, 4)}
    if save_debug:
        base, _ = os.path.splitext(image_path)
        bar_path = base + '_bar.png'
        debug_path = base + '_debug.png'
        cv2.imwrite(bar_path, bar_img)
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        split_x = x + int(round(w * black_ratio))
        cv2.line(vis, (split_x, y), (split_x, y + h), (0, 255, 0), 3)
        cv2.imwrite(debug_path, vis)
        result['debug'] = {'bar_path': bar_path, 'debug_path': debug_path}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 6.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
