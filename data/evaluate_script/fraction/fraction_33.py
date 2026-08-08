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
PROMPT_ID = 33
GREEN_H_LOW, GREEN_H_HIGH = (35, 85)
S_MIN, V_MIN = (50, 50)
TARGET_RATIO = 4 / 9.0
RATIO_TOL = 0.05
TARGET_CELLS = 4

def load_image(path: str) -> Optional[np.ndarray]:
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

def get_green_mask(hsv: np.ndarray) -> np.ndarray:
    lower = np.array([GREEN_H_LOW, S_MIN, V_MIN])
    upper = np.array([GREEN_H_HIGH, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

def bbox_from_green_mask(mask: np.ndarray, shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    H, W = shape
    size = max(w, h)
    pad = int(round(size * 0.04))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)
    if w < 40 or h < 40:
        return None
    return (x, y, w, h)

def sample_3x3_cells(img: np.ndarray, green_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, list]:
    x, y, w, h = bbox
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_cells = 0
    pale_cells = 0
    reports = []
    for row in range(3):
        for col in range(3):
            x0 = x + int(col * w / 3)
            x1 = x + int((col + 1) * w / 3)
            y0 = y + int(row * h / 3)
            y1 = y + int((row + 1) * h / 3)
            padx = max(1, int((x1 - x0) * 0.18))
            pady = max(1, int((y1 - y0) * 0.18))
            gm = green_mask[y0 + pady:y1 - pady, x0 + padx:x1 - padx]
            cell_hsv = hsv[y0 + pady:y1 - pady, x0 + padx:x1 - padx]
            if gm.size == 0 or cell_hsv.size == 0:
                reports.append({'row': row, 'col': col, 'green_fraction': 0.0, 'pale_fraction': 0.0, 'class': 'empty'})
                continue
            green_frac = float(np.mean(gm > 0))
            sat = cell_hsv[:, :, 1]
            val = cell_hsv[:, :, 2]
            pale_frac = float(np.mean((sat < 65) & (val > 170)))
            if green_frac >= 0.2:
                cls = 'green'
                green_cells += 1
            elif pale_frac >= 0.45:
                cls = 'pale'
                pale_cells += 1
            else:
                cls = 'other'
            reports.append({'row': row, 'col': col, 'green_fraction': round(green_frac, 4), 'pale_fraction': round(pale_frac, 4), 'class': cls})
    return (green_cells, pale_cells, reports)

def get_grid_bbox(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    if cv2.contourArea(c) < 10000:
        return None
    x, y, w, h = cv2.boundingRect(c)
    H, W = img.shape[:2]
    if w > W * 0.95 and h > H * 0.95:
        if len(contours) < 2:
            return None
        c = contours[1]
        if cv2.contourArea(c) < 10000:
            return None
        x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)

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

def evaluate(image_path: str) -> Dict[str, object]:
    result = {'criteria': {}, 'passed': False}
    if not os.path.isfile(image_path):
        result['criteria'] = {'image_exists': False, 'passed': False}
        return result
    img = load_image(image_path)
    if img is None:
        result['criteria'] = {'image_exists': True, 'image_readable': False, 'passed': False}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    bbox = get_grid_bbox(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = get_green_mask(hsv)
    if bbox is None:
        bbox = bbox_from_green_mask(green_mask, img.shape[:2])
    c_grid_detected = bbox is not None
    if not c_grid_detected:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'grid_detected': False, 'ratio_correct': False}
        return result
    x, y, w, h = bbox
    grid_area_pixels = w * h
    green_roi = green_mask[y:y + h, x:x + w]
    green_pixels = np.count_nonzero(green_roi)
    ratio = green_pixels / grid_area_pixels if grid_area_pixels > 0 else 0
    c_ratio_ok = abs(ratio - TARGET_RATIO) <= RATIO_TOL
    cell_area = w * h / float(9)
    if 'green_mask' in locals():
        blobs, _ = cv2.findContours(green_mask[y:y + h, x:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        blobs, _ = cv2.findContours(green_mask[y:y + h, x:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    estimated_cells = 0
    for blob in blobs:
        area = cv2.contourArea(blob)
        if area > cell_area * 0.2:
            estimated_cells += max(1, int(round(area / cell_area)))
    c_cell_count_ok = estimated_cells == TARGET_CELLS
    sampled_green_cells, sampled_pale_cells, sampled_cell_reports = sample_3x3_cells(img, green_mask, bbox)
    c_sampled_count_ok = sampled_green_cells == TARGET_CELLS and sampled_pale_cells >= 4
    if c_sampled_count_ok:
        c_cell_count_ok = True
        c_ratio_ok = True
    elif sampled_green_cells >= TARGET_CELLS + 1:
        c_cell_count_ok = False
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'grid_detected': True, 'ratio_correct': bool(c_ratio_ok), 'cell_count_correct': bool(c_cell_count_ok), 'sampled_cell_count_correct': bool(c_sampled_count_ok)}
    result['passed'] = bool(c_ratio_ok and c_bg_white and c_cell_count_ok)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'green_pixels': green_pixels, 'grid_area': grid_area_pixels, 'measured_ratio': round(ratio, 4), 'target_ratio': round(TARGET_RATIO, 4), 'tolerance': RATIO_TOL, 'estimated_cells': estimated_cells, 'sampled_green_cells': sampled_green_cells, 'sampled_pale_cells': sampled_pale_cells, 'sampled_cell_reports': sampled_cell_reports}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 33.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
