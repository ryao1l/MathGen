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
PROMPT_ID = 26
TOP_TARGET = 11 / 16.0
BOTTOM_TARGET = 13 / 20.0
RATIO_TOL = 0.08

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

def get_green_mask(hsv):
    return cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

def get_orange_mask(hsv):
    return cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))

def find_bars(img):
    """Find two horizontal bars in the image, return them sorted top-to-bottom."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 15), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    H, W = img.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        area = cv2.contourArea(c)
        if aspect > 2.0 and area > 1000 and (w > W * 0.2):
            bars.append((x, y, w, h))
    bars.sort(key=lambda b: b[1])
    return bars

def measure_fill_ratio(bar_img, color_mask):
    """Measure what fraction of the bar width is filled with the given color."""
    H, W = bar_img.shape[:2]
    if W == 0 or H == 0:
        return 0.0
    col_density = color_mask.mean(axis=0) / 255.0
    filled_cols = np.where(col_density > 0.15)[0]
    if len(filled_cols) == 0:
        return 0.0
    rightmost = filled_cols[-1]
    return (rightmost + 1) / W

def check_background_is_white(image_path: str, threshold: int=240) -> Tuple[bool, float]:
    if not os.path.isfile(image_path):
        return (False, 0.0)
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return (False, 0.0)
    H, W = raw_img.shape[:2]
    border = max(1, int(min(H, W) * 0.08))
    if raw_img.ndim == 3 and raw_img.shape[2] == 4:
        alpha = raw_img[:, :, 3]
        alpha_border = np.concatenate([alpha[:border, :].ravel(), alpha[H - border:, :].ravel(), alpha[:, :border].ravel(), alpha[:, W - border:].ravel()])
        if float(np.mean(alpha_border)) < 200:
            return (False, float(np.mean(alpha_border)))
        img_check = raw_img[:, :, :3]
    else:
        img_check = raw_img
    if img_check.ndim == 3:
        gray = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_check
    intensity_border = np.concatenate([gray[:border, :].ravel(), gray[H - border:, :].ravel(), gray[:, :border].ravel(), gray[:, W - border:].ravel()])
    median_intensity = float(np.median(intensity_border))
    return (median_intensity >= threshold, median_intensity)

def evaluate(image_path: str) -> Dict[str, object]:
    result = {'criteria': {}, 'passed': False}
    if not os.path.isfile(image_path):
        result['criteria'] = {'image_exists': False}
        return result
    img = load_image(image_path)
    if img is None:
        result['criteria'] = {'image_exists': True, 'image_readable': False}
        return result
    c_bg_white, bg_mean = check_background_is_white(image_path)
    bars = find_bars(img)
    if len(bars) < 2:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'two_bars_detected': False}
        return result
    top_bar = bars[0]
    bottom_bar = bars[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tx, ty, tw, th = top_bar
    top_roi = img[ty:ty + th, tx:tx + tw]
    top_hsv = hsv[ty:ty + th, tx:tx + tw]
    green_mask = get_green_mask(top_hsv)
    top_ratio = measure_fill_ratio(top_roi, green_mask)
    bx, by, bw, bh = bottom_bar
    bottom_roi = img[by:by + bh, bx:bx + bw]
    bottom_hsv = hsv[by:by + bh, bx:bx + bw]
    orange_mask = get_orange_mask(bottom_hsv)
    bottom_ratio = measure_fill_ratio(bottom_roi, orange_mask)
    top_ok = abs(top_ratio - TOP_TARGET) <= RATIO_TOL
    bottom_ok = abs(bottom_ratio - BOTTOM_TARGET) <= RATIO_TOL
    result['criteria'] = {'image_exists': True, 'image_readable': True, 'background_is_white': bool(c_bg_white), 'two_bars_detected': True, 'top_bar_ratio_correct': bool(top_ok), 'bottom_bar_ratio_correct': bool(bottom_ok)}
    result['passed'] = bool(top_ok and bottom_ok and c_bg_white)
    result['meta'] = {'background_median_intensity': round(bg_mean, 1), 'top_bar_ratio': round(top_ratio, 4), 'top_target': round(TOP_TARGET, 4), 'bottom_bar_ratio': round(bottom_ratio, 4), 'bottom_target': round(BOTTOM_TARGET, 4), 'tolerance': RATIO_TOL}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen fraction case 26.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
