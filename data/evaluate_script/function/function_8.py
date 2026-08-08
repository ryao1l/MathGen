#!/usr/bin/env python3
from __future__ import annotations
import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
try:
    import pytesseract
except Exception:
    pytesseract = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
import function_common
from function_common import *
globals().update({name: getattr(function_common, name) for name in dir(function_common) if name.startswith('_')})
PROMPT_ID = 8
DEBUG = False
DEBUG_DIR = 'debug_function_20'
BLUE_H_LO, BLUE_H_HI = (100, 130)
BLUE_S_MIN, BLUE_V_MIN = (50, 50)
RED_MAX_PIXELS = 50
MIN_BLUE_PIXELS = 100
MIN_COVERAGE = 0.6
MIN_R_SQUARED = 0.95

def _dsave(name: str, img: np.ndarray):
    if not DEBUG or img is None or img.size == 0:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, name), img)

def extract_blue_mask(img_hsv: np.ndarray) -> np.ndarray:
    lower = np.array([BLUE_H_LO, BLUE_S_MIN, BLUE_V_MIN])
    upper = np.array([BLUE_H_HI, 255, 255])
    return cv2.inRange(img_hsv, lower, upper)

def extract_red_mask(img_hsv: np.ndarray) -> np.ndarray:
    lo = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    hi = cv2.inRange(img_hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(lo, hi)

def detect_axes(img_gray: np.ndarray):
    _, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    x_axis_y = float(np.argmax(np.sum(h_lines, axis=1)))
    y_axis_x = float(np.argmax(np.sum(v_lines, axis=0)))
    return (x_axis_y, y_axis_x)

def evaluate(image_path: str) -> Dict[str, Any]:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'criteria': {'image_readable': False}, 'meta': {}}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _dsave('00_input.png', img_bgr)
    x_axis_y, y_axis_x = detect_axes(img_gray)
    blue_mask = extract_blue_mask(img_hsv)
    _dsave('10_blue_mask.png', blue_mask)
    red_mask = extract_red_mask(img_hsv)
    red_pixel_count = int(np.sum(red_mask > 0))
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    criteria: Dict[str, bool] = {}
    criteria['blue_curve_exists'] = len(blue_pixels) > MIN_BLUE_PIXELS
    criteria['no_red_hallucination'] = red_pixel_count < RED_MAX_PIXELS
    slope = intercept = r_squared = None
    if len(blue_pixels) > MIN_BLUE_PIXELS:
        from collections import defaultdict
        col_ys = defaultdict(list)
        for row, col in blue_pixels:
            col_ys[col].append(row)
        med_cols = sorted(col_ys.keys())
        rel_x = np.array([c - y_axis_x for c in med_cols])
        rel_y = np.array([x_axis_y - np.median(col_ys[c]) for c in med_cols])
        coeffs = np.polyfit(rel_x, rel_y, 1)
        slope, intercept = (float(coeffs[0]), float(coeffs[1]))
        y_pred = np.polyval(coeffs, rel_x)
        ss_res = np.sum((rel_y - y_pred) ** 2)
        ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        criteria['is_linear'] = r_squared > MIN_R_SQUARED
        criteria['positive_slope'] = slope > 0
        criteria['negative_y_intercept'] = intercept / h < -0.05 if h > 0 else False
        x_min = np.min(blue_pixels[:, 1])
        x_max = np.max(blue_pixels[:, 1])
        coverage = float(x_max - x_min) / w if w > 0 else 0
        criteria['domain_coverage'] = coverage > MIN_COVERAGE
    else:
        criteria['is_linear'] = False
        criteria['positive_slope'] = False
        criteria['negative_y_intercept'] = False
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_axis_y, 'y_axis_x': y_axis_x, 'red_pixels': red_pixel_count, 'slope': slope, 'intercept': intercept, 'r_squared': r_squared}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 8.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
