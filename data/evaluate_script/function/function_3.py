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
PROMPT_ID = 3
sys.path.insert(0, os.path.dirname(__file__))
DEBUG = False
DEBUG_DIR = 'debug_id6'
BLUE_H_LO, BLUE_H_HI = (100, 130)
OCR_CONF_THRESH = 15.0

def ensure_debug(img, name):
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, name), img)

def extract_color_mask(img_hsv, h_range):
    lower = np.array([h_range[0], 50, 50])
    upper = np.array([h_range[1], 255, 255])
    return cv2.inRange(img_hsv, lower, upper)

def extract_red_mask(img_hsv):
    """Red wraps around the HSV hue range, so combine both red intervals."""
    mask_lo = extract_color_mask(img_hsv, (0, 10))
    mask_hi = extract_color_mask(img_hsv, (170, 180))
    return cv2.bitwise_or(mask_lo, mask_hi)

def count_axis_tick_positions(img_gray, x_axis_y, y_axis_x, axis='x'):
    """Count visible black tick marks, ignoring the origin and arrow endpoints."""
    h, w = img_gray.shape[:2]
    _, binary = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    x_ay = int(round(x_axis_y))
    y_ax = int(round(y_axis_x))
    if axis == 'x':
        strip = binary[max(0, x_ay - 15):min(h, x_ay + 15), :]
        sums = np.sum(strip > 0, axis=0)
        idx = np.where(sums > np.median(sums) + 3)[0]
        if len(idx) == 0:
            return 0
        groups = np.split(idx, np.where(np.diff(idx) > 8)[0] + 1)
        positions = [int(np.mean(g)) for g in groups if len(g) > 0]
        positions = [p for p in positions if abs(p - y_ax) > 18 and 40 < p < w - 40]
    else:
        strip = binary[:, max(0, y_ax - 15):min(w, y_ax + 15)]
        sums = np.sum(strip > 0, axis=1)
        idx = np.where(sums > np.median(sums) + 3)[0]
        if len(idx) == 0:
            return 0
        groups = np.split(idx, np.where(np.diff(idx) > 8)[0] + 1)
        positions = [int(np.mean(g)) for g in groups if len(g) > 0]
        positions = [p for p in positions if abs(p - x_ay) > 18 and 40 < p < h - 40]
    return len(positions)

def detect_axes(img_gray):
    _, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    x_axis_y = np.argmax(np.sum(h_lines, axis=1))
    y_axis_x = np.argmax(np.sum(v_lines, axis=0))
    return (float(x_axis_y), float(y_axis_x))

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x_ay, y_ax = detect_axes(img_gray)
    pixel_aspect = h / w if w > 0 else 1.0
    blue_mask = extract_color_mask(img_hsv, (BLUE_H_LO, BLUE_H_HI))
    ensure_debug(blue_mask, 'blue_mask.png')
    red_mask = extract_red_mask(img_hsv)
    red_pixel_count = np.sum(red_mask > 0)
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        coeffs = np.polyfit(rel_x, rel_y, 1)
        slope, intercept = (coeffs[0], coeffs[1])
        y_pred = np.polyval(coeffs, rel_x)
        ss_res = np.sum((rel_y - y_pred) ** 2)
        ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        pixel_aspect = h / w if w > 0 else 1.0
        slope_ratio = slope / max(pixel_aspect, 1e-06)
        criteria['positive_slope'] = slope > 0
        criteria['unit_slope_scale'] = 0.97 <= slope_ratio <= 1.03
        criteria['is_linear'] = r_squared > 0.95
        intercept_ratio = abs(intercept) / h if h > 0 else 999
        criteria['passes_origin'] = intercept_ratio < 0.08
        left_count = np.sum(rel_x < 0)
        right_count = np.sum(rel_x > 0)
        total = left_count + right_count
        if total > 0:
            balance = min(left_count, right_count) / max(left_count, right_count)
            criteria['symmetric_distribution'] = balance > 0.3
        else:
            criteria['symmetric_distribution'] = False
    else:
        criteria['positive_slope'] = False
        criteria['unit_slope_scale'] = False
        criteria['is_linear'] = False
        criteria['passes_origin'] = False
        criteria['symmetric_distribution'] = False
        slope_ratio = None
    x_tick_count = count_axis_tick_positions(img_gray, x_ay, y_ax, axis='x')
    y_tick_count = count_axis_tick_positions(img_gray, x_ay, y_ax, axis='y')
    criteria['reasonable_tick_density'] = x_tick_count <= 12 and y_tick_count <= 12
    if len(blue_pixels) > 0:
        x_min = np.min(blue_pixels[:, 1])
        x_max = np.max(blue_pixels[:, 1])
        coverage = (x_max - x_min) / w if w > 0 else 0
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 6, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'red_pixels': int(red_pixel_count), 'pixel_aspect': float(pixel_aspect), 'slope': float(slope) if len(blue_pixels) > 50 else None, 'slope_ratio': float(slope_ratio) if slope_ratio is not None else None, 'intercept': float(intercept) if len(blue_pixels) > 50 else None, 'r_squared': float(r_squared) if len(blue_pixels) > 50 else None, 'x_tick_count': x_tick_count, 'y_tick_count': y_tick_count, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 3.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
