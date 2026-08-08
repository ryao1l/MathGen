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
PROMPT_ID = 1
sys.path.insert(0, os.path.dirname(__file__))
DEBUG = False
DEBUG_DIR = 'debug_id7'
BLUE_H_LO, BLUE_H_HI = (100, 130)

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

def detect_axes(img_gray):
    """Detect axis positions as the x-axis row and y-axis column."""
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
    blue_mask = extract_color_mask(img_hsv, (BLUE_H_LO, BLUE_H_HI))
    ensure_debug(blue_mask, 'blue_mask.png')
    red_mask = extract_red_mask(img_hsv)
    red_pixel_count = np.sum(red_mask > 0)
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['axes_detected'] = 0.05 * h < x_ay < 0.95 * h and 0.05 * w < y_ax < 0.95 * w
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    slope = None
    intercept = None
    y_std = None
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        y_std = float(np.std(rel_y))
        criteria['is_horizontal'] = y_std < 25
        coeffs = np.polyfit(rel_x, rel_y, 1)
        slope, intercept = (float(coeffs[0]), float(coeffs[1]))
        criteria['slope_near_zero'] = abs(slope) < 0.2
        mean_rel_y = float(np.mean(rel_y))
        criteria['below_x_axis'] = mean_rel_y < -10
        dist_below_axis = x_ay - np.mean(blue_pixels[:, 0])
        axis_to_bottom = h - x_ay
        if axis_to_bottom > 0:
            position_ratio = abs(dist_below_axis) / axis_to_bottom
            criteria['correct_y_position'] = 0.15 < position_ratio < 0.85
        else:
            criteria['correct_y_position'] = False
    else:
        criteria['is_horizontal'] = False
        criteria['slope_near_zero'] = False
        criteria['below_x_axis'] = False
        criteria['correct_y_position'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 7, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'red_pixels': int(red_pixel_count), 'slope': slope, 'intercept': intercept, 'y_std_px': y_std, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 1.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
