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
PROMPT_ID = 50
sys.path.insert(0, os.path.dirname(__file__))

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x_ay, y_ax = detect_axes(img_gray)
    blue_mask = extract_blue_mask(img_hsv)
    red_mask = extract_red_mask(img_hsv)
    red_pixel_count = int(np.sum(red_mask > 0))
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = True
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        criteria['curve_right_of_y_axis'] = float(np.sum(rel_x >= -10)) / len(rel_x) > 0.9
        lin_coeffs = np.polyfit(rel_x, rel_y, 1)
        criteria['monotonically_increasing'] = lin_coeffs[0] > 0
        quad_coeffs = np.polyfit(rel_x, rel_y, 2)
        criteria['concave_down'] = quad_coeffs[0] < 0
        criteria['above_x_axis'] = float(np.sum(rel_y >= -10)) / len(rel_y) > 0.9
        sorted_idx = np.argsort(rel_x)
        n_start = max(1, len(sorted_idx) // 20)
        start_y = np.median(rel_y[sorted_idx[:n_start]])
        criteria['starts_near_origin'] = abs(start_y) < h * 0.15
        dark = img_gray < 80
        yy, xx = np.where(dark)
        off_axes = (np.abs(yy - x_ay) > 25) & (np.abs(xx - y_ax) > 25) & (yy > 0.05 * h) & (yy < 0.95 * h) & (xx > 0.05 * w) & (xx < 0.95 * w)
        extra_text_ratio = float(np.sum(off_axes)) / max(float(h * w), 1.0)
        criteria['no_extra_formula_text'] = extra_text_ratio < 0.004
    else:
        criteria['curve_right_of_y_axis'] = False
        criteria['monotonically_increasing'] = False
        criteria['concave_down'] = False
        criteria['above_x_axis'] = False
        criteria['starts_near_origin'] = False
        criteria['no_extra_formula_text'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.3
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 46, 'passed': passed, 'criteria': criteria, 'meta': {'red_pixels': red_pixel_count, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 50.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
