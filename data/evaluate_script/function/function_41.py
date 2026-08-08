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
PROMPT_ID = 41
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
    slope = None
    intercept = None
    r_squared = None
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        coeffs = np.polyfit(rel_x, rel_y, 1)
        slope, intercept = (float(coeffs[0]), float(coeffs[1]))
        y_pred = np.polyval(coeffs, rel_x)
        ss_res = np.sum((rel_y - y_pred) ** 2)
        ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        criteria['is_linear'] = r_squared > 0.95
        criteria['negative_slope'] = slope < -0.9
        intercept_ratio = intercept / max(float(h), 1.0)
        criteria['positive_y_intercept'] = 0.08 < intercept_ratio < 0.2
        has_above = np.any(rel_y > 5)
        has_below = np.any(rel_y < -5)
        criteria['crosses_x_axis'] = bool(has_above and has_below)
    else:
        criteria['is_linear'] = False
        criteria['negative_slope'] = False
        criteria['positive_y_intercept'] = False
        criteria['crosses_x_axis'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 37, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'red_pixels': int(red_pixel_count), 'slope': slope, 'intercept': intercept, 'r_squared': r_squared, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 41.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
