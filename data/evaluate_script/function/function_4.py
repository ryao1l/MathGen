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
PROMPT_ID = 4
sys.path.insert(0, os.path.dirname(__file__))

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x_ay, y_ax = detect_axes(img_gray)
    pixel_aspect = h / w if w > 0 else 1.0
    blue_mask = extract_blue_mask(img_hsv)
    red_mask = extract_red_mask(img_hsv)
    red_pixel_count = int(np.sum(red_mask > 0))
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    criteria = {}
    slope = intercept = r_squared = slope_ratio = None
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
        slope_ratio = abs(slope) / max(pixel_aspect, 1e-06)
        criteria['negative_slope'] = slope < 0 and 0.95 <= slope_ratio <= 1.05
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
        criteria['negative_slope'] = False
        criteria['is_linear'] = False
        criteria['passes_origin'] = False
        criteria['symmetric_distribution'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 9, 'passed': passed, 'criteria': criteria, 'meta': {'slope': slope, 'intercept': intercept, 'r_squared': r_squared, 'slope_ratio': slope_ratio, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 4.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
