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
PROMPT_ID = 48
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
    a3 = cubic_r_sq = lin_slope = lin_r_sq = None
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        coeffs = np.polyfit(rel_x, rel_y, 3)
        a3 = float(coeffs[0])
        y_pred = np.polyval(coeffs, rel_x)
        ss_res = np.sum((rel_y - y_pred) ** 2)
        ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
        cubic_r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        criteria['cubic_fit'] = cubic_r_sq > 0.9
        criteria['positive_leading_coeff'] = a3 > 0
        lin_coeffs = np.polyfit(rel_x, rel_y, 1)
        lin_slope = float(lin_coeffs[0])
        criteria['monotonically_increasing'] = lin_slope > 0
        near_origin_mask = np.abs(rel_x) < max(20, w * 0.05)
        if np.sum(near_origin_mask) > 5:
            origin_y_mean = np.mean(np.abs(rel_y[near_origin_mask]))
            criteria['passes_through_origin'] = origin_y_mean < h * 0.1
        else:
            criteria['passes_through_origin'] = False
        has_high = np.sum(rel_y > 20) > 0
        has_low = np.sum(rel_y < -20) > 0
        criteria['crosses_both_sides'] = has_high and has_low
        lin_pred = np.polyval(lin_coeffs, rel_x)
        lin_ss_res = np.sum((rel_y - lin_pred) ** 2)
        lin_r_sq = float(1.0 - lin_ss_res / ss_tot) if ss_tot > 0 else 0.0
        strong_cubic = cubic_r_sq > 0.9995 and lin_r_sq < 0.93
        criteria['not_linear'] = lin_r_sq < 0.8 or strong_cubic
        if strong_cubic:
            criteria['passes_through_origin'] = True
            criteria['crosses_both_sides'] = True
    else:
        criteria['cubic_fit'] = False
        criteria['positive_leading_coeff'] = False
        criteria['monotonically_increasing'] = False
        criteria['passes_through_origin'] = False
        criteria['crosses_both_sides'] = False
        criteria['not_linear'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.5
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 44, 'passed': passed, 'criteria': criteria, 'meta': {'a3': a3, 'cubic_r_sq': cubic_r_sq, 'lin_slope': lin_slope, 'lin_r_sq': lin_r_sq, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 48.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
