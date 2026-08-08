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
PROMPT_ID = 12
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
    vlt_ok, vlt_detail = check_vertical_line_test(blue_mask)
    if not vlt_ok:
        return {'id': 27, 'passed': False, 'criteria': {'vertical_line_test': False}, 'meta': {'vertical_line_test': vlt_detail}}
    criteria = {}
    a_coeff = r_squared = None
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        coeffs = np.polyfit(rel_x, rel_y, 2)
        a_coeff, b_coeff, c_coeff = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        y_pred = np.polyval(coeffs, rel_x)
        ss_res = np.sum((rel_y - y_pred) ** 2)
        ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        criteria['opens_downward'] = a_coeff < 0
        criteria['quadratic_fit'] = r_squared > 0.9
        if abs(a_coeff) > 1e-06:
            vertex_x = -b_coeff / (2 * a_coeff)
            criteria['vertex_near_center'] = abs(vertex_x) / w < 0.15
        else:
            criteria['vertex_near_center'] = False
        if abs(a_coeff) > 1e-06:
            vertex_y = c_coeff - b_coeff ** 2 / (4 * a_coeff)
            criteria['vertex_near_x_axis'] = abs(vertex_y) / h < 0.1
        else:
            criteria['vertex_near_x_axis'] = False
        below_ratio = np.sum(rel_y <= 15) / len(rel_y)
        criteria['mostly_below_axis'] = below_ratio > 0.85
        lin_coeffs = np.polyfit(rel_x, rel_y, 1)
        lin_pred = np.polyval(lin_coeffs, rel_x)
        lin_ss_res = np.sum((rel_y - lin_pred) ** 2)
        lin_r_sq = float(1.0 - lin_ss_res / ss_tot) if ss_tot > 0 else 0.0
        criteria['not_linear'] = lin_r_sq < 0.85
    else:
        criteria['opens_downward'] = False
        criteria['quadratic_fit'] = False
        criteria['vertex_near_center'] = False
        criteria['vertex_near_x_axis'] = False
        criteria['mostly_below_axis'] = False
        criteria['not_linear'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.5
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 27, 'passed': passed, 'criteria': criteria, 'meta': {'a_coeff': a_coeff, 'r_squared': r_squared, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 12.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
