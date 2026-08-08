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
PROMPT_ID = 9
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
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    y_axis_ok, y_axis_detail = check_axis_labels_or_ticks(img_gray, img_bgr, x_ay, y_ax, axis='y')
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    criteria['y_axis_numbered_ticks'] = y_axis_ok
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        left_mask = rel_x < -10
        right_mask = rel_x > 10
        left_slope = right_slope = None
        left_ok = right_ok = False
        if np.sum(left_mask) > 30:
            lx, ly = (rel_x[left_mask], rel_y[left_mask])
            lcoeffs = np.polyfit(lx, ly, 1)
            left_slope = float(lcoeffs[0])
            left_ok = left_slope < -0.2
        if np.sum(right_mask) > 30:
            rx, ry = (rel_x[right_mask], rel_y[right_mask])
            rcoeffs = np.polyfit(rx, ry, 1)
            right_slope = float(rcoeffs[0])
            right_ok = right_slope > 0.2
        criteria['v_shape_slopes'] = left_ok and right_ok
        if left_slope is not None and right_slope is not None:
            slope_sum = left_slope + right_slope
            criteria['slope_symmetry'] = abs(slope_sum) < 0.4
        else:
            criteria['slope_symmetry'] = False
        above_ratio = np.sum(rel_y >= -15) / len(rel_y)
        criteria['mostly_above_axis'] = above_ratio > 0.85
        lin_coeffs = np.polyfit(rel_x, rel_y, 1)
        lin_pred = np.polyval(lin_coeffs, rel_x)
        ss_res = np.sum((rel_y - lin_pred) ** 2)
        ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
        lin_r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        criteria['not_single_line'] = lin_r_sq < 0.85
        lowest_y_indices = np.where(rel_y < np.min(rel_y) + 15)[0]
        if len(lowest_y_indices) > 0:
            vertex_x_mean = np.mean(rel_x[lowest_y_indices])
            criteria['vertex_near_origin'] = abs(vertex_x_mean) / w < 0.15
        else:
            criteria['vertex_near_origin'] = False
    else:
        criteria['v_shape_slopes'] = False
        criteria['slope_symmetry'] = False
        criteria['mostly_above_axis'] = False
        criteria['not_single_line'] = False
        criteria['vertex_near_origin'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 12, 'passed': passed, 'criteria': criteria, 'meta': {'arrow_info': arrow_info, 'y_axis_detail': y_axis_detail, 'red_pixels': red_pixel_count}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 9.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
