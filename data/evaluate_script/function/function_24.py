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
PROMPT_ID = 24
sys.path.insert(0, os.path.dirname(__file__))

def _fit_segment(pixels, x_ay, degree=1):
    """Fit polynomial to column medians within pixel set. Returns (coeffs, r2) or (None, None)."""
    if len(pixels) < 5:
        return (None, None)
    col_ys = defaultdict(list)
    for row, col in pixels:
        col_ys[col].append(row)
    cols = sorted(col_ys.keys())
    if len(cols) < 5:
        return (None, None)
    rel_x = np.array([float(c) for c in cols])
    rel_y = np.array([x_ay - np.median(col_ys[c]) for c in cols], dtype=float)
    coeffs = np.polyfit(rel_x, rel_y, degree)
    pred = np.polyval(coeffs, rel_x)
    ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
    r2 = float(1 - np.sum((rel_y - pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0
    return (coeffs, r2)

def _piecewise_segment_checks(blue_pixels, x_ay, y_ax, px_per_unit, buf_ratio=0.2):
    col_neg2 = int(y_ax - 2 * px_per_unit)
    col_pos1 = int(y_ax + 1 * px_per_unit)
    col_pos3 = int(y_ax + 3 * px_per_unit)
    buf = max(3, int(px_per_unit * buf_ratio))
    s1 = blue_pixels[blue_pixels[:, 1] < col_neg2 - buf]
    s2 = blue_pixels[(blue_pixels[:, 1] >= col_neg2 + buf) & (blue_pixels[:, 1] < col_pos1 - buf)]
    s3 = blue_pixels[(blue_pixels[:, 1] >= col_pos1 + buf) & (blue_pixels[:, 1] < col_pos3 - buf)]
    s4 = blue_pixels[blue_pixels[:, 1] >= col_pos3 + buf]
    out = {}
    coeffs2, r2_2 = _fit_segment(s1, x_ay, degree=2)
    coeffs1, r2_1 = _fit_segment(s1, x_ay, degree=1)
    if coeffs2 is not None and r2_2 is not None:
        out['left_curved'] = r2_2 > 0.75 and (r2_1 is None or r2_1 < 0.85 or r2_2 - r2_1 > 0.02)
    else:
        out['left_curved'] = False
    coeffs_s2, r2_s2 = _fit_segment(s2, x_ay, degree=1)
    if coeffs_s2 is not None:
        out['midleft_falling'] = float(coeffs_s2[0]) < -0.1 and r2_s2 > 0.65
    else:
        out['midleft_falling'] = False
    if len(s3) > 5:
        std_row = float(np.std(s3[:, 0]))
        mean_rel_y = float(np.mean(x_ay - s3[:, 0]))
        out['flat_segment'] = std_row < 14 and mean_rel_y > 3
    else:
        out['flat_segment'] = False
    coeffs_s4_3, r2_s4_3 = _fit_segment(s4, x_ay, degree=3)
    coeffs_s4_1, r2_s4_1 = _fit_segment(s4, x_ay, degree=1)
    if r2_s4_3 is not None and r2_s4_1 is not None:
        strong_cubic = r2_s4_3 > 0.98 and coeffs_s4_3 is not None and (abs(float(coeffs_s4_3[0])) > 1e-05)
        out['right_nonlinear'] = r2_s4_3 > 0.7 and r2_s4_1 < 0.9 or strong_cubic
    else:
        out['right_nonlinear'] = len(s4) > 5
    return out

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
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    if len(blue_pixels) > 50:
        segment_criteria = _piecewise_segment_checks(blue_pixels, x_ay, y_ax, w / 12.0)
        if not all(segment_criteria.values()):
            blue_span = float(np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1]))
            if blue_span > 0:
                fallback = _piecewise_segment_checks(blue_pixels, x_ay, y_ax, blue_span / 12.0, buf_ratio=0.15)
                for key, value in fallback.items():
                    segment_criteria[key] = segment_criteria[key] or value
        criteria.update(segment_criteria)
    else:
        for k in ['left_curved', 'midleft_falling', 'flat_segment', 'right_nonlinear']:
            criteria[k] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 95, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'red_pixels': red_pixel_count, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 24.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
