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
PROMPT_ID = 16
sys.path.insert(0, os.path.dirname(__file__))

def _count_axis_crossings(blue_mask, x_axis_y, w):
    """Count how many times the blue curve crosses the x-axis."""
    strip_h = 15
    top = max(0, int(x_axis_y) - strip_h)
    bot = min(blue_mask.shape[0], int(x_axis_y) + strip_h)
    col_has_blue = np.any(blue_mask[top:bot, :] > 0, axis=0)
    changes = np.diff(col_has_blue.astype(int))
    starts = np.where(changes == 1)[0]
    crossing_count = len(starts) + (1 if col_has_blue[0] else 0)
    return int(crossing_count)

def _count_local_extrema(blue_mask, x_axis_y, w):
    """Estimate number of peaks and valleys."""
    profile = []
    step = max(1, w // 200)
    for col in range(0, w, step):
        rows = np.where(blue_mask[:, col] > 0)[0]
        if len(rows) > 0:
            profile.append(float(np.median(rows)))
        else:
            profile.append(None)
    valid = [(i, y) for i, y in enumerate(profile) if y is not None]
    if len(valid) < 10:
        return 0
    values = np.array([x_axis_y - v[1] for v in valid])
    if len(values) > 7:
        k = min(7, len(values) // 3)
        if k % 2 == 0:
            k += 1
        if k >= 3:
            kernel = np.ones(k) / k
            values = np.convolve(values, kernel, mode='valid')
    if len(values) < 3:
        return 0
    diffs = np.diff(values)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    return sign_changes

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
    crossings = extrema = 0
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    x_labels, x_ocr_err = ocr_axis_labels(img_bgr, x_ay, y_ax, axis='x')
    if x_ocr_err:
        x_label_detail = {'error': x_ocr_err}
        x_axis_even_tick_labels = False
    else:
        x_labels_ok, x_label_detail = check_tick_labels(x_labels, axis='x')
        values = x_label_detail.get('values', [])
        x_axis_even_tick_labels = len(values) >= 5 and all((abs(v % 2) < 1e-06 for v in values)) and (min(values) <= -4) and (max(values) >= 4)
        x_axis_even_tick_labels = bool(x_labels_ok and x_axis_even_tick_labels)
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        crossings = _count_axis_crossings(blue_mask, x_ay, w)
        criteria['multiple_crossings'] = crossings >= 2
        criteria['full_sine_period_crossings'] = crossings == 5 or (x_axis_even_tick_labels and crossings == 4)
        y_range = np.max(rel_y) - np.min(rel_y)
        criteria['bounded_amplitude'] = y_range / h < 0.72
        has_positive = np.sum(rel_y > 5) > 20
        has_negative = np.sum(rel_y < -5) > 20
        criteria['oscillates_both_sides'] = has_positive and has_negative
        extrema = _count_local_extrema(blue_mask, x_ay, w)
        criteria['multiple_extrema'] = extrema >= 3
        ss_tot = np.sum((rel_y - np.mean(rel_y)) ** 2)
        if ss_tot > 0:
            lin_coeffs = np.polyfit(rel_x, rel_y, 1)
            lin_pred = np.polyval(lin_coeffs, rel_x)
            lin_ss_res = np.sum((rel_y - lin_pred) ** 2)
            lin_r_sq = 1.0 - lin_ss_res / ss_tot
            criteria['not_linear'] = lin_r_sq < 0.5
        else:
            criteria['not_linear'] = False
    else:
        criteria['multiple_crossings'] = False
        criteria['bounded_amplitude'] = False
        criteria['oscillates_both_sides'] = False
        criteria['multiple_extrema'] = False
        criteria['not_linear'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 14, 'passed': passed, 'criteria': criteria, 'meta': {'crossings': crossings, 'extrema': extrema, 'red_pixels': red_pixel_count, 'arrow_info': arrow_info, 'x_axis_labels': x_label_detail}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 16.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
