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
PROMPT_ID = 23
sys.path.insert(0, os.path.dirname(__file__))

def _segment_colmedians(blue_pixels, x_ay, col_lo, col_hi):
    """Get column medians for pixels in col range [col_lo, col_hi]."""
    region = blue_pixels[(blue_pixels[:, 1] >= col_lo) & (blue_pixels[:, 1] < col_hi)]
    if len(region) < 5:
        return (None, None)
    col_ys = defaultdict(list)
    for row, col in region:
        col_ys[col].append(row)
    cols = sorted(col_ys.keys())
    rel_x = np.array([float(c) for c in cols])
    rel_y = np.array([x_ay - np.median(col_ys[c]) for c in cols], dtype=float)
    return (rel_x, rel_y)

def _hollow_marker_near(img_bgr, center_col, center_row, radius):
    """Detect a hollow endpoint marker: visible ring with a mostly light center."""
    h, w = img_bgr.shape[:2]
    c = int(round(center_col))
    r = int(round(center_row))
    rad = max(8, int(round(radius)))
    y0, y1 = (max(0, r - rad), min(h, r + rad + 1))
    x0, x1 = (max(0, c - rad), min(w, c + rad + 1))
    if y1 <= y0 or x1 <= x0:
        return False
    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yy, xx = np.indices(gray.shape)
    cy = r - y0
    cx = c - x0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dark = gray < 120
    blue = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 155) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
    ink = dark | blue
    center = dist <= max(3, rad * 0.35)
    ring = (dist >= rad * 0.55) & (dist <= rad * 1.05)
    if np.sum(center) == 0 or np.sum(ring) == 0:
        return False
    center_ink = float(np.mean(ink[center]))
    ring_ink = float(np.mean(ink[ring]))
    return ring_ink > 0.08 and center_ink < 0.2

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
        px_per_unit = w / 10.0
        col_x0 = y_ax
        col_x2 = y_ax + 2 * px_per_unit
        s1_lo = int(y_ax - 3.0 * px_per_unit)
        s1_hi = int(y_ax - 0.3 * px_per_unit)
        s2_lo = int(y_ax + 0.2 * px_per_unit)
        s2_hi = int(y_ax + 1.8 * px_per_unit)
        s3_lo = int(y_ax + 2.5 * px_per_unit)
        s3_hi = int(y_ax + 5.5 * px_per_unit)
        s1_px = blue_pixels[(blue_pixels[:, 1] >= s1_lo) & (blue_pixels[:, 1] < s1_hi)]
        s2_px = blue_pixels[(blue_pixels[:, 1] >= s2_lo) & (blue_pixels[:, 1] < s2_hi)]
        s3_px = blue_pixels[(blue_pixels[:, 1] >= s3_lo) & (blue_pixels[:, 1] < s3_hi)]
        criteria['three_segments'] = len(s1_px) > 5 and len(s2_px) > 3 and (len(s3_px) > 5)
        if len(s2_px) > 3:
            std_row = float(np.std(s2_px[:, 0]))
            criteria['middle_flat'] = std_row < 12
        else:
            criteria['middle_flat'] = False
        rx1, ry1 = _segment_colmedians(s1_px, x_ay, s1_lo, s1_hi)
        if rx1 is not None and len(rx1) >= 3:
            c1 = np.polyfit(rx1, ry1, 1)
            criteria['left_positive_slope'] = float(c1[0]) > 0.15
        else:
            criteria['left_positive_slope'] = False
        rx3, ry3 = _segment_colmedians(s3_px, x_ay, s3_lo, s3_hi)
        if rx3 is not None and len(rx3) >= 3:
            c3 = np.polyfit(rx3, ry3, 1)
            criteria['right_negative_slope'] = float(c3[0]) < -0.15
        else:
            criteria['right_negative_slope'] = False
        if len(s2_px) > 3:
            mid_rel_y = float(np.mean(x_ay - s2_px[:, 0]))
            criteria['middle_above_axis'] = mid_rel_y > 5
            middle_row = float(np.median(s2_px[:, 0]))
        else:
            criteria['middle_above_axis'] = False
            middle_row = None
        if middle_row is not None:
            marker_radius = max(10.0, 0.1 * px_per_unit)
            hollow_at_0 = _hollow_marker_near(img_bgr, col_x0, middle_row, marker_radius)
            hollow_at_2 = _hollow_marker_near(img_bgr, col_x2, middle_row, marker_radius)
            criteria['hollow_excluded_endpoints'] = hollow_at_0 and hollow_at_2
        else:
            hollow_at_0 = hollow_at_2 = False
            criteria['hollow_excluded_endpoints'] = False
    else:
        for k in ['three_segments', 'middle_flat', 'left_positive_slope', 'right_negative_slope', 'middle_above_axis', 'hollow_excluded_endpoints']:
            criteria[k] = False
        hollow_at_0 = hollow_at_2 = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.5
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 88, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'red_pixels': red_pixel_count, 'arrow_info': arrow_info, 'hollow_at_0': hollow_at_0, 'hollow_at_2': hollow_at_2}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 23.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
