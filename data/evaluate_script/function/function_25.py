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
PROMPT_ID = 25
sys.path.insert(0, os.path.dirname(__file__))
GRAY_S_MAX = 60
GRAY_V_LO, GRAY_V_HI = (80, 200)

def extract_gray_mask(img_hsv, img_bgr):
    """Detect gray pixels (low saturation, medium brightness) that are not black axes."""
    mask = cv2.inRange(img_hsv, np.array([0, 0, GRAY_V_LO]), np.array([179, GRAY_S_MAX, GRAY_V_HI]))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask[gray < 60] = 0
    mask[gray > 230] = 0
    return mask

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
    gray_mask = extract_gray_mask(img_hsv, img_bgr)
    red_pixel_count = int(np.sum(red_mask > 0))
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    gray_pixel_count = int(np.sum(gray_mask > 0))
    vlt_ok, vlt_detail = check_vertical_line_test(blue_mask)
    if not vlt_ok:
        return {'id': 22, 'passed': False, 'criteria': {'vertical_line_test': False}, 'meta': {'vertical_line_test': vlt_detail}}
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    criteria['gray_envelope_exists'] = gray_pixel_count > 200
    crossings = 0
    if len(blue_pixels) > 50:
        ya = int(round(x_ay))
        num_strips = 20
        strip_w = max(1, w // num_strips)
        prev_above = None
        for i in range(num_strips):
            x0 = i * strip_w
            x1 = min(w, (i + 1) * strip_w)
            strip = blue_mask[:, x0:x1]
            ys = np.where(strip > 0)[0]
            if len(ys) == 0:
                continue
            median_y = float(np.median(ys))
            above = median_y < ya
            if prev_above is not None and above != prev_above:
                crossings += 1
            prev_above = above
    criteria['oscillation_detected'] = crossings >= 3
    damping_ok = False
    if len(blue_pixels) > 50:
        ya = int(round(x_ay))
        mid_x = (y_ax + w) / 2.0
        left_pixels = blue_pixels[blue_pixels[:, 1] < mid_x]
        right_pixels = blue_pixels[blue_pixels[:, 1] >= mid_x]
        if len(left_pixels) > 10 and len(right_pixels) > 10:
            left_amp = float(np.std(left_pixels[:, 0] - ya))
            right_amp = float(np.std(right_pixels[:, 0] - ya))
            damping_ok = left_amp > right_amp * 1.2
    criteria['damping_detected'] = damping_ok
    starts_at_origin = False
    if len(blue_pixels) > 50:
        ya = int(round(x_ay))
        left_window = blue_pixels[(blue_pixels[:, 1] >= y_ax - 5) & (blue_pixels[:, 1] <= y_ax + 0.02 * w)]
        if len(left_window) > 10:
            starts_at_origin = abs(float(np.median(left_window[:, 0])) - ya) < 0.12 * h
    criteria['starts_at_origin'] = starts_at_origin
    criteria['vertical_line_test'] = True
    x_labels_ok, x_label_detail = _tick_structure_ok(img_gray, x_ay, y_ax, axis='x', img_hsv=img_hsv)
    criteria['x_axis_labels_correct'] = x_labels_ok
    y_labels_ok, y_label_detail = _tick_structure_ok(img_gray, x_ay, y_ax, axis='y', img_hsv=img_hsv)
    criteria['y_axis_labels_correct'] = y_labels_ok
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.5
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 22, 'passed': passed, 'criteria': criteria, 'meta': {'gray_pixels': gray_pixel_count, 'red_pixels': red_pixel_count, 'crossings': crossings, 'arrow_info': arrow_info, 'vertical_line_test': vlt_detail, 'x_axis_labels': x_label_detail, 'y_axis_labels': y_label_detail}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 25.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
