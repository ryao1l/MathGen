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
PROMPT_ID = 17
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
    red_pixels = np.column_stack(np.where(red_mask > 0)) if red_pixel_count > 50 else np.array([]).reshape(0, 2)
    criteria = {}
    red_line_x = None
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['red_asymptote_present'] = red_pixel_count > 200
    if len(red_pixels) > 50:
        red_line_x = int(np.median(red_pixels[:, 1]))
        red_rel_x = red_line_x - y_ax
        criteria['asymptote_right_of_y_axis'] = red_rel_x > 10
        criteria['asymptote_is_vertical'] = np.std(red_pixels[:, 1]) < 20
    else:
        criteria['asymptote_right_of_y_axis'] = False
        criteria['asymptote_is_vertical'] = False
    if len(blue_pixels) > 50 and red_line_x is not None:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        left_of_asym = np.sum(blue_pixels[:, 1] < red_line_x - 10)
        total_blue = len(blue_pixels)
        criteria['curve_right_of_asymptote'] = left_of_asym / max(1, total_blue) < 0.05
        step = max(1, len(blue_pixels) // 100)
        sorted_by_x = blue_pixels[blue_pixels[:, 1].argsort()]
        sampled = sorted_by_x[::step]
        if len(sampled) > 3:
            x_vals = sampled[:, 1].astype(float)
            y_vals = (x_ay - sampled[:, 0]).astype(float)
            coeffs = np.polyfit(x_vals, y_vals, 1)
            criteria['monotonically_increasing'] = coeffs[0] > 0
        else:
            criteria['monotonically_increasing'] = False
        has_pos = np.sum(rel_y > 5) > 10
        has_neg = np.sum(rel_y < -5) > 10
        criteria['crosses_x_axis'] = has_pos and has_neg
    elif len(blue_pixels) > 50:
        criteria['curve_right_of_asymptote'] = False
        criteria['monotonically_increasing'] = False
        criteria['crosses_x_axis'] = False
    else:
        criteria['curve_right_of_asymptote'] = False
        criteria['monotonically_increasing'] = False
        criteria['crosses_x_axis'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.3
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 15, 'passed': passed, 'criteria': criteria, 'meta': {'red_pixels': red_pixel_count, 'red_line_x': red_line_x, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 17.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
