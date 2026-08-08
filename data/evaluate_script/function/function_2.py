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
PROMPT_ID = 2
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
        return {'id': 29, 'passed': False, 'criteria': {'vertical_line_test': False}, 'meta': {'vertical_line_test': vlt_detail}}
    criteria = {}
    slope = intercept = y_std = None
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    if len(blue_pixels) > 50:
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        y_std = float(np.std(rel_y))
        criteria['is_horizontal'] = y_std < 10
        coeffs = np.polyfit(rel_x, rel_y, 1)
        slope, intercept = (float(coeffs[0]), float(coeffs[1]))
        criteria['slope_near_zero'] = abs(slope) < 0.1
        mean_rel_y = float(np.mean(rel_y))
        criteria['above_x_axis'] = mean_rel_y > 10
        axis_to_top = x_ay
        if axis_to_top > 0:
            dist_above = float(np.mean(rel_y))
            position_ratio = dist_above / axis_to_top
            criteria['correct_y_position'] = 0.1 < position_ratio < 0.7
        else:
            criteria['correct_y_position'] = False
    else:
        criteria['is_horizontal'] = False
        criteria['slope_near_zero'] = False
        criteria['above_x_axis'] = False
        criteria['correct_y_position'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.6
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 29, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'red_pixels': red_pixel_count, 'slope': slope, 'intercept': intercept, 'y_std_px': y_std, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 2.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
