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
PROMPT_ID = 26
sys.path.insert(0, os.path.dirname(__file__))
GOLD_H_LO, GOLD_H_HI = (15, 45)
GOLD_S_MIN = 30
GOLD_V_MIN = 100

def extract_gold_mask(img_hsv):
    """Detect gold/yellow shaded region."""
    lower = np.array([GOLD_H_LO, GOLD_S_MIN, GOLD_V_MIN])
    upper = np.array([GOLD_H_HI, 255, 255])
    return cv2.inRange(img_hsv, lower, upper)

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
    gold_mask = extract_gold_mask(img_hsv)
    red_pixel_count = int(np.sum(red_mask > 0))
    blue_pixels = np.column_stack(np.where(blue_mask > 0))
    gold_pixels = np.column_stack(np.where(gold_mask > 0))
    gold_pixel_count = len(gold_pixels)
    vlt_ok, vlt_detail = check_vertical_line_test(blue_mask)
    if not vlt_ok:
        return {'id': 23, 'passed': False, 'criteria': {'vertical_line_test': False}, 'meta': {'vertical_line_test': vlt_detail}}
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = True
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 300
    criteria['gold_shading_exists'] = gold_pixel_count > 50000
    curve_above_axis = False
    if len(blue_pixels) > 50:
        ya = int(round(x_ay))
        above_count = np.sum(blue_pixels[:, 0] <= ya + 5)
        curve_above_axis = above_count > 0.7 * len(blue_pixels)
    criteria['curve_above_axis'] = curve_above_axis
    shading_centered = False
    if gold_pixel_count > 100:
        gold_center_x = float(np.median(gold_pixels[:, 1]))
        shading_centered = abs(gold_center_x - y_ax) < 0.25 * w
    criteria['shading_centered'] = shading_centered
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
    return {'id': 23, 'passed': passed, 'criteria': criteria, 'meta': {'gold_pixels': gold_pixel_count, 'red_pixels': red_pixel_count, 'arrow_info': arrow_info, 'x_axis_labels': x_label_detail, 'y_axis_labels': y_label_detail, 'vertical_line_test': vlt_detail}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 26.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
