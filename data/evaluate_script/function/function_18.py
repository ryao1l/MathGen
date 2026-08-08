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
PROMPT_ID = 18
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
        return {'id': 25, 'passed': False, 'criteria': {'vertical_line_test': False}, 'meta': {'vertical_line_test': vlt_detail}}
    criteria = {}
    arrow_info = check_single_headed_arrows(img_gray, x_ay, y_ax)
    criteria['single_headed_arrows'] = arrow_info['is_single_headed']
    criteria['blue_curve_exists'] = len(blue_pixels) > 100
    criteria['no_red_hallucination'] = red_pixel_count < 50
    segments_ok = False
    if len(blue_pixels) > 50:
        ya = int(round(x_ay))
        plot_left = int(y_ax - 0.5 * w * 0.8)
        plot_right = int(y_ax + 0.5 * w * 0.8)
        plot_range = max(1, plot_right - plot_left)
        s1_x0 = plot_left + int(0.5 / 8.0 * plot_range)
        s1_x1 = plot_left + int(2.5 / 8.0 * plot_range)
        s3_x0 = plot_left + int(6.5 / 8.0 * plot_range)
        s3_x1 = plot_left + int(7.5 / 8.0 * plot_range)
        s1_pixels = blue_pixels[(blue_pixels[:, 1] >= s1_x0) & (blue_pixels[:, 1] < s1_x1)]
        s3_pixels = blue_pixels[(blue_pixels[:, 1] >= s3_x0) & (blue_pixels[:, 1] <= s3_x1)]
        s1_flat = len(s1_pixels) > 10 and np.std(s1_pixels[:, 0]) < 15 if len(s1_pixels) > 10 else False
        s3_flat = len(s3_pixels) > 10 and np.std(s3_pixels[:, 0]) < 15 if len(s3_pixels) > 10 else False
        s1_below = len(s1_pixels) > 5 and np.median(s1_pixels[:, 0]) > ya if len(s1_pixels) > 5 else False
        s3_above = len(s3_pixels) > 5 and np.median(s3_pixels[:, 0]) < ya if len(s3_pixels) > 5 else False
        segments_ok = s1_flat and s3_flat and s1_below and s3_above
    criteria['three_segments'] = segments_ok
    level_ok = False
    if len(blue_pixels) > 50:
        third = w // 3
        left_band = blue_pixels[blue_pixels[:, 1] < third]
        right_band = blue_pixels[blue_pixels[:, 1] > 2 * third]
        if len(left_band) > 5 and len(right_band) > 5:
            left_y_median = float(np.median(left_band[:, 0]))
            right_y_median = float(np.median(right_band[:, 0]))
            level_ok = left_y_median > right_y_median
    criteria['y_levels_correct'] = level_ok
    criteria['vertical_line_test'] = True
    x_labels_ok, x_label_detail = check_axis_labels_or_ticks(img_gray, img_bgr, x_ay, y_ax, axis='x')
    criteria['x_axis_labels_correct'] = x_labels_ok
    y_labels_ok, y_label_detail = check_axis_labels_or_ticks(img_gray, img_bgr, x_ay, y_ax, axis='y')
    criteria['y_axis_labels_correct'] = y_labels_ok
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.5
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 25, 'passed': passed, 'criteria': criteria, 'meta': {'red_pixels': red_pixel_count, 'arrow_info': arrow_info, 'vertical_line_test': vlt_detail, 'x_axis_labels': x_label_detail, 'y_axis_labels': y_label_detail}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 18.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
