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
PROMPT_ID = 21
sys.path.insert(0, os.path.dirname(__file__))

def _fallback_y_axis_x(img_gray: np.ndarray) -> float | None:
    h, w = img_gray.shape[:2]
    dark = img_gray < 100
    x0, x1 = (int(0.15 * w), int(0.85 * w))
    y0, y1 = (int(0.08 * h), int(0.9 * h))
    if x1 <= x0 or y1 <= y0:
        return None
    counts = dark[y0:y1, x0:x1].sum(axis=0)
    if counts.size == 0:
        return None
    best = int(np.argmax(counts))
    if counts[best] < 0.35 * (y1 - y0):
        return None
    return float(x0 + best)

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x_ay, y_ax = detect_axes(img_gray)
    fallback_y_ax = None
    if y_ax < 0.05 * w or y_ax > 0.95 * w:
        fallback_y_ax = _fallback_y_axis_x(img_gray)
        if fallback_y_ax is not None:
            y_ax = fallback_y_ax
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
        rel_x = blue_pixels[:, 1] - y_ax
        rel_y = x_ay - blue_pixels[:, 0]
        left_mask = rel_x < -15
        if np.sum(left_mask) > 30:
            lx = rel_x[left_mask]
            ly = rel_y[left_mask]
            coeffs1 = np.polyfit(lx, ly, 1)
            pred1 = np.polyval(coeffs1, lx)
            ss_res = np.sum((ly - pred1) ** 2)
            ss_tot = np.sum((ly - np.mean(ly)) ** 2)
            r2_left = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            criteria['left_branch_linear'] = r2_left > 0.9
            criteria['left_branch_negative_slope'] = coeffs1[0] < -0.2
        else:
            criteria['left_branch_linear'] = False
            criteria['left_branch_negative_slope'] = False
        right_mask = rel_x > 15
        if np.sum(right_mask) > 30:
            rx = rel_x[right_mask]
            ry = rel_y[right_mask]
            coeffs2 = np.polyfit(rx, ry, 2)
            pred2 = np.polyval(coeffs2, rx)
            ss_res2 = np.sum((ry - pred2) ** 2)
            ss_tot_r = np.sum((ry - np.mean(ry)) ** 2)
            r2_quad = 1.0 - ss_res2 / ss_tot_r if ss_tot_r > 0 else 0.0
            coeffs1r = np.polyfit(rx, ry, 1)
            pred1r = np.polyval(coeffs1r, rx)
            ss_res1r = np.sum((ry - pred1r) ** 2)
            r2_lin_r = 1.0 - ss_res1r / ss_tot_r if ss_tot_r > 0 else 0.0
            strong_quadratic = r2_quad > 0.999 and coeffs2[0] > 0.002 and (r2_quad - r2_lin_r > 0.04)
            criteria['right_branch_curved'] = r2_quad > 0.85 and r2_lin_r < 0.8 or strong_quadratic
            criteria['right_branch_opens_up'] = coeffs2[0] > 0
        else:
            criteria['right_branch_curved'] = False
            criteria['right_branch_opens_up'] = False
        above_frac = float(np.sum(rel_y >= -10)) / len(rel_y)
        criteria['mostly_above_axis'] = above_frac > 0.8
        near_origin = np.abs(rel_x) < 15
        if np.sum(near_origin) > 5:
            median_y_origin = float(np.median(rel_y[near_origin]))
            criteria['continuous_at_origin'] = abs(median_y_origin) < 20
        else:
            criteria['continuous_at_origin'] = False
    else:
        criteria['left_branch_linear'] = False
        criteria['left_branch_negative_slope'] = False
        criteria['right_branch_curved'] = False
        criteria['right_branch_opens_up'] = False
        criteria['mostly_above_axis'] = False
        criteria['continuous_at_origin'] = False
    if len(blue_pixels) > 0:
        coverage = (np.max(blue_pixels[:, 1]) - np.min(blue_pixels[:, 1])) / w
        criteria['domain_coverage'] = coverage > 0.5
    else:
        criteria['domain_coverage'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 59, 'passed': passed, 'criteria': criteria, 'meta': {'x_axis_y': x_ay, 'y_axis_x': y_ax, 'fallback_y_axis_x': fallback_y_ax, 'red_pixels': red_pixel_count, 'arrow_info': arrow_info}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 21.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
