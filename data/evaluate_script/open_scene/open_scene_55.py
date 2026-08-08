#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'plane'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
from open_scene_common import *
import plane_common as _common
from plane_common import *
PROMPT_ID = 55

def extract_blue_mask(img_hsv):
    """Blue hue roughly [95, 130], with reasonable saturation/value."""
    lo = np.array([95, 90, 70])
    hi = np.array([130, 255, 255])
    return cv2.inRange(img_hsv, lo, hi)

def detect_blue_cylinders(img_bgr):
    """Return list of (cx, cy, width, height)."""
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = extract_blue_mask(img_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cyls = []
    min_area = 0.002 * h * w
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area:
            continue
        if cw < 5 or ch < 5:
            continue
        aspect = ch / max(1.0, float(cw))
        if aspect < 0.5 or aspect > 8.0:
            continue
        cx, cy = centroids[i]
        cyls.append((float(cx), float(cy), float(cw), float(ch)))
    cyls.sort(key=lambda b: b[0])
    return cyls

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    cyls = detect_blue_cylinders(img_bgr)
    n = len(cyls)
    criteria = {}
    heights_list = []
    log_fit_slope = log_fit_r2 = None
    widths_cov = None
    criteria['blue_cylinders_detected'] = 4 <= n <= 6
    if n >= 3:
        xs = np.array([b[0] for b in cyls])
        ys = np.array([b[1] for b in cyls])
        ws_ = np.array([b[2] for b in cyls])
        hs_ = np.array([b[3] for b in cyls])
        heights_list = hs_.tolist()
        mean_w = float(np.mean(ws_))
        std_w = float(np.std(ws_))
        widths_cov = std_w / max(1e-06, mean_w)
        criteria['similar_widths'] = widths_cov < 0.3
        x_spread = float(np.max(xs) - np.min(xs))
        y_std = float(np.std(ys))
        criteria['arranged_horizontally'] = x_spread > 0.4 * w and y_std < 0.1 * h
        diffs = np.diff(hs_)
        non_decreases = int(np.sum(diffs >= -5.0))
        strict_increases = int(np.sum(diffs > 0.0))
        criteria['heights_monotonic_increase'] = non_decreases >= n - 1 and strict_increases >= n - 2
        valid = hs_ > 1.0
        if np.sum(valid) >= 3:
            idx = np.arange(n)[valid].astype(float)
            log_h = np.log(hs_[valid])
            coeffs = np.polyfit(idx, log_h, 1)
            log_fit_slope = float(coeffs[0])
            pred = np.polyval(coeffs, idx)
            ss_res = float(np.sum((log_h - pred) ** 2))
            ss_tot = float(np.sum((log_h - np.mean(log_h)) ** 2))
            log_fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-09 else 0.0
            criteria['exponential_growth_fit'] = log_fit_r2 > 0.985 and log_fit_slope > 0.42 and (n == 4)
        else:
            criteria['exponential_growth_fit'] = False
    else:
        criteria['similar_widths'] = False
        criteria['arranged_horizontally'] = False
        criteria['heights_monotonic_increase'] = False
        criteria['exponential_growth_fit'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 67, 'passed': passed, 'criteria': criteria, 'meta': {'num_blobs': n, 'heights': heights_list, 'widths_coefficient_of_variation': widths_cov, 'log_fit_slope': log_fit_slope, 'log_fit_r_squared': log_fit_r2}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 55.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
