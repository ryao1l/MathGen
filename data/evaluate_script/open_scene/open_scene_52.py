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
PROMPT_ID = 52

def extract_ramp_mask(img_bgr):
    """Dark-brown ramp against white surroundings. Use low-V OR low saturation brown detection.

    Strategy: convert to grayscale, threshold darker pixels. Additionally guard
    against capturing shadows by requiring moderate saturation in a wide red/orange hue range.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < 120).astype(np.uint8) * 255
    brownish1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([25, 255, 200]))
    brownish2 = cv2.inRange(hsv, np.array([160, 30, 30]), np.array([179, 255, 200]))
    brownish = cv2.bitwise_or(brownish1, brownish2)
    combined = cv2.bitwise_and(dark, brownish)
    if int(np.sum(combined > 0)) < 1000:
        combined = dark
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    if num <= 1:
        return combined
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    largest = (labels == idx).astype(np.uint8) * 255
    return largest

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    mask = extract_ramp_mask(img_bgr)
    ys, xs = np.where(mask > 0)
    dark_count = int(len(xs))
    criteria = {}
    slope_pix = r_sq = angle_deg = residual_std = None
    criteria['ramp_present'] = dark_count > max(20000, 0.005 * h * w)
    if dark_count > 200:
        col_y_list = []
        for col in range(w):
            rows = np.where(mask[:, col] > 0)[0]
            if len(rows) > 0:
                col_y_list.append((col, float(np.median(rows))))
        if len(col_y_list) >= 30:
            xs_c = np.array([c[0] for c in col_y_list], dtype=float)
            ys_c = np.array([c[1] for c in col_y_list], dtype=float)
            coeffs = np.polyfit(xs_c, ys_c, 1)
            slope_pix = float(coeffs[0])
            intercept_pix = float(coeffs[1])
            pred = np.polyval(coeffs, xs_c)
            ss_res = float(np.sum((ys_c - pred) ** 2))
            ss_tot = float(np.sum((ys_c - np.mean(ys_c)) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-09 else 0.0
            residual_std = float(np.std(ys_c - pred))
            angle_deg = float(np.degrees(np.arctan(abs(slope_pix))))
            criteria['ramp_is_linear'] = r_sq > 0.975
            criteria['slope_angle_valid'] = 25.0 <= angle_deg <= 65.0
            criteria['positive_math_slope'] = slope_pix < -0.05
            span = (np.max(xs_c) - np.min(xs_c)) / w
            criteria['horizontal_span'] = span > 0.62
            criteria['thin_ramp'] = residual_std < 0.025 * h
        else:
            criteria['ramp_is_linear'] = False
            criteria['slope_angle_valid'] = False
            criteria['positive_math_slope'] = False
            criteria['horizontal_span'] = False
            criteria['thin_ramp'] = False
    else:
        criteria['ramp_is_linear'] = False
        criteria['slope_angle_valid'] = False
        criteria['positive_math_slope'] = False
        criteria['horizontal_span'] = False
        criteria['thin_ramp'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 64, 'passed': passed, 'criteria': criteria, 'meta': {'dark_pixel_count': dark_count, 'slope_pixel_coords': slope_pix, 'angle_degrees_from_horizontal': angle_deg, 'linear_r_squared': r_sq, 'residual_std_px': residual_std}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 52.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
