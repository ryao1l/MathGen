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
PROMPT_ID = 54

def extract_blue_water_mask(img_hsv):
    """Vivid blue water: H in [95,130], S >= 90, V >= 70."""
    lower = np.array([95, 90, 70])
    upper = np.array([130, 255, 255])
    return cv2.inRange(img_hsv, lower, upper)

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = extract_blue_water_mask(img_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ys, xs = np.where(mask > 0)
    criteria = {}
    a_coeff = r_sq = lin_r_sq = vertex_x_rel = None
    spread = 0.0
    criteria['blue_water_present'] = len(xs) > 300
    if len(xs) > 100:
        xs_f = xs.astype(float)
        ys_f = ys.astype(float)
        coeffs2 = np.polyfit(xs_f, ys_f, 2)
        a_coeff, b_coeff, c_coeff = map(float, coeffs2)
        y_pred = np.polyval(coeffs2, xs_f)
        ss_res = float(np.sum((ys_f - y_pred) ** 2))
        ss_tot = float(np.sum((ys_f - np.mean(ys_f)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-09 else 0.0
        lin_coeffs = np.polyfit(xs_f, ys_f, 1)
        lin_pred = np.polyval(lin_coeffs, xs_f)
        lin_ss_res = float(np.sum((ys_f - lin_pred) ** 2))
        lin_r_sq = 1.0 - lin_ss_res / ss_tot if ss_tot > 1e-09 else 0.0
        criteria['parabolic_fit'] = r_sq > 0.8
        criteria['arch_opens_down_in_math'] = a_coeff > 0
        criteria['not_linear'] = lin_r_sq < 0.85
        criteria['curvature_plausible'] = 30000 <= len(xs) and 0.0048 <= a_coeff <= 0.011
        if abs(a_coeff) > 1e-09:
            vertex_x = -b_coeff / (2 * a_coeff)
            arc_center_x = (np.min(xs_f) + np.max(xs_f)) / 2
            arc_width = max(1.0, np.max(xs_f) - np.min(xs_f))
            vertex_x_rel = float((vertex_x - arc_center_x) / arc_width)
            criteria['vertex_near_center'] = abs(vertex_x_rel) < 0.25
        else:
            criteria['vertex_near_center'] = False
    else:
        criteria['parabolic_fit'] = False
        criteria['arch_opens_down_in_math'] = False
        criteria['not_linear'] = False
        criteria['vertex_near_center'] = False
        criteria['curvature_plausible'] = False
    if len(xs) > 0:
        spread = (np.max(xs) - np.min(xs)) / w
        criteria['horizontal_spread'] = spread > 0.3
    else:
        criteria['horizontal_spread'] = False
    criteria = {k: bool(v) for k, v in criteria.items()}
    passed = all(criteria.values())
    return {'id': 61, 'passed': passed, 'criteria': criteria, 'meta': {'blue_pixels': int(len(xs)), 'a_coeff': a_coeff, 'r_squared': r_sq, 'linear_r_squared': lin_r_sq, 'vertex_x_relative_to_arc_center': vertex_x_rel, 'horizontal_spread': float(spread)}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen open_scene case 54.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
