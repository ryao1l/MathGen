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
PROMPT_ID = 20
sys.path.insert(0, os.path.dirname(__file__))

def _large_blue_components(blue_mask):
    num, _, stats, _ = cv2.connectedComponentsWithStats((blue_mask > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 250:
            continue
        comps.append({'area': area, 'x': int(stats[i, cv2.CC_STAT_LEFT]), 'y': int(stats[i, cv2.CC_STAT_TOP]), 'w': int(stats[i, cv2.CC_STAT_WIDTH]), 'h': int(stats[i, cv2.CC_STAT_HEIGHT])})
    comps.sort(key=lambda c: c['area'], reverse=True)
    return comps

def _fit_scores(blue_mask):
    pts = np.column_stack(np.where(blue_mask > 0))
    if len(pts) <= 50:
        return None
    xs = pts[:, 1].astype(float)
    ys = pts[:, 0].astype(float)
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    if ss_tot <= 1e-06:
        return None
    coeff1 = np.polyfit(xs, ys, 1)
    pred1 = np.polyval(coeff1, xs)
    r2_line = 1.0 - float(np.sum((ys - pred1) ** 2)) / ss_tot
    coeff2 = np.polyfit(xs, ys, 2)
    pred2 = np.polyval(coeff2, xs)
    r2_quad = 1.0 - float(np.sum((ys - pred2) ** 2)) / ss_tot
    curvature = abs(float(coeff2[0])) / max(abs(float(coeff2[1])), 1e-06)
    return {'r2_line': float(r2_line), 'r2_quad': float(r2_quad), 'curvature': float(curvature), 'coverage': float((xs.max() - xs.min()) / max(1, blue_mask.shape[1])), 'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}

def evaluate(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {'passed': False, 'criteria': {'image_readable': False}, 'meta': {}}
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = extract_blue_mask(img_hsv)
    red_mask = extract_red_mask(img_hsv)
    blue_pixels = int(np.sum(blue_mask > 0))
    red_pixels = int(np.sum(red_mask > 0))
    comps = _large_blue_components(blue_mask)
    scores = _fit_scores(blue_mask)
    major = [c for c in comps if c['area'] >= 500]
    medium = [c for c in comps if c['area'] >= 250]
    major_sorted_x = sorted(major, key=lambda c: c['x'])
    three_piece_layout = False
    if len(major_sorted_x) == 3:
        widths = [c['w'] for c in major_sorted_x]
        heights = [c['h'] for c in major_sorted_x]
        x_gaps = [major_sorted_x[i + 1]['x'] - (major_sorted_x[i]['x'] + major_sorted_x[i]['w']) for i in range(2)]
        three_piece_layout = min(widths) >= 100 and min(heights) >= 120 and (max(x_gaps) < 120) and (min(x_gaps) > -30)
    criteria = {'image_readable': True, 'blue_curve_exists': 2500 <= blue_pixels <= 8000, 'no_red_asymptote': red_pixels < 100, 'three_piece_blue_structure': three_piece_layout, 'no_extra_large_blue_piece': len(major) == 3 and len(medium) <= 4, 'domain_coverage': bool(scores and 0.65 <= scores['coverage'] <= 0.9), 'piecewise_curvature': bool(scores and scores['r2_line'] < 0.35 and (scores['r2_quad'] > 0.8) and (0.0006 <= scores['curvature'] <= 0.003))}
    passed = all((bool(v) for v in criteria.values()))
    return {'passed': passed, 'criteria': {k: bool(v) for k, v in criteria.items()}, 'meta': {'blue_pixels': blue_pixels, 'red_pixels': red_pixels, 'components': comps[:6], 'fit': scores}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 20.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
