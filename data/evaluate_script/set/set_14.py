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
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
from ocr_label_utils import find_labels_in_circles, point_in_circle, verify_circle_and_rect_labels, verify_circle_labels
PROMPT_ID = 14
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CircleI = Tuple[int, int, int]

def red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of red pixels in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (np.array([0, 50, 50]), np.array([10, 255, 255]))
    lower2, upper2 = (np.array([170, 50, 50]), np.array([180, 255, 255]))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def hough_three_circles(gray: np.ndarray, dp: float=1.2, min_dist: float=100.0, param1: float=50.0, param2: float=30.0, min_radius: int=80, max_radius: int=0) -> List[CircleI]:
    """Detect circles by Hough and return exactly three circles as integer tuples, or [] on failure."""
    if max_radius <= 0:
        max_radius = int(gray.shape[0] // 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is None or circles.shape[1] < 3:
        return []
    c = np.uint16(np.around(circles[0, :3]))
    return [(int(x), int(y), int(r)) for x, y, r in c]

def circle_bool_mask(shape_hw: Tuple[int, int], circle: CircleI) -> np.ndarray:
    """Return a boolean filled-circle mask."""
    h, w = shape_hw
    cx, cy, r = circle
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m.astype(bool)

def classify_abc(circles: List[CircleI]) -> Tuple[CircleI, CircleI, CircleI]:
    """
    Classify A/B/C by typical Venn layout:
      - C is the lowest circle (largest y).
      - A and B are the remaining circles; A is left (smaller x), B is right (larger x).
    """
    by_y = sorted(circles, key=lambda c: c[1])
    c_circle = by_y[-1]
    top_two = sorted(by_y[:2], key=lambda c: c[0])
    a_circle, b_circle = (top_two[0], top_two[1])
    return (a_circle, b_circle, c_circle)

def red_ratio_in_region(region_mask: np.ndarray, red_mask: np.ndarray) -> float:
    """Return the fraction of red pixels within a boolean region mask."""
    denom = int(np.sum(region_mask))
    if denom == 0:
        return 0.0
    num = int(np.sum(np.logical_and(region_mask, red_mask > 0)))
    return float(num) / float(denom)

def evaluate(image_path: str, red_on_thresh: float=0.2, red_off_thresh: float=0.15, require_all_regions_present: bool=True) -> Dict[str, object]:
    """
    Target logic:
      Shade (A ∪ B ∪ C) minus (A ∩ B ∩ C) in red.
      The central triple intersection must remain white.

    Equivalent region expectations:
      - A_only: red
      - B_only: red
      - C_only: red
      - A∩B (no C): red
      - A∩C (no B): red
      - B∩C (no A): red
      - A∩B∩C: white

    Criteria:
      C1: File exists and image is readable.
      C2: Three circles detected by Hough.
      C3: Each canonical region matches expected red/white status.
    """
    result: Dict[str, object] = {'criteria': {}, 'passed': False}
    c1_exists = os.path.exists(image_path)
    if not c1_exists:
        result['criteria'] = {'image_exists': False, 'image_readable': False, 'three_circles_detected': False, 'all_regions_correct': False}
        return result
    img = cv2.imread(image_path)
    c1_readable = img is not None
    if not c1_readable:
        result['criteria'] = {'image_exists': True, 'image_readable': False, 'three_circles_detected': False, 'all_regions_correct': False}
        return result
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    circles = hough_three_circles(gray=gray, dp=1.2, min_dist=100.0, param1=50.0, param2=30.0, min_radius=80, max_radius=int(h // 2))
    c2_three = len(circles) == 3
    if not c2_three:
        result['criteria'] = {'image_exists': True, 'image_readable': True, 'three_circles_detected': False, 'all_regions_correct': False}
        return result
    a_circle, b_circle, c_circle = classify_abc(circles)
    try:
        _ocr_hits = find_labels_in_circles(image_path, circles, target_letters=['A', 'B', 'C'])
        _label_map = {}
        _used = set()
        for _letter in ['A', 'B', 'C']:
            _lh = [(l, cx, cy, c) for l, cx, cy, c in _ocr_hits if l == _letter]
            if not _lh:
                continue
            _lh.sort(key=lambda h: h[3], reverse=True)
            _, _lx, _ly, _ = _lh[0]
            _inside = [i for i, c in enumerate(circles) if point_in_circle(_lx, _ly, c)]
            if _inside and _inside[0] not in _used:
                _used.add(_inside[0])
                _label_map[_letter] = _inside[0]
        if len(_label_map) == 3:
            a_circle = circles[_label_map['A']]
            b_circle = circles[_label_map['B']]
            c_circle = circles[_label_map['C']]
    except Exception:
        pass
    rm = red_mask_hsv(img)
    a_mask = circle_bool_mask((h, w), a_circle)
    b_mask = circle_bool_mask((h, w), b_circle)
    c_mask = circle_bool_mask((h, w), c_circle)
    regions: List[Tuple[str, np.ndarray, bool]] = [('A_only', a_mask & ~b_mask & ~c_mask, True), ('B_only', b_mask & ~a_mask & ~c_mask, True), ('C_only', c_mask & ~a_mask & ~b_mask, True), ('A_and_B_no_C', a_mask & b_mask & ~c_mask, True), ('A_and_C_no_B', a_mask & c_mask & ~b_mask, True), ('B_and_C_no_A', b_mask & c_mask & ~a_mask, True), ('A_and_B_and_C', a_mask & b_mask & c_mask, False)]
    per_region_ok: Dict[str, bool] = {}
    for name, region_mask, should_be_red in regions:
        if not np.any(region_mask):
            per_region_ok[name] = not require_all_regions_present
            continue
        ratio = red_ratio_in_region(region_mask, rm)
        if should_be_red:
            per_region_ok[name] = ratio >= red_on_thresh
        else:
            per_region_ok[name] = ratio <= red_off_thresh
    c3_all = all(per_region_ok.values())
    criteria: Dict[str, bool] = {'image_exists': c1_exists, 'image_readable': c1_readable, 'three_circles_detected': c2_three, 'all_regions_correct': c3_all}
    for name, _, _ in regions:
        criteria[f'region_{name}'] = per_region_ok.get(name, False)
    result['criteria'] = criteria
    result['passed'] = all(criteria.values())
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 14.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
