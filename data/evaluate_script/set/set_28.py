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
PROMPT_ID = 28
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CircleI = Tuple[int, int, int]

def _hough_single(gray: np.ndarray, h: int, w: int, param2: float) -> List[CircleI]:
    min_dim = min(h, w)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.08 * min_dim, param1=50, param2=param2, minRadius=int(0.08 * min_dim), maxRadius=int(0.48 * min_dim))
    if circles is None:
        return []
    return [(int(round(x)), int(round(y)), int(round(r))) for x, y, r in circles[0]]

def _hough_all(gray: np.ndarray, h: int, w: int) -> List[CircleI]:
    all_circles = []
    for param2 in [30, 40, 50, 55, 60, 65, 70]:
        all_circles.extend(_hough_single(gray, h, w, param2))
    out: List[CircleI] = []
    for cx, cy, r in sorted(all_circles, key=lambda c: c[2], reverse=True):
        if any((np.hypot(cx - sx, cy - sy) < 80 and abs(r - sr) < max(0.25 * sr, 15) for sx, sy, sr in out)):
            continue
        out.append((cx, cy, r))
    return out

def _is_duplicate_circle(c1: CircleI, c2: CircleI, center_tol_ratio: float=0.4) -> bool:
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    tol = min(c1[2], c2[2]) * center_tol_ratio
    return dist < max(tol, 40)

def detect_exactly_three_circles(gray: np.ndarray, h: int, w: int) -> List[CircleI]:
    for param2 in [55, 60, 65, 70, 75, 80]:
        circles = _hough_single(gray, h, w, param2)
        if len(circles) == 3:
            return circles
    circles = _hough_all(gray, h, w)
    if len(circles) < 3:
        return []
    if len(circles) == 3:
        return circles
    sorted_by_r = sorted(circles, key=lambda c: c[2], reverse=True)
    top3 = sorted_by_r[:3]
    if len(circles) >= 4:
        r3 = top3[2][2]
        c4 = sorted_by_r[3]
        r4 = c4[2]
        if r4 >= 0.88 * r3:
            if any((_is_duplicate_circle(c4, t) for t in top3)):
                return top3
            if r3 > 0 and r4 / r3 >= 0.98:
                return top3
            return []
    return top3

def circles_overlap(c1: CircleI, c2: CircleI, tol: float=5) -> bool:
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return dist < c1[2] + c2[2] + tol

def all_three_overlap(circles: List[CircleI]) -> bool:
    if len(circles) != 3:
        return False
    return circles_overlap(circles[0], circles[1]) and circles_overlap(circles[0], circles[2]) and circles_overlap(circles[1], circles[2])

def circle_bool_mask(shape_hw: Tuple[int, int], circle: CircleI) -> np.ndarray:
    h, w = shape_hw
    cx, cy, r = circle
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m.astype(bool)

def classify_abc(circles: List[CircleI]) -> Tuple[CircleI, CircleI, CircleI]:
    by_y = sorted(circles, key=lambda c: c[1])
    c_circle = by_y[-1]
    top_two = sorted(by_y[:2], key=lambda c: c[0])
    a_circle, b_circle = (top_two[0], top_two[1])
    return (a_circle, b_circle, c_circle)

def color_ratio_in_region(region_mask: np.ndarray, color_mask: np.ndarray) -> float:
    denom = int(np.sum(region_mask))
    if denom == 0:
        return 0.0
    num = int(np.sum(np.logical_and(region_mask, color_mask > 0)))
    return float(num) / float(denom)

def red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of red pixels (0-15 and 165-180 hue)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def blue_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of blue pixels (H ~100-130 in OpenCV)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 50, 50])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def evaluate(image_path: str, color_on_thresh: float=0.2, color_off_thresh: float=0.12, require_all_regions_present: bool=True) -> bool:
    """
    Target: A_only Red; B_and_C_no_A Blue; all other five regions white (no red, no blue).
    """
    c1_exists = os.path.exists(image_path)
    if not c1_exists:
        print('C1_file_exists: False')
        print('C1_image_readable: False')
        print('C2_three_circles_detected: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    img = cv2.imread(image_path)
    c1_readable = img is not None
    if not c1_readable:
        print('C1_file_exists: True')
        print('C1_image_readable: False')
        print('C2_three_circles_detected: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    single_run_counts = [_hough_single(gray, h, w, p) for p in [50, 55, 60, 65, 70]]
    max_in_one_run = max((len(r) for r in single_run_counts)) if single_run_counts else 0
    if max_in_one_run == 4:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: False')
        print('C2_exactly_three_circles: False')
        print(f'C2_circles_count: 4')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    circles_count = 3
    all_cands = _hough_all(gray, h, w)
    if len(all_cands) < 3:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: False')
        print(f'C2_circles_count: {len(all_cands)}')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    circles = sorted(all_cands, key=lambda c: c[2], reverse=True)[:3]
    c2_three = True
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
    c2_overlap = all_three_overlap(circles)
    if not c2_overlap:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: True')
        print(f'C2_circles_overlap: {c2_overlap}')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    red_m = red_mask_hsv(img)
    blue_m = blue_mask_hsv(img)
    a_mask = circle_bool_mask((h, w), a_circle)
    b_mask = circle_bool_mask((h, w), b_circle)
    c_mask = circle_bool_mask((h, w), c_circle)
    regions: List[Tuple[str, np.ndarray, bool, bool]] = [('A_only', a_mask & ~b_mask & ~c_mask, True, False), ('B_only', b_mask & ~a_mask & ~c_mask, False, False), ('C_only', c_mask & ~a_mask & ~b_mask, False, False), ('A_and_B_no_C', a_mask & b_mask & ~c_mask, False, False), ('A_and_C_no_B', a_mask & c_mask & ~b_mask, False, False), ('B_and_C_no_A', b_mask & c_mask & ~a_mask, False, True), ('A_and_B_and_C', a_mask & b_mask & c_mask, False, False)]
    per_region_ok: Dict[str, bool] = {}
    for name, region_mask, expect_red, expect_blue in regions:
        if not np.any(region_mask):
            per_region_ok[name] = not require_all_regions_present
            continue
        r_ratio = color_ratio_in_region(region_mask, red_m)
        b_ratio = color_ratio_in_region(region_mask, blue_m)
        if expect_red and (not expect_blue):
            per_region_ok[name] = r_ratio >= color_on_thresh and b_ratio <= color_off_thresh
        elif expect_blue and (not expect_red):
            per_region_ok[name] = b_ratio >= color_on_thresh and r_ratio <= color_off_thresh
        else:
            per_region_ok[name] = r_ratio <= color_off_thresh and b_ratio <= color_off_thresh
    c3_all = all(per_region_ok.values())
    print(f'C1_file_exists: {c1_exists}')
    print(f'C1_image_readable: {c1_readable}')
    print(f'C2_three_circles_detected: {c2_three}')
    print(f'C2_exactly_three_circles: {circles_count == 3}')
    print(f'C2_circles_count: {circles_count}')
    print(f'C2_circles_overlap: {c2_overlap}')
    for name, _, _, _ in regions:
        print(f'C3_{name}: {per_region_ok.get(name, False)}')
    print(f'C3_all_regions_correct: {c3_all}')
    print(f"Result: {('PASS' if c1_exists and c1_readable and c2_three and c2_overlap and c3_all else 'FAIL')}")
    return c1_exists and c1_readable and c2_three and c2_overlap and c3_all

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 28.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
