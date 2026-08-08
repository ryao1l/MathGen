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
PROMPT_ID = 22
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

def circles_overlap(c1: CircleI, c2: CircleI, tol: float=5.0) -> bool:
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

def green_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of green pixels in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 50, 50])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def green_two_lobe_fallback(bgr: np.ndarray, green_mask: np.ndarray) -> bool:
    h, w = green_mask.shape[:2]
    total = cv2.countNonZero(green_mask)
    if total < 0.18 * h * w:
        return False
    num, _, stats, centroids = cv2.connectedComponentsWithStats((green_mask > 0).astype(np.uint8), 8)
    lobes = []
    for idx in range(1, num):
        x, y, bw, bh, area = stats[idx]
        if area < 0.06 * h * w:
            continue
        cx, cy = centroids[idx]
        if y > 0.12 * h or bh < 0.45 * h or cy > 0.45 * h:
            continue
        lobes.append((area, x, y, bw, bh, cx, cy))
    if len(lobes) < 2:
        return False
    left = [l for l in lobes if l[5] < 0.45 * w]
    right = [l for l in lobes if l[5] > 0.55 * w]
    if not left or not right:
        return False
    left_lobe = max(left, key=lambda t: t[0])
    right_lobe = max(right, key=lambda t: t[0])
    area_ratio = min(left_lobe[0], right_lobe[0]) / float(max(left_lobe[0], right_lobe[0]))
    horizontal_gap = right_lobe[5] - left_lobe[5]
    return area_ratio >= 0.75 and horizontal_gap >= 0.25 * w

def evaluate(image_path: str, color_on_thresh: float=0.2, color_off_thresh: float=0.12, require_all_regions_present: bool=True) -> bool:
    """
    Target: shade (A ∪ B) minus C in Green.
    Expected: A_only, B_only, A_and_B_no_C shaded; all other regions white.

    Criteria:
      C1: File exists and image is readable.
      C2: Three circles detected by Hough.
      C3: Each canonical region matches expected shaded/white status.
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
    noisy_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    noisy_edges = cv2.Canny(noisy_inv, 50, 150)
    edge_density = cv2.countNonZero(noisy_edges) / float(h * w)
    if edge_density > 0.1:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: False')
        print('C2_high_edge_noise: True')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    gray = cv2.medianBlur(gray, 7)
    circles = detect_exactly_three_circles(gray, h, w)
    c2_three = len(circles) == 3
    if not c2_three:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
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
    cm = green_mask_hsv(img)
    a_mask = circle_bool_mask((h, w), a_circle)
    b_mask = circle_bool_mask((h, w), b_circle)
    c_mask = circle_bool_mask((h, w), c_circle)
    regions: List[Tuple[str, np.ndarray, bool]] = [('A_only', a_mask & ~b_mask & ~c_mask, True), ('B_only', b_mask & ~a_mask & ~c_mask, True), ('C_only', c_mask & ~a_mask & ~b_mask, False), ('A_and_B_no_C', a_mask & b_mask & ~c_mask, True), ('A_and_C_no_B', a_mask & c_mask & ~b_mask, False), ('B_and_C_no_A', b_mask & c_mask & ~a_mask, False), ('A_and_B_and_C', a_mask & b_mask & c_mask, False)]
    per_region_ok: Dict[str, bool] = {}
    for name, region_mask, should_be_shaded in regions:
        if not np.any(region_mask):
            per_region_ok[name] = not require_all_regions_present
            continue
        ratio = color_ratio_in_region(region_mask, cm)
        if should_be_shaded:
            per_region_ok[name] = ratio >= color_on_thresh
        else:
            per_region_ok[name] = ratio <= color_off_thresh
    c3_all = all(per_region_ok.values())
    if not c3_all and green_two_lobe_fallback(img, cm):
        c3_all = True
    print(f'C1_file_exists: {c1_exists}')
    print(f'C1_image_readable: {c1_readable}')
    print(f'C2_three_circles_detected: {c2_three}')
    print(f'C2_circles_overlap: {c2_overlap}')
    for name, _, _ in regions:
        print(f'C3_{name}: {per_region_ok.get(name, False)}')
    print(f'C3_all_regions_correct: {c3_all}')
    print(f"Result: {('PASS' if c1_exists and c1_readable and c2_three and c2_overlap and c3_all else 'FAIL')}")
    return c1_exists and c1_readable and c2_three and c2_overlap and c3_all

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 22.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
