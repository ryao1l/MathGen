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
PROMPT_ID = 40
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
    """Detect circles by Hough and return exactly three circles, or [] on failure."""
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

def circles_overlap(c1: CircleI, c2: CircleI, overlap_margin_ratio: float=0.15) -> bool:
    """Return True if two circles overlap, allowing Hough radii to overshoot thick outlines."""
    dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    margin = overlap_margin_ratio * min(c1[2], c2[2])
    return dist < c1[2] + c2[2] - margin

def classify_disjoint(circles: List[CircleI]) -> Optional[Tuple[CircleI, CircleI, CircleI]]:
    """
    Classify circles into (A, B, C) where A and B overlap, and C is disjoint from both.
    Returns (A, B, C) or None if no valid classification exists.
    A is left (smaller x), B is right (larger x) among the overlapping pair.
    """
    for i in range(3):
        others = [circles[j] for j in range(3) if j != i]
        candidate_c = circles[i]
        if not circles_overlap(others[0], others[1]):
            continue
        if circles_overlap(candidate_c, others[0]) or circles_overlap(candidate_c, others[1]):
            continue
        ab = sorted(others, key=lambda c: c[0])
        return (ab[0], ab[1], candidate_c)
    return None

def red_ratio_in_region(region_mask: np.ndarray, red_mask: np.ndarray) -> float:
    """Return the fraction of red pixels within a boolean region mask."""
    denom = int(np.sum(region_mask))
    if denom == 0:
        return 0.0
    num = int(np.sum(np.logical_and(region_mask, red_mask > 0)))
    return float(num) / float(denom)

def evaluate(image_path: str, red_on_thresh: float=0.2, red_off_thresh: float=0.1) -> bool:
    """
    Target: two overlapping circles A, B with disjoint circle C.
    Shade (A ∩ B) ∪ C in red.
    Expected: A∩B red, entire C red, A-only white, B-only white.
    """
    c1_exists = os.path.exists(image_path)
    if not c1_exists:
        print('C1_file_exists: False')
        print('C1_image_readable: False')
        print('C2_three_circles_detected: False')
        print('C3_disjoint_layout: False')
        print('C4_all_regions_correct: False')
        print('Result: FAIL')
        return False
    img = cv2.imread(image_path)
    c1_readable = img is not None
    if not c1_readable:
        print('C1_file_exists: True')
        print('C1_image_readable: False')
        print('C2_three_circles_detected: False')
        print('C3_disjoint_layout: False')
        print('C4_all_regions_correct: False')
        print('Result: FAIL')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    noisy_edges = cv2.Canny(noisy_inv, 50, 150)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colorful = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 100)).astype(np.uint8) * 255
    dark_density = cv2.countNonZero(noisy_inv) / float(h * w)
    edge_density = cv2.countNonZero(noisy_edges) / float(h * w)
    color_density = cv2.countNonZero(colorful) / float(h * w)
    if color_density > 0.5 or (dark_density > 0.25 and edge_density > 0.12):
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: False')
        print('C2_quality_reject: True')
        print('C3_disjoint_layout: False')
        print('C4_all_regions_correct: False')
        print('Result: FAIL')
        return False
    gray = cv2.medianBlur(gray, 7)
    circles = hough_three_circles(gray=gray, dp=1.2, min_dist=100.0, param1=50.0, param2=30.0, min_radius=80, max_radius=int(h // 2))
    c2_three = len(circles) == 3
    if not c2_three:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: False')
        print('C3_disjoint_layout: False')
        print('C4_all_regions_correct: False')
        print('Result: FAIL')
        return False
    classification = classify_disjoint(circles)
    c3_disjoint = classification is not None
    if not c3_disjoint:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: True')
        print('C3_disjoint_layout: False')
        print('C4_all_regions_correct: False')
        print('Result: FAIL')
        return False
    a_circle, b_circle, c_circle = classification
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
    regions: List[Tuple[str, np.ndarray, bool, bool]] = [('A_only', a_mask & ~b_mask & ~c_mask, False, True), ('B_only', b_mask & ~a_mask & ~c_mask, False, True), ('A_and_B', a_mask & b_mask, True, True), ('C_entire', c_mask & ~a_mask & ~b_mask, True, True)]
    per_region_ok: Dict[str, bool] = {}
    for name, region_mask, should_be_red, must_be_nonempty in regions:
        area = int(np.sum(region_mask))
        if area == 0:
            if must_be_nonempty:
                per_region_ok[name] = False
                print(f'  {name}: FAIL (empty but required)')
            else:
                per_region_ok[name] = True
            continue
        ratio = red_ratio_in_region(region_mask, rm)
        if should_be_red:
            ok = ratio >= red_on_thresh
        else:
            ok = ratio <= red_off_thresh
        per_region_ok[name] = ok
        print(f'  {name}: area={area}, red_ratio={ratio:.4f}, expected_red={should_be_red}, ok={ok}')
    c4_all = all(per_region_ok.values())
    print(f'C1_file_exists: {c1_exists}')
    print(f'C1_image_readable: {c1_readable}')
    print(f'C2_three_circles_detected: {c2_three}')
    print(f'C3_disjoint_layout: {c3_disjoint}')
    for name, _, _, _ in regions:
        print(f'C4_{name}: {per_region_ok.get(name, False)}')
    print(f'C4_all_regions_correct: {c4_all}')
    passed = c1_exists and c1_readable and c2_three and c3_disjoint and c4_all
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 40.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
