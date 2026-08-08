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
PROMPT_ID = 42
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CircleI = Tuple[int, int, int]

def _hough_single(gray: np.ndarray, h: int, w: int, param2: float) -> List[CircleI]:
    min_dim = min(h, w)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.08 * min_dim, param1=50, param2=param2, minRadius=int(0.08 * min_dim), maxRadius=int(0.48 * min_dim))
    if circles is None:
        return []
    return [(int(round(x)), int(round(y)), int(round(r))) for x, y, r in circles[0]]

def _hough_all(gray: np.ndarray, h: int, w: int) -> List[CircleI]:
    all_circles: List[CircleI] = []
    for param2 in [30, 40, 50, 55, 60, 65, 70]:
        all_circles.extend(_hough_single(gray, h, w, param2))
    out: List[CircleI] = []
    for cx, cy, r in sorted(all_circles, key=lambda c: c[2], reverse=True):
        if any((np.hypot(cx - sx, cy - sy) < 80 and abs(r - sr) < max(0.25 * sr, 15) for sx, sy, sr in out)):
            continue
        out.append((cx, cy, r))
    return out

def detect_three_circle_candidates(gray: np.ndarray, h: int, w: int) -> List[CircleI]:
    for param2 in [55, 60, 65, 70, 75, 80]:
        circles = _hough_single(gray, h, w, param2)
        if len(circles) == 3:
            return circles
    circles = _hough_all(gray, h, w)
    if len(circles) <= 3:
        return circles
    return sorted(circles, key=lambda c: c[2], reverse=True)[:9]

def circles_overlap(c1: CircleI, c2: CircleI, tol: float=5.0) -> bool:
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return dist < c1[2] + c2[2] + tol

def circles_clearly_separate(c1: CircleI, c2: CircleI, min_gap_ratio: float=0.25) -> bool:
    """True only when circles are clearly not overlapping: dist >= r1 + r2 + gap.
    Requires a visible gap of at least min_gap_ratio * min(r1,r2) so that barely-touching or overlapping fails."""
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    min_r = min(c1[2], c2[2])
    gap = min_gap_ratio * min_r
    return dist >= c1[2] + c2[2] + gap

def circle_contains(outer: CircleI, inner: CircleI, tol: float=10.0) -> bool:
    """True if inner circle is fully inside outer (center distance + inner_r <= outer_r)."""
    dist = np.hypot(outer[0] - inner[0], outer[1] - inner[1])
    return dist + inner[2] <= outer[2] + tol

def classify_abc_containment(circles: List[CircleI]) -> Optional[Tuple[CircleI, CircleI, CircleI]]:
    """
    Given 3 circles: two overlap (A, B), one (C) is inside A and does not overlap B.
    Return (A_circle, B_circle, C_circle) or None if no valid assignment.
    """
    if len(circles) != 3:
        return None
    by_r = sorted(circles, key=lambda c: c[2], reverse=True)
    large1, large2, small = (by_r[0], by_r[1], by_r[2])
    if not circles_overlap(large1, large2):
        return None
    if circle_contains(large1, small) and circles_clearly_separate(small, large2):
        return (large1, large2, small)
    if circle_contains(large2, small) and circles_clearly_separate(small, large1):
        return (large2, large1, small)
    return None

def circle_bool_mask(shape_hw: Tuple[int, int], circle: CircleI, scale_r: float=1.0) -> np.ndarray:
    h, w = shape_hw
    cx, cy, r = circle
    r_scaled = max(1, int(round(r * scale_r)))
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r_scaled, 255, -1)
    return m.astype(bool)

def masks_overlap_significant(shape_hw: Tuple[int, int], c1: CircleI, c2: CircleI, scale_r: float=1.2, overlap_ratio_thresh: float=0.01) -> bool:
    """True if scaled circle masks overlap more than thresh of the smaller circle's area (catches drawn overlap)."""
    m1 = circle_bool_mask(shape_hw, c1, scale_r)
    m2 = circle_bool_mask(shape_hw, c2, scale_r)
    overlap = np.sum(m1 & m2)
    area_smaller = min(np.sum(m1), np.sum(m2))
    if area_smaller == 0:
        return False
    return overlap / area_smaller > overlap_ratio_thresh

def color_ratio_in_region(region_mask: np.ndarray, color_mask: np.ndarray) -> float:
    denom = int(np.sum(region_mask))
    if denom == 0:
        return 0.0
    num = int(np.sum(np.logical_and(region_mask, color_mask > 0)))
    return float(num) / float(denom)

def brown_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of brown pixels (dark orange/brown)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([8, 50, 30])
    upper = np.array([25, 255, 180])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def evaluate(image_path: str, color_on_thresh: float=0.2, color_off_thresh: float=0.12, require_all_regions_present: bool=True) -> bool:
    """
    Target: shade A minus C (Circle A minus Circle C) in Brown; C interior and B_only white.
    Geometry: exactly 3 circles — A and B overlap; C fully inside A, C does not overlap B.
    """
    c1_exists = os.path.exists(image_path)
    if not c1_exists:
        print('C1_file_exists: False')
        print('C1_image_readable: False')
        print('C2_three_circles_valid: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    img = cv2.imread(image_path)
    c1_readable = img is not None
    if not c1_readable:
        print('C1_file_exists: True')
        print('C1_image_readable: False')
        print('C2_three_circles_valid: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    cm = brown_mask_hsv(img)
    candidates = detect_three_circle_candidates(gray, h, w)
    best_triplet: Optional[Tuple[CircleI, CircleI, CircleI]] = None
    best_score = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                tri = [candidates[i], candidates[j], candidates[k]]
                classified = classify_abc_containment(tri)
                if classified is None:
                    continue
                a_c, b_c, c_c = classified
                if masks_overlap_significant((h, w), c_c, b_c):
                    continue
                a_m = circle_bool_mask((h, w), a_c)
                b_m = circle_bool_mask((h, w), b_c)
                c_m = circle_bool_mask((h, w), c_c)
                a_minus_c = a_m & ~c_m
                c_interior = c_m
                b_only = b_m & ~a_m
                if require_all_regions_present and (not np.any(a_minus_c) or not np.any(c_interior) or (not np.any(b_only))):
                    continue
                score = color_ratio_in_region(a_minus_c, cm) - color_ratio_in_region(c_interior, cm) - color_ratio_in_region(b_only, cm)
                if score > best_score:
                    best_score = score
                    best_triplet = (a_c, b_c, c_c)
    c2_ok = best_triplet is not None
    if not c2_ok:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_valid: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    a_circle, b_circle, c_circle = best_triplet
    if masks_overlap_significant((h, w), c_circle, b_circle, scale_r=1.35, overlap_ratio_thresh=0.005):
        c2_ok = False
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_valid: False')
        print('C2_C_and_B_overlap_in_image: True')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    a_mask = circle_bool_mask((h, w), a_circle)
    b_mask = circle_bool_mask((h, w), b_circle)
    c_mask = circle_bool_mask((h, w), c_circle)
    a_minus_c = a_mask & ~c_mask
    regions: List[Tuple[str, np.ndarray, bool]] = [('A_minus_C', a_minus_c, True), ('C_interior', c_mask, False), ('B_only', b_mask & ~a_mask, False)]
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
    print(f'C1_file_exists: {c1_exists}')
    print(f'C1_image_readable: {c1_readable}')
    print(f'C2_three_circles_valid: {c2_ok}')
    print(f'C2_A_B_overlap: {circles_overlap(a_circle, b_circle)}')
    print(f'C2_C_inside_A: {circle_contains(a_circle, c_circle)}')
    print(f'C2_C_not_overlap_B: {circles_clearly_separate(c_circle, b_circle)}')
    for name, _, _ in regions:
        print(f'C3_{name}: {per_region_ok.get(name, False)}')
    print(f'C3_all_regions_correct: {c3_all}')
    print(f"Result: {('PASS' if c1_exists and c1_readable and c2_ok and c3_all else 'FAIL')}")
    return c1_exists and c1_readable and c2_ok and c3_all

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 42.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
