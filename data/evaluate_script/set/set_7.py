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
PROMPT_ID = 7
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
    """Treat as same circle if centers are very close relative to radii."""
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    tol = min(c1[2], c2[2]) * center_tol_ratio
    return dist < max(tol, 40)

def detect_two_circle_candidates(gray: np.ndarray, h: int, w: int) -> List[CircleI]:
    """Return candidate circles for two-circle Venn diagram."""
    for param2 in [55, 60, 65, 70, 75, 80]:
        circles = _hough_single(gray, h, w, param2)
        if len(circles) == 2:
            return circles
    circles = _hough_all(gray, h, w)
    if len(circles) < 2:
        return []
    sorted_by_r = sorted(circles, key=lambda c: c[2], reverse=True)
    return sorted_by_r[:8]

def circles_overlap(c1: CircleI, c2: CircleI, tol: float=5) -> bool:
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return dist < c1[2] + c2[2] + tol

def circle_bool_mask(shape_hw: Tuple[int, int], circle: CircleI) -> np.ndarray:
    h, w = shape_hw
    cx, cy, r = circle
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m.astype(bool)

def classify_ab(circles: List[CircleI]) -> Tuple[CircleI, CircleI]:
    """A = left (smaller x), B = right (larger x)."""
    sorted_by_x = sorted(circles, key=lambda c: c[0])
    return (sorted_by_x[0], sorted_by_x[1])

def color_ratio_in_region(region_mask: np.ndarray, color_mask: np.ndarray) -> float:
    denom = int(np.sum(region_mask))
    if denom == 0:
        return 0.0
    num = int(np.sum(np.logical_and(region_mask, color_mask > 0)))
    return float(num) / float(denom)

def pink_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of pink pixels in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([140, 50, 50]), np.array([175, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def shaded_region_mask(bgr: np.ndarray) -> np.ndarray:
    """Broad non-white, non-black fill mask for human-style shaded Venn regions."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 35) & (val > 70)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def shaded_region_fallback(bgr: np.ndarray) -> bool:
    sm = shaded_region_mask(bgr)
    area = int(cv2.countNonZero(sm))
    img_area = float(bgr.shape[0] * bgr.shape[1])
    if area <= 0:
        return False
    area_ratio = area / img_area
    if not 0.01 <= area_ratio <= 0.22:
        return False
    pts = cv2.findNonZero(sm)
    if pts is None:
        return False
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return False
    aspect = w / float(h)
    fill = area / float(max(1, w * h))
    return 0.45 <= aspect <= 1.75 and fill >= 0.28

def evaluate(image_path: str, color_on_thresh: float=0.6, color_off_thresh: float=0.12, require_all_regions_present: bool=True) -> bool:
    """
    Target: shade symmetric difference of A and B (A_only, B_only) in Pink; intersection white.

    Criteria:
      C1: File exists and image is readable.
      C2: Two circles detected by Hough, overlapping.
      C3: Each region matches expected shaded/white status.
    """
    c1_exists = os.path.exists(image_path)
    if not c1_exists:
        print('C1_file_exists: False')
        print('C1_image_readable: False')
        print('C2_two_circles_detected: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    img = cv2.imread(image_path)
    c1_readable = img is not None
    if not c1_readable:
        print('C1_file_exists: True')
        print('C1_image_readable: False')
        print('C2_two_circles_detected: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    max_dim = max(img.shape[:2])
    if max_dim > 1024:
        scale = 1024.0 / float(max_dim)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    noisy_edges = cv2.Canny(noisy_inv, 50, 150)
    dark_density = cv2.countNonZero(noisy_inv) / float(h * w)
    edge_density = cv2.countNonZero(noisy_edges) / float(h * w)
    if dark_density > 0.3 and edge_density > 0.08:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print(f'C2_noise_rejected: True dark={dark_density:.3f} edge={edge_density:.3f}')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    gray = cv2.medianBlur(gray, 7)
    cm = pink_mask_hsv(img)
    fallback_mask = shaded_region_mask(img)
    candidates = detect_two_circle_candidates(gray, h, w)
    best_pair: Tuple[CircleI, CircleI] | None = None
    best_score = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = (candidates[i], candidates[j])
            if not circles_overlap(c1, c2):
                continue
            a_tmp, b_tmp = classify_ab([c1, c2])
            a_m = circle_bool_mask((h, w), a_tmp)
            b_m = circle_bool_mask((h, w), b_tmp)
            active_mask = cm if cv2.countNonZero(cm) > 0 else fallback_mask
            a_only_ratio = color_ratio_in_region(a_m & ~b_m, active_mask)
            b_only_ratio = color_ratio_in_region(b_m & ~a_m, active_mask)
            ab_ratio = color_ratio_in_region(a_m & b_m, active_mask)
            score = min(a_only_ratio, b_only_ratio) - ab_ratio
            if score > best_score:
                best_score = score
                best_pair = (a_tmp, b_tmp)
    c2_two = best_pair is not None
    if not c2_two:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_two_circles_detected: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    a_circle, b_circle = best_pair
    c2_overlap = circles_overlap(a_circle, b_circle)
    if not c2_overlap:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_two_circles_detected: True')
        print(f'C2_circles_overlap: {c2_overlap}')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    visual_fallback = False
    a_mask = circle_bool_mask((h, w), a_circle)
    b_mask = circle_bool_mask((h, w), b_circle)
    regions: List[Tuple[str, np.ndarray, bool]] = [('A_only', a_mask & ~b_mask, True), ('B_only', b_mask & ~a_mask, True), ('A_and_B', a_mask & b_mask, False)]
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
    print(f'C2_two_circles_detected: {c2_two}')
    print(f'C2_circles_overlap: {c2_overlap}')
    for name, _, _ in regions:
        print(f'C3_{name}: {per_region_ok.get(name, False)}')
    print(f'C3_shaded_region_fallback: {visual_fallback}')
    print(f'C3_all_regions_correct: {c3_all}')
    print(f"Result: {('PASS' if c1_exists and c1_readable and c2_two and c2_overlap and c3_all else 'FAIL')}")
    return c1_exists and c1_readable and c2_two and c2_overlap and c3_all

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 7.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
