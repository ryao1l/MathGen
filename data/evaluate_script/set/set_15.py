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
PROMPT_ID = 15
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CircleI = Tuple[int, int, int]

def red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of red pixels in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (np.array([0, 70, 50]), np.array([10, 255, 255]))
    lower2, upper2 = (np.array([170, 70, 50]), np.array([180, 255, 255]))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def hough_three_circles(gray: np.ndarray, dp: float=1.2, min_dist: float=100.0, param1: float=50.0, param2: float=30.0, min_radius: int=100, max_radius: int=0) -> List[CircleI]:
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

def red_ratio_in_region(region_mask: np.ndarray, red_mask: np.ndarray) -> float:
    """Return the fraction of red pixels within a boolean region mask."""
    denom = int(np.sum(region_mask))
    if denom == 0:
        return 0.0
    num = int(np.sum(np.logical_and(region_mask, red_mask > 0)))
    return float(num) / float(denom)

def evaluate(image_path: str, red_on_thresh: float=0.3, require_all_regions_present: bool=True) -> bool:
    """
    Evaluate a 3-circle Venn diagram with expected coloring:
      - Pairwise intersections (excluding the third circle) are red.
      - Unique-only regions are white.
      - Triple intersection is white.

    Criteria:
      C1: File exists and image is readable.
      C2: Three circles detected by Hough.
      C3: Each of the seven canonical regions matches expected red/white status.
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
    circles = hough_three_circles(gray=gray, dp=1.2, min_dist=100.0, param1=50.0, param2=30.0, min_radius=100, max_radius=int(h // 2))
    c2_three = len(circles) == 3
    if not c2_three:
        print('C1_file_exists: True')
        print('C1_image_readable: True')
        print('C2_three_circles_detected: False')
        print('C3_all_regions_correct: False')
        print('Result: FAIL')
        return False
    rm = red_mask_hsv(img)
    masks = [circle_bool_mask((h, w), c) for c in circles]
    a_mask, b_mask, c_mask = (masks[0], masks[1], masks[2])
    regions: List[Tuple[str, np.ndarray, bool]] = [('A_only', a_mask & ~b_mask & ~c_mask, False), ('B_only', b_mask & ~a_mask & ~c_mask, False), ('C_only', c_mask & ~a_mask & ~b_mask, False), ('A_and_B_no_C', a_mask & b_mask & ~c_mask, True), ('B_and_C_no_A', b_mask & c_mask & ~a_mask, True), ('A_and_C_no_B', a_mask & c_mask & ~b_mask, True), ('A_and_B_and_C', a_mask & b_mask & c_mask, False)]
    per_region_ok: Dict[str, bool] = {}
    for name, region_mask, should_be_red in regions:
        if not np.any(region_mask):
            per_region_ok[name] = not require_all_regions_present
            continue
        ratio = red_ratio_in_region(region_mask, rm)
        is_red = ratio >= red_on_thresh
        per_region_ok[name] = is_red == should_be_red
    c3_all = all(per_region_ok.values())
    print(f'C1_file_exists: {c1_exists}')
    print(f'C1_image_readable: {c1_readable}')
    print(f'C2_three_circles_detected: {c2_three}')
    for name, _, _ in regions:
        print(f'C3_{name}: {per_region_ok.get(name, False)}')
    print(f'C3_all_regions_correct: {c3_all}')
    print(f"Result: {('PASS' if c1_exists and c1_readable and c2_three and c3_all else 'FAIL')}")
    return c1_exists and c1_readable and c2_three and c3_all

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 15.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
