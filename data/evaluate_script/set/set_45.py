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
PROMPT_ID = 45
Circle = Tuple[float, float, float]

def cyan_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of cyan pixels.

    Cyan in HSV (OpenCV convention, H 0-180):
      Hue â\x89?80-100
      Saturation â\x89?40
      Value â\x89?50
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 40, 50])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def black_edge_map(bgr: np.ndarray) -> np.ndarray:
    """Build an edge map emphasizing black strokes on a light background."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.Canny(binary_inv, 50, 150)

def hough_circles(img_like: np.ndarray, dp: float, min_dist: float, param1: float, param2: float, min_radius: int, max_radius: int) -> List[Circle]:
    """Run HoughCircles and return all detected circles."""
    circles = cv2.HoughCircles(img_like, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]

def dedupe_circles(circles: List[Circle], center_tol: float=25.0, radius_tol: float=25.0) -> List[Circle]:
    """Greedy de-duplication of near-identical circles."""
    out: List[Circle] = []
    for cx, cy, cr in sorted(circles, key=lambda t: t[2], reverse=True):
        if any((np.hypot(cx - sx, cy - sy) < center_tol and abs(cr - sr) < radius_tol for sx, sy, sr in out)):
            continue
        out.append((cx, cy, cr))
    return out

def circle_perimeter_support(edge_map: np.ndarray, circle: Circle, n_samples: int=360, band: int=3) -> float:
    """Score how well a circle is supported by edges near its circumference."""
    h, w = edge_map.shape[:2]
    cx, cy, r = circle
    if r <= 1:
        return 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    xs = np.round(cx + r * np.cos(angles)).astype(np.int32)
    ys = np.round(cy + r * np.sin(angles)).astype(np.int32)
    ok, total = (0, 0)
    for x, y in zip(xs, ys):
        if x < 0 or x >= w or y < 0 or (y >= h):
            continue
        x0, x1 = (max(0, x - band), min(w, x + band + 1))
        y0, y1 = (max(0, y - band), min(h, y + band + 1))
        total += 1
        ok += int(np.any(edge_map[y0:y1, x0:x1] > 0))
    return float(ok) / float(total) if total > 0 else 0.0

def circle_mask(shape_hw: Tuple[int, int], circle: Circle) -> np.ndarray:
    """Binary uint8 mask (0/255) for a filled circle."""
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(round(circle[0])), int(round(circle[1]))), int(round(circle[2])), 255, -1)
    return m

def circles_overlap(c1: Circle, c2: Circle) -> bool:
    """Check whether two circles overlap."""
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    return dist < c1[2] + c2[2]

def color_ratio_in_mask(region: np.ndarray, color: np.ndarray) -> float:
    """Fraction of colored pixels within a uint8 255 region mask."""
    denom = int(cv2.countNonZero(region))
    if denom == 0:
        return 0.0
    return float(cv2.countNonZero(cv2.bitwise_and(color, region))) / float(denom)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_asymmetric_intersection(image_path: str, cyan_on_thresh: float=0.2, cyan_off_thresh: float=0.05, support_thresh: float=0.43, size_ratio_thresh: float=1.2, spill_ratio: float=0.05) -> bool:
    """
    Evaluate: Asymmetric Venn diagram. Circle A is significantly larger
    than Circle B. Shade only (A â\x88?B) in Cyan.

    Criteria:
      C1: File exists and image is readable.
      C2: Exactly two circles detected.
      C3: Circles overlap.
      C4: Labels A, B each appear once inside a distinct circle.
      C5: Circle A is significantly larger than Circle B (radius_A / radius_B >= threshold).
      C6: Region colour verification:
          - A â\x88?B:  cyan  â\x9c?
          - A only: white â\x9c?
          - B only: white â\x9c?
      C7: Cyan does not spill outside the intersection.
    """
    if not os.path.exists(image_path):
        _print_fail('C1_file_exists: False')
        return False
    img = cv2.imread(image_path)
    if img is None:
        _print_fail('C1_file_exists: True', 'C1_image_readable: False')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cm = cyan_mask_hsv(img)
    edges = black_edge_map(img)
    print('C1_file_exists: True')
    print('C1_image_readable: True')
    min_dim = min(h, w)
    passes = [(1.2, 0.1 * min_dim, 80, 40, int(0.05 * min_dim), int(0.35 * min_dim)), (1.2, 0.1 * min_dim, 80, 35, int(0.3 * min_dim), int(0.8 * min_dim))]
    all_circles: List[Circle] = []
    for dp, md, p1, p2, rmin, rmax in passes:
        all_circles.extend(hough_circles(edges, dp, md, p1, p2, rmin, rmax))
    all_circles = dedupe_circles(all_circles, 30.0, 30.0)
    validated = [c for c in all_circles if circle_perimeter_support(edges, c) >= support_thresh]
    c2_ok = len(validated) == 2
    print(f'C2_circles_found: {len(validated)}')
    print(f'C2_exactly_two: {c2_ok}')
    if not c2_ok:
        print('Result: FAIL')
        return False
    c3_ok = circles_overlap(validated[0], validated[1])
    print(f'C3_circles_overlap: {c3_ok}')
    if not c3_ok:
        print('Result: FAIL')
        return False
    hits = find_labels_in_circles(image_path, validated, target_letters=['A', 'B'])
    label_map = {}
    used_circles = set()
    c4_ok = True
    for letter in ['A', 'B']:
        letter_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == letter]
        if not letter_hits:
            print(f'C4_label_{letter}_ok: False (not detected)')
            c4_ok = False
            continue
        letter_hits.sort(key=lambda h: h[3], reverse=True)
        _, lx, ly, conf = letter_hits[0]
        inside = [i for i, c in enumerate(validated) if point_in_circle(lx, ly, c)]
        if inside and inside[0] not in used_circles:
            used_circles.add(inside[0])
            label_map[letter] = inside[0]
            print(f'C4_label_{letter}_ok: True (conf={conf:.3f})')
        else:
            print(f'C4_label_{letter}_ok: False (not inside unique circle)')
            c4_ok = False
    if not c4_ok or 'A' not in label_map or 'B' not in label_map:
        print('C4_labels_all_ok: False')
        print('Result: FAIL')
        return False
    print('C4_labels_all_ok: True')
    circle_a = validated[label_map['A']]
    circle_b = validated[label_map['B']]
    r_a, r_b = (circle_a[2], circle_b[2])
    ratio = r_a / r_b if r_b > 0 else 0.0
    c5_ok = ratio >= size_ratio_thresh
    ma = circle_mask((h, w), circle_a)
    mb = circle_mask((h, w), circle_b)
    intersection = cv2.bitwise_and(ma, mb)
    a_only = cv2.bitwise_and(ma, cv2.bitwise_not(mb))
    b_only = cv2.bitwise_and(mb, cv2.bitwise_not(ma))
    regions = [('A_inter_B', intersection, True), ('A_only', a_only, False), ('B_only', b_only, False)]
    c6_ok = True
    for name, region, should_be_cyan in regions:
        if cv2.countNonZero(region) == 0:
            continue
        ratio_val = color_ratio_in_mask(region, cm)
        if should_be_cyan:
            ok = ratio_val >= cyan_on_thresh
        else:
            ok = ratio_val <= cyan_off_thresh
        c6_ok = c6_ok and ok
    cyan_total = int(cv2.countNonZero(cm))
    union_mask = cv2.bitwise_or(ma, mb)
    if cyan_total > 0:
        cyan_outside = int(cv2.countNonZero(cv2.bitwise_and(cm, cv2.bitwise_not(union_mask))))
        spill = float(cyan_outside) / float(cyan_total)
    else:
        spill = 0.0
    c7_ok = spill <= spill_ratio
    print(f'C5_radius_A: {r_a:.1f}')
    print(f'C5_radius_B: {r_b:.1f}')
    print(f'C5_size_ratio: {ratio:.3f}')
    print(f'C5_A_larger_than_B: {c5_ok}')
    print(f'C6_all_regions_correct: {c6_ok}')
    print(f'C7_cyan_spill_ratio: {spill:.4f}')
    print(f'C7_spill_ok: {c7_ok}')
    passed = c5_ok and c6_ok and c7_ok
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def _print_fail(*lines):
    for line in lines:
        print(line)
    print('Result: FAIL')

def evaluate(image_path: str):
    passed = verify_asymmetric_intersection(image_path)
    return {'passed': bool(passed), 'criteria': {'script_passed': bool(passed)}, 'meta': {'judge': 'set_45'}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 45.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
