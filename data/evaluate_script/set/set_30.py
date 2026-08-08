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
PROMPT_ID = 30
Circle = Tuple[float, float, float]

def magenta_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of magenta/pink pixels.

    Magenta in HSV (OpenCV convention, H 0-180):
      Hue â\x89?140-180 (wraps into red)
      Saturation â\x89?40
      Value â\x89?50
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([140, 40, 50])
    upper = np.array([180, 255, 255])
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

def verify_c_intersect_ab_excl_triple(image_path: str, mag_on_thresh: float=0.2, mag_off_thresh: float=0.05, support_thresh: float=0.43, spill_ratio: float=0.05) -> bool:
    """
    Evaluate: Three circles A, B, C.
    Shade (C â\x88?A) â\x88?(C â\x88?B), excluding (A â\x88?B â\x88?C). Colour = Magenta.

    Criteria:
      C1: File exists and image is readable.
      C2: Exactly three circles detected.
      C3: Circles overlap pairwise.
      C4: Labels A, B, C each appear once inside a distinct circle.
      C5: Region colour verification:
          - (Câ\x88©A) minus (Aâ\x88©Bâ\x88©C): magenta  â\x9c?
          - (Câ\x88©B) minus (Aâ\x88©Bâ\x88©C): magenta  â\x9c?
          - A only:            white   â\x9c?
          - B only:            white   â\x9c?
          - C only:            white   â\x9c?
          - (Aâ\x88©B) minus C:         white   â\x9c?
          - (Aâ\x88©Bâ\x88©C):           white   â\x9c?
      C6: Magenta does not spill outside intended regions.
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
    mm = magenta_mask_hsv(img)
    edges = black_edge_map(img)
    print('C1_file_exists: True')
    print('C1_image_readable: True')
    min_dim = min(h, w)
    passes = [(1.2, 0.1 * min_dim, 80, 40, int(0.1 * min_dim), int(0.35 * min_dim)), (1.2, 0.1 * min_dim, 80, 35, int(0.3 * min_dim), int(0.8 * min_dim))]
    all_circles: List[Circle] = []
    for dp, md, p1, p2, rmin, rmax in passes:
        all_circles.extend(hough_circles(edges, dp, md, p1, p2, rmin, rmax))
    all_circles = dedupe_circles(all_circles, 30.0, 30.0)
    validated = [c for c in all_circles if circle_perimeter_support(edges, c) >= support_thresh]
    c2_ok = len(validated) == 3
    print(f'C2_circles_found: {len(validated)}')
    print(f'C2_exactly_three: {c2_ok}')
    if not c2_ok:
        print('Result: FAIL')
        return False
    ov01 = circles_overlap(validated[0], validated[1])
    ov02 = circles_overlap(validated[0], validated[2])
    ov12 = circles_overlap(validated[1], validated[2])
    c3_ok = ov01 and ov02 and ov12
    print(f'C3_overlap_0_1: {ov01}')
    print(f'C3_overlap_0_2: {ov02}')
    print(f'C3_overlap_1_2: {ov12}')
    print(f'C3_all_overlapping: {c3_ok}')
    if not c3_ok:
        print('Result: FAIL')
        return False
    c4_ok, label_details = verify_circle_labels(image_path, validated, ['A', 'B', 'C'])
    for k, v in label_details.items():
        print(f'C4_{k}: {v}')
    if not c4_ok:
        print('Result: FAIL')
        return False
    ma = circle_mask((h, w), validated[0])
    mb = circle_mask((h, w), validated[1])
    mc = circle_mask((h, w), validated[2])
    ca_only = cv2.bitwise_and(mc, cv2.bitwise_and(ma, cv2.bitwise_not(mb)))
    cb_only = cv2.bitwise_and(mc, cv2.bitwise_and(mb, cv2.bitwise_not(ma)))
    a_only = cv2.bitwise_and(ma, cv2.bitwise_not(cv2.bitwise_or(mb, mc)))
    b_only = cv2.bitwise_and(mb, cv2.bitwise_not(cv2.bitwise_or(ma, mc)))
    c_only = cv2.bitwise_and(mc, cv2.bitwise_not(cv2.bitwise_or(ma, mb)))
    ab_not_c = cv2.bitwise_and(cv2.bitwise_and(ma, mb), cv2.bitwise_not(mc))
    triple = cv2.bitwise_and(ma, cv2.bitwise_and(mb, mc))
    regions = [('C_inter_A_excl_triple', ca_only, True), ('C_inter_B_excl_triple', cb_only, True), ('A_only', a_only, False), ('B_only', b_only, False), ('C_only', c_only, False), ('A_inter_B_not_C', ab_not_c, False), ('A_inter_B_inter_C', triple, False)]
    c5_ok = True
    for name, region, should_be_mag in regions:
        if cv2.countNonZero(region) == 0:
            print(f'C5_{name}: SKIP (empty region)')
            continue
        ratio = color_ratio_in_mask(region, mm)
        if should_be_mag:
            ok = ratio >= mag_on_thresh
        else:
            ok = ratio <= mag_off_thresh
        c5_ok = c5_ok and ok
        print(f"C5_{name}: ratio={ratio:.4f} {('OK' if ok else 'FAIL')}")
    print(f'C5_all_regions_correct: {c5_ok}')
    intended = cv2.bitwise_or(ca_only, cb_only)
    mag_total = int(cv2.countNonZero(mm))
    if mag_total > 0:
        mag_outside = int(cv2.countNonZero(cv2.bitwise_and(mm, cv2.bitwise_not(intended))))
        spill = float(mag_outside) / float(mag_total)
    else:
        spill = 0.0
    c6_ok = spill <= spill_ratio
    print(f'C6_magenta_spill_ratio: {spill:.4f}')
    print(f'C6_spill_ok: {c6_ok}')
    success = c2_ok and c3_ok and c4_ok and c5_ok and c6_ok
    print(f"Result: {('PASS' if success else 'FAIL')}")
    return success

def _print_fail(*lines):
    for line in lines:
        print(line)
    print('Result: FAIL')

def evaluate(image_path: str):
    passed = verify_c_intersect_ab_excl_triple(image_path)
    return {'passed': bool(passed), 'criteria': {'script_passed': bool(passed)}, 'meta': {'judge': 'set_30'}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 30.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
