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
PROMPT_ID = 50
Circle = Tuple[float, float, float]

def yellow_mask_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def black_edge_map(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.Canny(binary, 50, 150)

def hough_circles(img_like, dp, min_dist, param1, param2, min_radius, max_radius):
    circles = cv2.HoughCircles(img_like, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]

def dedupe_circles(circles, center_tol=25.0, radius_tol=25.0):
    out = []
    for cx, cy, cr in sorted(circles, key=lambda t: t[2], reverse=True):
        if any((np.hypot(cx - sx, cy - sy) < center_tol and abs(cr - sr) < radius_tol for sx, sy, sr in out)):
            continue
        out.append((cx, cy, cr))
    return out

def circle_perimeter_support(edge_map, circle, n_samples=360, band=3):
    h, w = edge_map.shape[:2]
    cx, cy, r = circle
    if r <= 1:
        return 0.0
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
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
    return float(ok) / total if total else 0.0

def circle_mask(shape_hw, circle):
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(round(circle[0])), int(round(circle[1]))), int(round(circle[2])), 255, -1)
    return m

def circles_overlap(c1, c2):
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < c1[2] + c2[2]

def color_ratio_in_mask(region, color):
    d = int(cv2.countNonZero(region))
    return float(cv2.countNonZero(cv2.bitwise_and(color, region))) / d if d else 0.0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _check_a_only(ms_abc, ym, on_thresh, off_thresh):
    """Check if A\\(Bâ\x88ªC) is yellow and everything else is not."""
    a_m, b_m, c_m = ms_abc
    a_only = a_m & ~b_m & ~c_m
    r_target = color_ratio_in_mask(a_only, ym)
    if r_target < on_thresh:
        return False
    other_regions = [b_m & ~a_m & ~c_m, c_m & ~a_m & ~b_m, a_m & b_m & ~c_m, a_m & c_m & ~b_m, b_m & c_m & ~a_m, a_m & b_m & c_m]
    for r in other_regions:
        if cv2.countNonZero(r) > 0 and color_ratio_in_mask(r, ym) > off_thresh:
            return False
    return True

def verify_a_only_yellow(image_path: str, yellow_on_thresh=0.2, yellow_off_thresh=0.05, support_thresh=0.43, spill_ratio=0.05) -> bool:
    if not os.path.exists(image_path):
        _print_fail('C1_file_exists: False')
        return False
    img = cv2.imread(image_path)
    if img is None:
        _print_fail('C1_file_exists: True', 'C1_image_readable: False')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('C1_file_exists: True')
    print('C1_image_readable: True')
    edges = black_edge_map(img)
    min_dim = min(h, w)
    all_c = []
    for dp, md, p1, p2, rmin, rmax in [(1.2, 0.1 * min_dim, 80, 40, int(0.1 * min_dim), int(0.35 * min_dim)), (1.2, 0.1 * min_dim, 80, 35, int(0.3 * min_dim), int(0.8 * min_dim))]:
        all_c.extend(hough_circles(edges, dp, md, p1, p2, rmin, rmax))
    all_c = dedupe_circles(all_c, 30.0, 30.0)
    circles = [c for c in all_c if circle_perimeter_support(edges, c) >= support_thresh]
    c2 = len(circles) == 3
    print(f'C2_three_circles: {c2} (found {len(circles)})')
    if not c2:
        _print_fail(f'C2_three_circles: False (found {len(circles)})')
        return False
    c3 = all((circles_overlap(circles[i], circles[j]) for i in range(3) for j in range(i + 1, 3)))
    print(f'C3_all_overlap: {c3}')
    hits = find_labels_in_circles(image_path, circles, target_letters=['A', 'B', 'C'])
    label_map = {}
    used_circles = set()
    label_ok = True
    for letter in ['A', 'B', 'C']:
        letter_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == letter]
        if not letter_hits:
            print(f'C4_label_{letter}_ok: False (not detected)')
            label_ok = False
            continue
        letter_hits.sort(key=lambda h: h[3], reverse=True)
        _, lx, ly, conf = letter_hits[0]
        inside = [i for i, c in enumerate(circles) if point_in_circle(lx, ly, c)]
        if inside and inside[0] not in used_circles:
            used_circles.add(inside[0])
            label_map[letter] = inside[0]
            print(f'C4_label_{letter}_ok: True (conf={conf:.3f})')
        else:
            print(f'C4_label_{letter}_ok: False (not inside unique circle)')
            label_ok = False
    if not label_ok or len(label_map) < 3:
        print('C4_labels_all_ok: False')
        print('Result: FAIL')
        return False
    print('C4_labels_all_ok: True')
    idx_a, idx_b, idx_c = (label_map['A'], label_map['B'], label_map['C'])
    ym = yellow_mask_hsv(img)
    ms = [circle_mask((h, w), c) for c in circles]
    ok = _check_a_only((ms[idx_a], ms[idx_b], ms[idx_c]), ym, yellow_on_thresh, yellow_off_thresh)
    a_only = ms[idx_a] & ~ms[idx_b] & ~ms[idx_c]
    print(f'C5_A_only_yellow: {color_ratio_in_mask(a_only, ym):.3f}')
    c56 = ok
    print(f'C5_C6_all_correct: {c56}')
    union = ms[0] | ms[1] | ms[2]
    outside = ~(union > 0)
    yellow_outside = int(np.sum(outside & (ym > 0)))
    total_yellow = int(np.sum(ym > 0))
    spill = yellow_outside / total_yellow if total_yellow > 0 else 0.0
    c7 = spill <= spill_ratio
    print(f'C7_yellow_spill: {spill:.4f}')
    print(f'C7_spill_ok: {c7}')
    passed = c2 and c3 and c56 and c7
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def _print_fail(*lines):
    for l in lines:
        print(l)
    print('Result: FAIL')

def evaluate(image_path: str):
    passed = verify_a_only_yellow(image_path)
    return {'passed': bool(passed), 'criteria': {'script_passed': bool(passed)}, 'meta': {'judge': 'set_50'}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 50.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
