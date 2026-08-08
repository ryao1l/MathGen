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
PROMPT_ID = 34
Circle = Tuple[float, float, float]

def red_mask_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def blue_mask_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def black_edge_map(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.Canny(binary, 50, 150)

def detect_rectangle(bgr, min_area_ratio=0.2):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    h, w = bgr.shape[:2]
    if cv2.contourArea(largest) < min_area_ratio * h * w:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
    if len(approx) < 4 or len(approx) > 6:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

def hough_circles(img_like, dp, min_dist, param1, param2, min_radius, max_radius) -> List[Circle]:
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

def _check_assignment(ms_abc, rect, rm, bm, red_on, blue_on, off):
    """Given (A_mask, B_mask, C_mask), check colours. Returns (ok, lines)."""
    a_m, b_m, c_m = ms_abc
    ab_no_c = a_m & b_m & ~c_m
    complement = (rect > 0).astype(np.uint8) * 255
    complement = complement & ~a_m & ~b_m & ~c_m
    r_ab = color_ratio_in_mask(ab_no_c, rm)
    r_comp = color_ratio_in_mask(complement, bm)
    if r_ab < red_on or r_comp < blue_on:
        return (False, [])
    others = [('A_only', a_m & ~b_m & ~c_m), ('B_only', b_m & ~a_m & ~c_m), ('C_only', c_m & ~a_m & ~b_m), ('AC_no_B', a_m & c_m & ~b_m), ('BC_no_A', b_m & c_m & ~a_m), ('ABC', a_m & b_m & c_m)]
    lines = [f'  AB_no_C_red: {r_ab:.3f}', f'  complement_blue: {r_comp:.3f}']
    for name, region in others:
        if cv2.countNonZero(region) == 0:
            continue
        rr = color_ratio_in_mask(region, rm)
        rb = color_ratio_in_mask(region, bm)
        if rr > off or rb > off:
            return (False, [])
        lines.append(f'  {name}: red={rr:.3f} blue={rb:.3f}')
    return (True, lines)

def verify_dual_color_rect(image_path: str, red_on_thresh=0.2, blue_on_thresh=0.2, off_thresh=0.05, support_thresh=0.43) -> bool:
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
    rect = detect_rectangle(img)
    c2 = rect is not None
    print(f'C2_rectangle_detected: {c2}')
    if not c2:
        _print_fail('C2_rectangle_detected: False')
        return False
    edges = black_edge_map(img)
    min_dim = min(h, w)
    all_c: List[Circle] = []
    for dp, md, p1, p2, rmin, rmax in [(1.2, 0.1 * min_dim, 80, 40, int(0.1 * min_dim), int(0.35 * min_dim)), (1.2, 0.1 * min_dim, 80, 35, int(0.3 * min_dim), int(0.8 * min_dim))]:
        all_c.extend(hough_circles(edges, dp, md, p1, p2, rmin, rmax))
    all_c = dedupe_circles(all_c, 30.0, 30.0)
    circles = [c for c in all_c if circle_perimeter_support(edges, c) >= support_thresh]
    c3 = len(circles) == 3
    print(f'C3_three_circles: {c3} (found {len(circles)})')
    if not c3:
        _print_fail(f'C3_three_circles: False (found {len(circles)})')
        return False
    c4 = all((circles_overlap(circles[i], circles[j]) for i in range(3) for j in range(i + 1, 3)))
    print(f'C4_all_overlap: {c4}')
    hits = find_labels_in_circles(image_path, circles, target_letters=['A', 'B', 'C'])
    label_map = {}
    used_circles = set()
    label_ok = True
    for letter in ['A', 'B', 'C']:
        letter_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == letter]
        if not letter_hits:
            print(f'C5_label_{letter}_ok: False (not detected)')
            label_ok = False
            continue
        letter_hits.sort(key=lambda h: h[3], reverse=True)
        _, lx, ly, conf = letter_hits[0]
        inside = [i for i, c in enumerate(circles) if point_in_circle(lx, ly, c)]
        if inside and inside[0] not in used_circles:
            used_circles.add(inside[0])
            label_map[letter] = inside[0]
            print(f'C5_label_{letter}_ok: True (conf={conf:.3f})')
        else:
            print(f'C5_label_{letter}_ok: False (not inside unique circle)')
            label_ok = False
    if not label_ok or len(label_map) < 3:
        print('C5_labels_all_ok: False')
        print('Result: FAIL')
        return False
    print('C5_labels_all_ok: True')
    idx_a, idx_b, idx_c = (label_map['A'], label_map['B'], label_map['C'])
    rm, bm = (red_mask_hsv(img), blue_mask_hsv(img))
    ms = [circle_mask((h, w), c) for c in circles]
    ok, lines = _check_assignment((ms[idx_a], ms[idx_b], ms[idx_c]), rect, rm, bm, red_on_thresh, blue_on_thresh, off_thresh)
    for l in lines:
        print(l)
    c67 = ok
    print(f'C6_C7_all_correct: {c67}')
    passed = c2 and c3 and c4 and c67
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def _print_fail(*lines):
    for l in lines:
        print(l)
    print('Result: FAIL')

def evaluate(image_path: str):
    passed = verify_dual_color_rect(image_path)
    return {'passed': bool(passed), 'criteria': {'script_passed': bool(passed)}, 'meta': {'judge': 'set_34'}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 34.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
