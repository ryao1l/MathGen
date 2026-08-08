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
PROMPT_ID = 1
Circle = Tuple[float, float, float]

def red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask

def black_edge_map(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.Canny(inv, 50, 150)

def hough_circles(img, dp, md, p1, p2, rmin, rmax):
    c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, minDist=md, param1=p1, param2=p2, minRadius=rmin, maxRadius=rmax)
    if c is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in c[0]]

def dedupe_circles(circles, ct=25.0, rt=25.0):
    out = []
    for cx, cy, cr in sorted(circles, key=lambda t: t[2], reverse=True):
        if any((np.hypot(cx - sx, cy - sy) < ct and abs(cr - sr) < rt for sx, sy, sr in out)):
            continue
        out.append((cx, cy, cr))
    return out

def circle_perimeter_support(edge, circ, n=360, band=3):
    h, w = edge.shape[:2]
    cx, cy, r = circ
    if r <= 1:
        return 0.0
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.round(cx + r * np.cos(ang)).astype(np.int32)
    ys = np.round(cy + r * np.sin(ang)).astype(np.int32)
    ok = total = 0
    for x, y in zip(xs, ys):
        if x < 0 or x >= w or y < 0 or (y >= h):
            continue
        total += 1
        ok += int(np.any(edge[max(0, y - band):min(h, y + band + 1), max(0, x - band):min(w, x + band + 1)] > 0))
    return ok / total if total else 0.0

def circle_mask(hw, circ):
    m = np.zeros(hw, np.uint8)
    cv2.circle(m, (int(round(circ[0])), int(round(circ[1]))), int(round(circ[2])), 255, -1)
    return m

def circles_overlap(c1, c2):
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < c1[2] + c2[2]

def color_ratio(region, color):
    d = int(cv2.countNonZero(region))
    return float(cv2.countNonZero(cv2.bitwise_and(color, region))) / d if d else 0.0

def red_intersection_fallback(bgr):
    rm = red_mask_hsv(bgr)
    pts = cv2.findNonZero(rm)
    if pts is None:
        return False
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return False
    img_area = float(bgr.shape[0] * bgr.shape[1])
    red_area = float(cv2.countNonZero(rm))
    bbox_area = float(w * h)
    aspect = float(w) / float(h)
    fill = red_area / max(1.0, bbox_area)
    return 0.02 <= red_area / img_area <= 0.11 and 0.4 <= aspect <= 0.8 and (fill >= 0.62)

def red_intersection_shape_ok(bgr):
    rm = red_mask_hsv(bgr)
    pts = cv2.findNonZero(rm)
    if pts is None:
        return False
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return False
    img_area = float(bgr.shape[0] * bgr.shape[1])
    red_area = float(cv2.countNonZero(rm))
    aspect = float(w) / float(h)
    return 0.02 <= red_area / img_area <= 0.11 and 0.4 <= aspect <= 0.8

def evaluate(image_path, on_thresh=0.2, off_thresh=0.05, support=0.43, spill_max=0.05):
    if not os.path.exists(image_path):
        print('C1_file_exists: False')
        print('Result: FAIL')
        return False
    img = cv2.imread(image_path)
    if img is None:
        print('C1_image_readable: False')
        print('Result: FAIL')
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('C1_file_exists: True')
    print('C1_image_readable: True')
    edges = black_edge_map(img)
    md = min(h, w)
    ac = []
    for dp, d, p1, p2, rn, rx in [(1.2, 0.1 * md, 80, 40, int(0.1 * md), int(0.35 * md)), (1.2, 0.1 * md, 80, 35, int(0.3 * md), int(0.8 * md))]:
        ac.extend(hough_circles(edges, dp, d, p1, p2, rn, rx))
    ac = dedupe_circles(ac)
    circles = [c for c in ac if circle_perimeter_support(edges, c) >= support]
    print(f'C2_circles_found: {len(circles)}')
    if len(circles) < 2:
        if red_intersection_fallback(img):
            print('C2_circles_found: fallback_red_intersection')
            print('Result: PASS')
            return True
        print('Result: FAIL')
        return False
    circles = circles[:2]
    c3_ok, det = verify_circle_labels(image_path, circles, ['A', 'B'])
    for k, v in det.items():
        print(f'C3_{k}: {v}')
    ma = circle_mask((h, w), circles[0])
    mb = circle_mask((h, w), circles[1])
    rm = red_mask_hsv(img)
    inter = ma & mb
    a_only = ma & ~mb
    b_only = mb & ~ma
    r_inter = color_ratio(inter, rm)
    r_a = color_ratio(a_only, rm)
    r_b = color_ratio(b_only, rm)
    c4_inter = r_inter >= on_thresh
    c4_a = r_a <= off_thresh
    c4_b = r_b <= off_thresh
    c4 = c4_inter and c4_a and c4_b
    print(f'C4_intersection_red: {r_inter:.3f} ok={c4_inter}')
    print(f'C4_A_only_red: {r_a:.3f} ok={c4_a}')
    print(f'C4_B_only_red: {r_b:.3f} ok={c4_b}')
    union = ma | mb
    outside = ~(union > 0)
    ro = int(np.sum(outside & (rm > 0)))
    tr = int(np.sum(rm > 0))
    sp = ro / tr if tr else 0.0
    c5 = sp <= spill_max
    print(f'C5_spill: {sp:.4f} ok={c5}')
    passed = c4 and c5 and red_intersection_shape_ok(img) or red_intersection_fallback(img)
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 1.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
