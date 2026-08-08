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
PROMPT_ID = 3
Circle = Tuple[float, float, float]

def green_mask_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
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

def color_ratio(region, color):
    d = int(cv2.countNonZero(region))
    return float(cv2.countNonZero(cv2.bitwise_and(color, region))) / d if d else 0.0

def green_union_fallback(bgr):
    gm = green_mask_hsv(bgr)
    pts = cv2.findNonZero(gm)
    if pts is None:
        return False
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return False
    img_area = float(bgr.shape[0] * bgr.shape[1])
    green_area = float(cv2.countNonZero(gm))
    bbox_area = float(w * h)
    aspect = float(w) / float(h)
    fill = green_area / max(1.0, bbox_area)
    return green_area / img_area >= 0.25 and 1.1 <= aspect <= 2.05 and (fill >= 0.6)

def green_union_shape_ok(bgr):
    gm = green_mask_hsv(bgr)
    pts = cv2.findNonZero(gm)
    if pts is None:
        return False
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return False
    img_area = float(bgr.shape[0] * bgr.shape[1])
    green_area = float(cv2.countNonZero(gm))
    aspect = float(w) / float(h)
    return green_area / img_area >= 0.1 and aspect >= 1.05

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
    max_dim = max(img.shape[:2])
    if max_dim > 1024:
        scale = 1024.0 / float(max_dim)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('C1_file_exists: True')
    print('C1_image_readable: True')
    edges = black_edge_map(img)
    md = min(h, w)
    ac = []
    for dp, d, p1, p2, rn, rx in [(1.2, 0.1 * md, 80, 40, int(0.1 * md), int(0.35 * md)), (1.2, 0.1 * md, 80, 35, int(0.3 * md), int(0.8 * md))]:
        ac.extend(hough_circles(edges, dp, d, p1, p2, rn, rx))
    if len(ac) < 2:
        gray_blur = cv2.medianBlur(gray, 7)
        gray_edges = cv2.Canny(gray_blur, 30, 100)
        for dp, d, p1, p2, rn, rx in [(1.2, 0.1 * md, 80, 40, int(0.1 * md), int(0.35 * md)), (1.2, 0.1 * md, 80, 30, int(0.15 * md), int(0.45 * md)), (1.2, 0.1 * md, 50, 25, int(0.2 * md), int(0.5 * md))]:
            ac.extend(hough_circles(gray_edges, dp, d, p1, p2, rn, rx))
    if len(ac) < 2:
        gray_blur = cv2.medianBlur(gray, 7)
        for dp, p2 in [(1.2, 30), (1.2, 25), (1.5, 25)]:
            ac.extend(hough_circles(gray_blur, dp, 0.1 * md, 100, p2, int(0.15 * md), int(0.45 * md)))
    ac = dedupe_circles(ac)
    supported = [c for c in ac if circle_perimeter_support(edges, c) >= support]
    if len(supported) >= 2:
        circles = supported[:2]
    else:
        circles = ac[:2] if len(ac) >= 2 else ac
    print(f'C2_circles_found: {len(circles)}')
    if len(circles) < 2:
        if green_union_fallback(img):
            print('C2_circles_found: fallback_green_union')
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
    gm = green_mask_hsv(img)
    a_only = ma & ~mb
    inter = ma & mb
    b_only = mb & ~ma
    r_ao = color_ratio(a_only, gm)
    r_i = color_ratio(inter, gm)
    r_bo = color_ratio(b_only, gm)
    c4_ao = r_ao >= on_thresh
    c4_i = r_i >= on_thresh
    c4_bo = r_bo >= on_thresh
    c4 = c4_ao and c4_i and c4_bo
    print(f'C4_A_only_green: {r_ao:.3f} ok={c4_ao}')
    print(f'C4_intersection_green: {r_i:.3f} ok={c4_i}')
    print(f'C4_B_only_green: {r_bo:.3f} ok={c4_bo}')
    union = ma | mb
    outside = ~(union > 0)
    go = int(np.sum(outside & (gm > 0)))
    tg = int(np.sum(gm > 0))
    sp = go / tg if tg else 0.0
    c5 = sp <= spill_max
    print(f'C5_spill: {sp:.4f} ok={c5}')
    passed = c4 and c5 and green_union_shape_ok(img) or green_union_fallback(img)
    print(f"Result: {('PASS' if passed else 'FAIL')}")
    return passed

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen set case 3.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
