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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
try:
    import pytesseract
except Exception:
    pytesseract = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
import function_common
from function_common import *
globals().update({name: getattr(function_common, name) for name in dir(function_common) if name.startswith('_')})
PROMPT_ID = 29
DEBUG = False
DEBUG_DIR = 'debug_function_16'
BLACK_HSV_LO = (0, 0, 0)
BLACK_HSV_HI = (179, 160, 140)
BLACK_OPEN_K = (3, 3)
KV_LEN = 55
KH_LEN = 55
MIN_AXIS_COL_FRAC = 0.02
MIN_AXIS_ROW_FRAC = 0.02
CANNY1, CANNY2 = (50, 150)
HOUGH_THRESH = 110
MAX_GAP = 30
AXIS_THICKNESS = 10
XAXIS_GAP = 6
YAXIS_GAP = 8
HSV_S_MIN = 15
HSV_V_MIN = 25
ASYM_SEARCH_HALF_W = 30
ASYM_MIN_RUN_LEN = 25
ASYM_COL_SMOOTH_K = 41
ASYM_PEAK_MIN_SEP = 80
BLUE_H_LO = 85
BLUE_H_HI = 155
BLUE_S_MIN = 20
BLUE_V_MIN = 30
CROSS_STRIP_HALF_W = 4
CROSS_DENSITY_MAX = 0.15
RUN_MIN_LEN = 8
RUN_ALLOW_GAP = 1
RUN_SEARCH_START_BAND = 8
RUN_SEARCH_MAX_BAND = 220
RUN_SEARCH_STEP = 6

def _dsave(name: str, img: np.ndarray):
    if not DEBUG or img is None or img.size == 0:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, name), img)

def red_asymptote_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, HSV_S_MIN, HSV_V_MIN), (15, 255, 255))
    m2 = cv2.inRange(hsv, (160, HSV_S_MIN, HSV_V_MIN), (179, 255, 255))
    m3 = cv2.inRange(hsv, (135, HSV_S_MIN, HSV_V_MIN), (170, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=1)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    mask = cv2.bitwise_or(mask, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kv, iterations=1))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
    return mask

def extract_blue_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (BLUE_H_LO, BLUE_S_MIN, BLUE_V_MIN), (BLUE_H_HI, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)), iterations=1)
    return mask

def blackish_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blk = cv2.inRange(hsv, BLACK_HSV_LO, BLACK_HSV_HI)
    blk = cv2.morphologyEx(blk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, BLACK_OPEN_K), iterations=1)
    return blk

def erase_mask_to_white(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    out[mask > 0] = (255, 255, 255)
    return out

def detect_axes_projection(img_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    h, w = img_bgr.shape[:2]
    blk = blackish_mask(img_bgr)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, KV_LEN))
    v = cv2.morphologyEx(blk, cv2.MORPH_OPEN, kv, iterations=1)
    col = (v > 0).sum(axis=0).astype(np.float32) / float(h)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (KH_LEN, 1))
    hh = cv2.morphologyEx(blk, cv2.MORPH_OPEN, kh, iterations=1)
    row = (hh > 0).sum(axis=1).astype(np.float32) / float(w)
    if float(np.max(col)) < MIN_AXIS_COL_FRAC or float(np.max(row)) < MIN_AXIS_ROW_FRAC:
        return (None, None)
    return (float(np.argmax(row)), float(np.argmax(col)))

def detect_axes_hough(img_bgr: np.ndarray) -> Tuple[float, float]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESH, minLineLength=int(0.45 * w), maxLineGap=MAX_GAP)
    x_axis_y, y_axis_x = (h / 2.0, w / 2.0)
    best_h: Tuple[Optional[float], float] = (None, -1.0)
    best_v: Tuple[Optional[float], float] = (None, -1.0)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = (x2 - x1, y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < 0.4 * w:
                continue
            if abs(dy) <= 4 and length > best_h[1]:
                best_h = ((y1 + y2) / 2.0, length)
            if abs(dx) <= 4 and length > best_v[1]:
                best_v = ((x1 + x2) / 2.0, length)
    if best_h[0] is not None:
        x_axis_y = best_h[0]
    if best_v[0] is not None:
        y_axis_x = best_v[0]
    return (x_axis_y, y_axis_x)

def detect_axes(img_bgr: np.ndarray) -> Tuple[float, float]:
    rmask = red_asymptote_mask(img_bgr)
    img_clean = erase_mask_to_white(img_bgr, rmask)
    result = detect_axes_projection(img_clean)
    if result[0] is not None and result[1] is not None:
        return (result[0], result[1])
    return detect_axes_hough(img_clean)

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode='same')

def find_vertical_peaks(mask: np.ndarray, n: int=2, min_sep: int=ASYM_PEAK_MIN_SEP) -> List[int]:
    h, w = mask.shape[:2]
    col = (mask > 0).sum(axis=0).astype(np.float32) / float(h)
    col_s = smooth_1d(col, ASYM_COL_SMOOTH_K)
    peaks: List[int] = []
    work = col_s.copy()
    for _ in range(n):
        p = int(np.argmax(work))
        if work[p] <= 1e-06:
            break
        peaks.append(p)
        left = max(0, p - min_sep)
        right = min(w, p + min_sep + 1)
        work[left:right] = 0
    peaks.sort()
    return peaks

def vertical_run_exists(mask: np.ndarray, px_center: int, half_w: int=ASYM_SEARCH_HALF_W, min_run: int=ASYM_MIN_RUN_LEN) -> Tuple[bool, int]:
    h, w = mask.shape[:2]
    x0 = max(0, px_center - half_w)
    x1 = min(w, px_center + half_w + 1)
    strip = mask[:, x0:x1]
    if strip.size == 0:
        return (False, px_center)
    best_run = 0
    best_x = px_center
    for xi in range(strip.shape[1]):
        col = (strip[:, xi] > 0).astype(np.uint8)
        run = max_run = 0
        for v in col:
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > best_run:
            best_run = max_run
            best_x = x0 + xi
    return (best_run >= min_run, best_x)

def remove_axes(mask: np.ndarray, x_axis_y: float, y_axis_x: float) -> np.ndarray:
    h, w = mask.shape[:2]
    out = mask.copy()
    ya, xa = (int(round(x_axis_y)), int(round(y_axis_x)))
    cv2.line(out, (0, ya), (w - 1, ya), 0, thickness=AXIS_THICKNESS)
    cv2.line(out, (xa, 0), (xa, h - 1), 0, thickness=AXIS_THICKNESS)
    out[max(0, ya - XAXIS_GAP):min(h, ya + XAXIS_GAP + 1), :] = 0
    out[:, max(0, xa - YAXIS_GAP):min(w, xa + YAXIS_GAP + 1)] = 0
    return out

def crossing_density(mask: np.ndarray, px_asym: int) -> float:
    h, w = mask.shape[:2]
    x0 = max(0, px_asym - CROSS_STRIP_HALF_W)
    x1 = min(w, px_asym + CROSS_STRIP_HALF_W + 1)
    strip = mask[:, x0:x1]
    if strip.size == 0:
        return 0.0
    return float(np.sum(strip > 0) / strip.size)

def sign_from_pixel_y(ypix: float, x_axis_y: float) -> int:
    return +1 if ypix < x_axis_y else -1

def longest_run_1d(bits: np.ndarray, allow_gap: int=RUN_ALLOW_GAP) -> Tuple[int, Optional[Tuple[int, int]]]:
    n = len(bits)
    best_len = 0
    best_seg: Optional[Tuple[int, int]] = None
    i = 0
    while i < n:
        while i < n and bits[i] == 0:
            i += 1
        if i >= n:
            break
        start = i
        gap = 0
        j = i
        while j < n:
            if bits[j] == 1:
                j += 1
                continue
            gap += 1
            if gap > allow_gap:
                break
            j += 1
        end = j - 1
        while end >= start and bits[end] == 0:
            end -= 1
        seg_len = max(0, end - start + 1)
        if seg_len > best_len:
            best_len = seg_len
            best_seg = (start, end)
        i = j + 1
    return (best_len, best_seg)

def find_run_near_x(curve_mask: np.ndarray, px_center: int, side: str, x_axis_y: float, want_sign: int) -> Dict[str, Any]:
    h, w = curve_mask.shape[:2]
    px_center = int(np.clip(px_center, 0, w - 1))
    ya = int(round(x_axis_y))
    if want_sign > 0:
        y0, y1 = (0, max(0, ya - 1))
    else:
        y0, y1 = (min(h - 1, ya + 1), h - 1)
    if y1 <= y0:
        return {'status': 'insufficient'}
    for band in range(RUN_SEARCH_START_BAND, RUN_SEARCH_MAX_BAND + 1, RUN_SEARCH_STEP):
        if side == 'left':
            x0 = max(0, px_center - band)
            x1 = max(0, px_center - 1)
        else:
            x0 = min(w - 1, px_center + 1)
            x1 = min(w - 1, px_center + band)
        if x1 < x0:
            continue
        roi = curve_mask[y0:y1 + 1, x0:x1 + 1]
        cols = range(roi.shape[1] - 1, -1, -1) if side == 'left' else range(roi.shape[1])
        for cx in cols:
            col = (roi[:, cx] > 0).astype(np.uint8)
            blen, seg = longest_run_1d(col)
            if blen >= RUN_MIN_LEN and seg is not None:
                y_mid = y0 + 0.5 * (seg[0] + seg[1])
                got = sign_from_pixel_y(y_mid, x_axis_y)
                return {'status': 'ok', 'got_sign': int(got), 'run_len': int(blen)}
    return {'status': 'insufficient'}

def evaluate(image_path: str) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        return {'passed': False, 'criteria': {'image_readable': False}, 'meta': {}}
    _dsave('00_input.png', img)
    h, w = img.shape[:2]
    x_axis_y, y_axis_x = detect_axes(img)
    rmask = red_asymptote_mask(img)
    _dsave('10_red_mask.png', rmask)
    peaks = find_vertical_peaks(rmask, n=2, min_sep=ASYM_PEAK_MIN_SEP)
    asym_present = False
    px_left: Optional[int] = None
    px_right: Optional[int] = None
    if len(peaks) >= 2:
        ok_l, px_left = vertical_run_exists(rmask, peaks[0])
        ok_r, px_right = vertical_run_exists(rmask, peaks[1])
        asym_present = ok_l and ok_r
    asym_position_ok = False
    if px_left is not None and px_right is not None and (w > 0):
        left_offset = (y_axis_x - px_left) / w
        right_offset = (px_right - y_axis_x) / w
        sep_ratio = (px_right - px_left) / w
        axis_ratio = y_axis_x / w
        asym_position_ok = 0.45 <= axis_ratio <= 0.55 and 0.13 <= left_offset <= 0.18 and (0.13 <= right_offset <= 0.18) and (0.27 <= sep_ratio <= 0.33)
    blue = extract_blue_mask(img)
    _dsave('20_blue_mask.png', blue)
    curve = remove_axes(blue, x_axis_y, y_axis_x)
    _dsave('21_curve.png', curve)
    cross_left = crossing_density(curve, px_left) < CROSS_DENSITY_MAX if px_left is not None else False
    cross_right = crossing_density(curve, px_right) < CROSS_DENSITY_MAX if px_right is not None else False
    far_right_pos = False
    if px_right is not None:
        res = find_run_near_x(curve, px_right, 'right', x_axis_y, want_sign=+1)
        far_right_pos = res['status'] == 'ok' and res['got_sign'] == +1
    far_left_neg = False
    if px_left is not None:
        res = find_run_near_x(curve, px_left, 'left', x_axis_y, want_sign=-1)
        far_left_neg = res['status'] == 'ok' and res['got_sign'] == -1
    criteria = {'image_readable': True, 'asymptotes_present': bool(asym_present), 'asymptotes_at_x_pm3': bool(asym_position_ok), 'curve_not_crossing_left': bool(cross_left), 'curve_not_crossing_right': bool(cross_right), 'far_right_positive': bool(far_right_pos), 'far_left_negative': bool(far_left_neg)}
    return {'passed': all(criteria.values()), 'criteria': criteria, 'meta': {'x_axis_y': x_axis_y, 'y_axis_x': y_axis_x, 'px_left_asym': px_left, 'px_right_asym': px_right, 'asym_position_ok': asym_position_ok}}

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 29.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
