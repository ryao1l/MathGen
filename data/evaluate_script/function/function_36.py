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
PROMPT_ID = 36
DEBUG = False
DEBUG_DIR = 'function_sqrt_debug'
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
BLUE_H_LO = 85
BLUE_H_HI = 155
BLUE_S_MIN = 20
BLUE_V_MIN = 30
CROSS_STRIP_HALF_W = 4
CROSS_DENSITY_MAX = 0.12
RUN_MIN_LEN = 8
RUN_ALLOW_GAP = 1
RUN_SEARCH_START_BAND = 8
RUN_SEARCH_MAX_BAND = 260
RUN_SEARCH_STEP = 6
MIN_OCR_CONF = 10.0
OCR_SCALE = 3
OCR_ADAPT_BLOCK = 31
OCR_ADAPT_C = 9
EXPECTED_X = list(range(-10, 11))
EXPECTED_Y = list(range(-50, 51))
RANSAC_ITERS = 250
INLIER_TOL_X = 0.35
MIN_INLIERS_X = 2
Y_MODEL_CANDIDATES = 450
Y_INLIER_TOL = 0.85
Y_SLOPE_RANGE = (-2.0, -0.0005)
Y_MAX_DUP_PER_VALUE = 2
YCOL_SMOOTH_K = 41
YCOL_HALF_W = 60
ASYM_TARGET_X = 2.0
ASYM_X_TOL = 0.35
SAMPLE_XS = [-4, 0, 4, 6]
SAMPLE_ERR_TOL = 1.0
SAMPLE_PASS_AT_LEAST = 2
FAR_BAND_HALF_W = 8
FAR_MIN_PIXELS = 30
FAR_Y_TRIM = 0.05

def ensure_debug_dir():
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def dsave(name: str, img: np.ndarray):
    if not DEBUG:
        return
    if img is None or img.size == 0:
        print(f'Debug skip (empty): {name}')
        return
    ensure_debug_dir()
    path = os.path.join(DEBUG_DIR, name)
    cv2.imwrite(path, img)
    print(f'Debug saved: {path}')

def maybe_set_tesseract_cmd():
    pass

def f(x: float) -> float:
    return float(np.sqrt(x * x + 1.0) / (x - 2.0))

def red_asymptote_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m_red1 = cv2.inRange(hsv, (0, HSV_S_MIN, HSV_V_MIN), (15, 255, 255))
    m_red2 = cv2.inRange(hsv, (160, HSV_S_MIN, HSV_V_MIN), (179, 255, 255))
    m_mag = cv2.inRange(hsv, (135, HSV_S_MIN, HSV_V_MIN), (170, 255, 255))
    mask = cv2.bitwise_or(m_red1, m_red2)
    mask = cv2.bitwise_or(mask, m_mag)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    mask_v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kv, iterations=1)
    mask = cv2.bitwise_or(mask, mask_v)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
    return mask

def extract_blue_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = (BLUE_H_LO, BLUE_S_MIN, BLUE_V_MIN)
    upper = (BLUE_H_HI, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    return mask

def blackish_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blk = cv2.inRange(hsv, BLACK_HSV_LO, BLACK_HSV_HI)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, BLACK_OPEN_K)
    blk = cv2.morphologyEx(blk, cv2.MORPH_OPEN, k, iterations=1)
    return blk

def erase_mask_to_white(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    out[mask > 0] = (255, 255, 255)
    return out

def erase_red_blue_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    r = red_asymptote_mask(img_bgr)
    b = extract_blue_mask(img_bgr)
    m = cv2.bitwise_or(r, b)
    return erase_mask_to_white(img_bgr, m)

def detect_axes_black_projection_no_center(img_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    blk = blackish_mask(img_bgr)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, KV_LEN))
    v = cv2.morphologyEx(blk, cv2.MORPH_OPEN, kv, iterations=1)
    col = (v > 0).sum(axis=0).astype(np.float32) / float(h)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (KH_LEN, 1))
    hh = cv2.morphologyEx(blk, cv2.MORPH_OPEN, kh, iterations=1)
    row = (hh > 0).sum(axis=1).astype(np.float32) / float(w)
    col_max = float(np.max(col)) if col.size else 0.0
    row_max = float(np.max(row)) if row.size else 0.0
    ok = col_max >= MIN_AXIS_COL_FRAC and row_max >= MIN_AXIS_ROW_FRAC
    dbg = {'black': blk, 'v': v, 'hh': hh, 'col_max': col_max, 'row_max': row_max}
    if not ok:
        return (None, None, dbg)
    y_axis_x = float(np.argmax(col))
    x_axis_y = float(np.argmax(row))
    return (x_axis_y, y_axis_x, dbg)

def detect_axes_hough(img_bgr: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESH, minLineLength=int(0.45 * w), maxLineGap=MAX_GAP)
    x_axis_y = h / 2.0
    y_axis_x = w / 2.0
    best_h = (None, -1.0)
    best_v = (None, -1.0)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = (x2 - x1, y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < 0.4 * w:
                continue
            if abs(dy) <= 4:
                ymid = (y1 + y2) / 2.0
                if length > best_h[1]:
                    best_h = (ymid, length)
            if abs(dx) <= 4:
                xmid = (x1 + x2) / 2.0
                if length > best_v[1]:
                    best_v = (xmid, length)
    if best_h[0] is not None:
        x_axis_y = float(best_h[0])
    if best_v[0] is not None:
        y_axis_x = float(best_v[0])
    return (x_axis_y, y_axis_x, {'edges': edges})

def detect_axes(img_bgr: np.ndarray) -> Tuple[float, float]:
    rmask = red_asymptote_mask(img_bgr)
    img_no_red = erase_mask_to_white(img_bgr, rmask)
    xpy, ypx, dbg = detect_axes_black_projection_no_center(img_no_red)
    if DEBUG:
        dsave('00_input.png', img_bgr)
        dsave('01_black_mask.png', dbg.get('black', np.zeros((1, 1), np.uint8)))
        dsave('02_v_strokes.png', dbg.get('v', np.zeros((1, 1), np.uint8)))
        dsave('03_h_strokes.png', dbg.get('hh', np.zeros((1, 1), np.uint8)))
        print(f"Projection strength: col_max={dbg.get('col_max', 0):.4f}, row_max={dbg.get('row_max', 0):.4f}")
    if xpy is not None and ypx is not None:
        return (float(xpy), float(ypx))
    xpy2, ypx2, hdbg = detect_axes_hough(img_no_red)
    if DEBUG:
        dsave('04_edges_hough.png', hdbg['edges'])
    return (float(xpy2), float(ypx2))

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode='same')

def find_one_vertical_peak(mask: np.ndarray) -> Optional[int]:
    h, w = mask.shape[:2]
    col = (mask > 0).sum(axis=0).astype(np.float32) / float(h)
    col_s = smooth_1d(col, ASYM_COL_SMOOTH_K)
    p = int(np.argmax(col_s))
    if col_s[p] <= 1e-06:
        return None
    return p

def vertical_run_exists_in_strip(mask: np.ndarray, px_center: int, half_w: int, min_run_len: int) -> Tuple[bool, Dict[str, Any]]:
    h, w = mask.shape[:2]
    x0 = max(0, px_center - half_w)
    x1 = min(w, px_center + half_w + 1)
    strip = mask[:, x0:x1]
    if strip.size == 0:
        return (False, {'best_run': 0, 'best_x': None, 'strip': (x0, x1)})
    best_run = 0
    best_x = None
    for xi in range(strip.shape[1]):
        col = (strip[:, xi] > 0).astype(np.uint8)
        run = 0
        max_run = 0
        for v in col:
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > best_run:
            best_run = max_run
            best_x = x0 + xi
    ok = best_run >= min_run_len
    return (ok, {'best_run': int(best_run), 'best_x': best_x, 'strip': (x0, x1)})

def remove_axes(mask: np.ndarray, x_axis_y: float, y_axis_x: float) -> np.ndarray:
    h, w = mask.shape[:2]
    out = mask.copy()
    ya = int(round(x_axis_y))
    xa = int(round(y_axis_x))
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

def longest_run_1d(bits: np.ndarray, allow_gap: int=1) -> Tuple[int, Optional[Tuple[int, int]]]:
    n = len(bits)
    best_len = 0
    best_seg = None
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

def find_run_near_x(curve_mask: np.ndarray, px_center: int, side: str, x_axis_y: float, want_sign: int, min_run_len: int, max_band: int) -> Dict[str, Any]:
    h, w = curve_mask.shape[:2]
    px_center = int(np.clip(px_center, 0, w - 1))
    ya = int(round(x_axis_y))
    if want_sign > 0:
        y0, y1 = (0, max(0, ya - 1))
    else:
        y0, y1 = (min(h - 1, ya + 1), h - 1)
    if y1 <= y0:
        return {'status': 'insufficient'}
    for band in range(RUN_SEARCH_START_BAND, max_band + 1, RUN_SEARCH_STEP):
        if side == 'left':
            x0 = max(0, px_center - band)
            x1 = max(0, px_center - 1)
        else:
            x0 = min(w - 1, px_center + 1)
            x1 = min(w - 1, px_center + band)
        if x1 < x0:
            continue
        roi = curve_mask[y0:y1 + 1, x0:x1 + 1]
        col_indices = range(roi.shape[1] - 1, -1, -1) if side == 'left' else range(0, roi.shape[1])
        for cx in col_indices:
            col = (roi[:, cx] > 0).astype(np.uint8)
            best_len, seg = longest_run_1d(col, allow_gap=RUN_ALLOW_GAP)
            if best_len >= min_run_len and seg is not None:
                x_pick = x0 + cx
                y_mid = y0 + 0.5 * (seg[0] + seg[1])
                got = sign_from_pixel_y(y_mid, x_axis_y)
                return {'status': 'ok', 'x_pick': int(x_pick), 'run_len': int(best_len), 'got_sign': int(got), 'band': int(band)}
    return {'status': 'insufficient'}

def band_median_sign(curve_mask: np.ndarray, px_center: int, x_axis_y: float, half_w: int=FAR_BAND_HALF_W) -> Dict[str, Any]:
    h, w = curve_mask.shape[:2]
    px_center = int(np.clip(px_center, 0, w - 1))
    x0 = max(0, px_center - half_w)
    x1 = min(w - 1, px_center + half_w)
    roi = curve_mask[:, x0:x1 + 1]
    ys, xs = np.where(roi > 0)
    n = len(ys)
    if n < FAR_MIN_PIXELS:
        return {'status': 'insufficient', 'n': n}
    ys_sorted = np.sort(ys.astype(np.float32))
    lo = int(np.floor(FAR_Y_TRIM * n))
    hi = int(np.ceil((1.0 - FAR_Y_TRIM) * n))
    ys_trim = ys_sorted[lo:hi] if hi > lo else ys_sorted
    y_med = float(np.median(ys_trim))
    got = sign_from_pixel_y(y_med, x_axis_y)
    return {'status': 'ok', 'got_sign': int(got), 'y_med': y_med, 'n': n, 'band': (x0, x1)}

def curve_y_pixel_at_x(mask: np.ndarray, px_x: int, half_window: int=6) -> Optional[float]:
    h, w = mask.shape[:2]
    px_x = int(np.clip(px_x, 0, w - 1))
    x0 = max(0, px_x - half_window)
    x1 = min(w, px_x + half_window + 1)
    strip = mask[:, x0:x1]
    ys = np.where(strip > 0)[0]
    if len(ys) == 0:
        return None
    return float(np.median(ys))

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, OCR_ADAPT_BLOCK, OCR_ADAPT_C)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    return bin_img

def ocr_numbers_robust(img_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]]=None, debug_prefix: str='', debug_print: bool=False) -> List[Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    rx0, ry0, rx1, ry1 = (0, 0, W, H)
    if roi is not None:
        rx0, ry0, rx1, ry1 = roi
        rx0, ry0 = (max(0, rx0), max(0, ry0))
        rx1, ry1 = (min(W, rx1), min(H, ry1))
        if rx1 <= rx0 or ry1 <= ry0:
            if debug_print:
                print(f'  [{debug_prefix}] ROI empty -> []')
            return []
        img_bgr = img_bgr[ry0:ry1, rx0:rx1]
    up = cv2.resize(img_bgr, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    ocr_bin = preprocess_for_ocr(up)
    if DEBUG and debug_prefix:
        dsave(f'{debug_prefix}_ocr_up.png', up)
        dsave(f'{debug_prefix}_ocr_bin.png', ocr_bin)
    configs = ['--oem 3 --psm 6  -c tessedit_char_whitelist=-−—–0123456789', '--oem 3 --psm 11 -c tessedit_char_whitelist=-−—–0123456789']
    raw_tokens = []
    for cfg in configs:
        data = pytesseract.image_to_data(ocr_bin, config=cfg, output_type=pytesseract.Output.DICT)
        n = len(data['text'])
        for i in range(n):
            txt = (data['text'][i] or '').strip()
            conf = float(data['conf'][i]) if data['conf'][i] != '-1' else -1.0
            if txt == '':
                continue
            txt_norm = txt.replace('−', '-').replace('—', '-').replace('–', '-').strip('., ')
            lx, ty, bw, bh = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cx, cy = (lx + bw / 2.0, ty + bh / 2.0)
            center = (cx / OCR_SCALE + rx0, cy / OCR_SCALE + ry0)
            bbox = (lx / OCR_SCALE + rx0, ty / OCR_SCALE + ry0, bw / OCR_SCALE, bh / OCR_SCALE)
            raw_tokens.append({'text': txt, 'text_norm': txt_norm, 'conf': conf, 'center': center, 'bbox': bbox, 'cfg': cfg})
    if debug_print:
        print(f'\n  [{debug_prefix}] OCR RAW TOKENS (before filtering, total={len(raw_tokens)})')
        for t in sorted(raw_tokens, key=lambda z: z['conf'], reverse=True)[:120]:
            cx, cy = t['center']
            x, y, bw, bh = t['bbox']
            print(f"    raw='{t['text']}' norm='{t['text_norm']}' conf={t['conf']:.1f} center=({cx:.1f},{cy:.1f}) bbox=({x:.1f},{y:.1f},{bw:.1f},{bh:.1f})")
    tokens = [t for t in raw_tokens if t['conf'] >= MIN_OCR_CONF and t['text_norm'] != '']
    tokens2 = [{'text': t['text_norm'], 'conf': t['conf'], 'center': t['center'], 'bbox': t['bbox']} for t in tokens]
    merged = []
    used = set()
    order = sorted(range(len(tokens2)), key=lambda i: tokens2[i]['center'][0])
    for i in order:
        if i in used:
            continue
        t = tokens2[i]
        if t['text'] == '-':
            best_j = None
            best_dx = 1000000000.0
            for j in order:
                if j == i or j in used:
                    continue
                u = tokens2[j]
                if abs(u['center'][1] - t['center'][1]) > 14:
                    continue
                dx = u['center'][0] - t['center'][0]
                if 0 < dx < best_dx:
                    best_dx = dx
                    best_j = j
            if best_j is not None:
                u = tokens2[best_j]
                if re.fullmatch('\\d+', u['text']) and best_dx < 35:
                    merged.append({'value': int('-' + u['text']), 'conf': min(t['conf'], u['conf']), 'center': ((t['center'][0] + u['center'][0]) / 2.0, (t['center'][1] + u['center'][1]) / 2.0), 'bbox': t['bbox'], 'raw': '-' + u['text']})
                    used.add(i)
                    used.add(best_j)
                    continue
        if re.fullmatch('-?\\d+', t['text']):
            merged.append({'value': int(t['text']), 'conf': t['conf'], 'center': t['center'], 'bbox': t['bbox'], 'raw': t['text']})
            used.add(i)
    out = []
    for d in sorted(merged, key=lambda z: (-z['conf'], z['center'][1])):
        keep = True
        for e in out:
            if d['value'] == e['value'] and abs(d['center'][0] - e['center'][0]) < 8 and (abs(d['center'][1] - e['center'][1]) < 8):
                keep = False
                break
        if keep:
            out.append(d)
    if debug_print:
        print(f'\n  [{debug_prefix}] OCR MERGED INTEGERS (after merge+dedup, total={len(out)})')
        for d in sorted(out, key=lambda z: (z['value'], -z['conf'])):
            cx, cy = d['center']
            print(f"    value={d['value']:+d} conf={d['conf']:.1f} center=({cx:.1f},{cy:.1f}) raw='{d.get('raw', '')}'")
    return out

def compute_rois(h: int, w: int, x_axis_y: float, y_axis_x: float) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    band = int(0.18 * h)
    y0 = max(0, int(x_axis_y - band))
    y1 = min(h, int(x_axis_y + band))
    if y1 <= y0:
        y0, y1 = (0, h)
    x_roi = (0, y0, w, y1)
    half = int(0.28 * w)
    x0 = max(0, int(y_axis_x - half))
    x1 = min(w, int(y_axis_x + half))
    if x1 - x0 < 60:
        x0 = max(0, int(y_axis_x) - 30)
        x1 = min(w, int(y_axis_x) + 30)
        if x1 <= x0:
            x0, x1 = (0, min(w, 120))
    y_roi = (x0, 0, x1, h)
    return (x_roi, y_roi)

def ransac_fit_v_from_p(points: List[Tuple[float, float]], iters: int, tol: float, min_inliers: int) -> Tuple[Optional[float], Optional[float]]:
    if len(points) < 2:
        return (None, None)
    P = np.array([p for p, _ in points], dtype=np.float64)
    V = np.array([v for _, v in points], dtype=np.float64)
    rng = np.random.default_rng(0)
    best_inliers: List[int] = []
    best_model = (None, None)
    for _ in range(iters):
        i, j = rng.choice(len(points), size=2, replace=False)
        if abs(P[j] - P[i]) < 1e-06:
            continue
        a = (V[j] - V[i]) / (P[j] - P[i])
        b = V[i] - a * P[i]
        pred = a * P + b
        err = np.abs(pred - V)
        inliers = np.where(err <= tol)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (float(a), float(b))
    if len(best_inliers) < min_inliers:
        return (None, None)
    inP = P[best_inliers]
    inV = V[best_inliers]
    A = np.vstack([inP, np.ones_like(inP)]).T
    a, b = np.linalg.lstsq(A, inV, rcond=None)[0]
    return (float(a), float(b))

def build_x_mapping(x_dets: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    per_val: Dict[int, List[Dict[str, Any]]] = {}
    for d in x_dets:
        v = int(d['value'])
        if v not in EXPECTED_X:
            continue
        per_val.setdefault(v, []).append(d)
    chosen = []
    for v, lst in per_val.items():
        chosen.extend(sorted(lst, key=lambda z: z['conf'], reverse=True)[:1])
    pts = [(float(d['center'][0]), float(d['value'])) for d in chosen]
    return ransac_fit_v_from_p(pts, RANSAC_ITERS, INLIER_TOL_X, MIN_INLIERS_X)

def find_tick_column_peak_x(y_dets: List[Dict[str, Any]], w: int) -> Optional[float]:
    if len(y_dets) < 2:
        return None
    xs = np.array([d['center'][0] for d in y_dets], dtype=np.float32)
    bins = np.zeros(w, dtype=np.float32)
    xi = np.clip(xs.astype(np.int32), 0, w - 1)
    for x in xi:
        bins[x] += 1.0
    bins_s = smooth_1d(bins, YCOL_SMOOTH_K)
    peak = float(np.argmax(bins_s))
    if bins_s[int(peak)] < 1.5:
        return None
    return peak

def filter_y_candidates_by_column(y_dets: List[Dict[str, Any]], x_axis_y: float, h: int, w: int) -> List[Dict[str, Any]]:
    base = []
    for d in y_dets:
        v = int(d['value'])
        if v not in EXPECTED_Y:
            continue
        _, cy = d['center']
        if abs(cy - x_axis_y) < 0.12 * h:
            continue
        base.append(d)
    if len(base) < 2:
        return base
    peak_x = find_tick_column_peak_x(base, w)
    if peak_x is None:
        keep = base
    else:
        keep = [d for d in base if abs(d['center'][0] - peak_x) <= YCOL_HALF_W]
    per_val: Dict[int, List[Dict[str, Any]]] = {}
    for d in keep:
        per_val.setdefault(int(d['value']), []).append(d)
    reduced = []
    for v, lst in per_val.items():
        reduced.extend(sorted(lst, key=lambda z: z['conf'], reverse=True)[:Y_MAX_DUP_PER_VALUE])
    return reduced

def build_y_mapping_multiverify(y_dets: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if len(y_dets) < 2:
        return (None, None)
    pts = [(float(d['center'][1]), float(d['value']), float(d['conf'])) for d in y_dets]
    rng = np.random.default_rng(1)

    def score_model(ay: float, by: float) -> float:
        if not Y_SLOPE_RANGE[0] < ay < Y_SLOPE_RANGE[1]:
            return -1000000000.0
        inliers = 0
        for py, yv, _ in pts:
            if abs(ay * py + by - yv) <= Y_INLIER_TOL:
                inliers += 1
        return float(inliers)
    best_score = -1e+18
    best_model = (None, None)
    for _ in range(Y_MODEL_CANDIDATES):
        i, j = rng.choice(len(pts), size=2, replace=False)
        py1, y1, _ = pts[i]
        py2, y2, _ = pts[j]
        if abs(py2 - py1) < 1e-06:
            continue
        ay = (y2 - y1) / (py2 - py1)
        by = y1 - ay * py1
        s = score_model(ay, by)
        if s > best_score:
            best_score = s
            best_model = (float(ay), float(by))
    if best_model[0] is None:
        return (None, None)
    ay, by = best_model
    if not Y_SLOPE_RANGE[0] < ay < Y_SLOPE_RANGE[1]:
        return (None, None)
    return (float(ay), float(by))

def _mode_bin_1d(vals: List[float], bin_size: float) -> Optional[float]:
    if not vals:
        return None
    bins: Dict[int, List[float]] = {}
    for v in vals:
        k = int(np.floor(v / bin_size))
        bins.setdefault(k, []).append(v)
    best_k = None
    best_lst = None
    for k, lst in bins.items():
        if best_lst is None or len(lst) > len(best_lst):
            best_k, best_lst = (k, lst)
        elif len(lst) == len(best_lst):
            if max(lst) - min(lst) < max(best_lst) - min(best_lst):
                best_k, best_lst = (k, lst)
    return float(np.mean(best_lst)) if best_lst else None

def filter_x_candidates_by_row(x_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int, *, y_band_frac: float=0.12, x_margin_frac: float=0.06, row_bin_px: float=8.0, row_tight_px: float=10.0) -> List[Dict[str, Any]]:
    if not x_dets:
        return []
    y_tol = float(y_band_frac) * float(h)
    x_margin = float(x_margin_frac) * float(w)
    cand = []
    for d in x_dets:
        cx, cy = d['center']
        if abs(cy - x_axis_y) > y_tol:
            continue
        if abs(cx - y_axis_x) < x_margin:
            continue
        cand.append(d)
    if len(cand) < 2:
        return cand
    y_mode = _mode_bin_1d([d['center'][1] for d in cand], bin_size=row_bin_px)
    if y_mode is None:
        return cand
    row = [d for d in cand if abs(d['center'][1] - y_mode) <= row_tight_px]
    if not row:
        return cand
    per: Dict[int, List[Dict[str, Any]]] = {}
    for d in row:
        v = int(d['value'])
        per.setdefault(v, []).append(d)
    out = []
    for v, lst in per.items():
        lst2 = sorted(lst, key=lambda z: (-z['conf'], abs(z['center'][1] - y_mode)))
        out.append(lst2[0])
    return out

def filter_y_candidates_by_col(y_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int, *, x_band_frac: float=0.12, y_margin_frac: float=0.1, col_bin_px: float=8.0, col_tight_px: float=10.0) -> List[Dict[str, Any]]:
    if not y_dets:
        return []
    x_tol = float(x_band_frac) * float(w)
    y_margin = float(y_margin_frac) * float(h)
    cand = []
    for d in y_dets:
        cx, cy = d['center']
        if abs(cx - y_axis_x) > x_tol:
            continue
        if abs(cy - x_axis_y) < y_margin:
            continue
        cand.append(d)
    if len(cand) < 2:
        return cand
    x_mode = _mode_bin_1d([d['center'][0] for d in cand], bin_size=col_bin_px)
    if x_mode is None:
        return cand
    col = [d for d in cand if abs(d['center'][0] - x_mode) <= col_tight_px]
    if not col:
        return cand
    per: Dict[int, List[Dict[str, Any]]] = {}
    for d in col:
        v = int(d['value'])
        per.setdefault(v, []).append(d)
    out = []
    for v, lst in per.items():
        lst2 = sorted(lst, key=lambda z: (-z['conf'], abs(z['center'][0] - x_mode)))
        out.append(lst2[0])
    return out

def build_x_mapping_xy(x_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    x_use = filter_x_candidates_by_row(x_dets, x_axis_y=x_axis_y, y_axis_x=y_axis_x, h=h, w=w)
    ax, bx = build_x_mapping(x_use)
    return (ax, bx, x_use)

def build_y_mapping_xy(y_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    y_use = filter_y_candidates_by_col(y_dets, x_axis_y=x_axis_y, y_axis_x=y_axis_x, h=h, w=w)
    ay, by = build_y_mapping_multiverify(y_use)
    return (ay, by, y_use)

def evaluate_plot(image_path: str):
    if DEBUG:
        return _evaluate_plot_impl(image_path)
    with contextlib.redirect_stdout(io.StringIO()):
        return _evaluate_plot_impl(image_path)

def _evaluate_plot_impl(image_path: str):
    maybe_set_tesseract_cmd()
    img = cv2.imread(image_path)
    if img is None:
        return {'passed': False, 'reason': 'Image not found'}
    h, w = img.shape[:2]
    print(f'\nEvaluator: sqrt(x^2+1)/(x-2) asymptote x=2 (full + OCR raw dump): {os.path.basename(image_path)}')
    print('-' * 90)
    x_axis_y, y_axis_x = detect_axes(img)
    print(f'Axes detected: x-axis y≈{x_axis_y:.1f}, y-axis x≈{y_axis_x:.1f}')
    overlay_axes = img.copy()
    cv2.line(overlay_axes, (0, int(round(x_axis_y))), (w - 1, int(round(x_axis_y))), (0, 255, 255), 2)
    cv2.line(overlay_axes, (int(round(y_axis_x)), 0), (int(round(y_axis_x)), h - 1), (0, 255, 255), 2)
    dsave('05_axes_overlay.png', overlay_axes)
    rmask = red_asymptote_mask(img)
    dsave('10_red_mask.png', rmask)
    px_asym = find_one_vertical_peak(rmask)
    if px_asym is None:
        return {'id': 36, 'passed': False, 'criteria': {'asymptote_present': False}, 'meta': {'x_axis_y': x_axis_y, 'y_axis_x': y_axis_x}}
    print(f'Red vertical candidate column detected: px={px_asym} (expected x≈2)')
    print('\n[A] Asymptote presence: vertical red segment near the candidate (CORE)')
    okA, infoA = vertical_run_exists_in_strip(rmask, px_asym, ASYM_SEARCH_HALF_W, ASYM_MIN_RUN_LEN)
    px_asym_use = infoA['best_x'] if infoA['best_x'] is not None else px_asym
    print(f"  x=2: ok={okA} | best_run={infoA['best_run']} at x={px_asym_use}  {('OK' if okA else 'FAIL')}")
    core_ok = bool(okA)
    overlay_all = img.copy()
    cv2.line(overlay_all, (int(px_asym_use), 0), (int(px_asym_use), h - 1), (0, 0, 255), 2)
    cv2.line(overlay_all, (0, int(round(x_axis_y))), (w - 1, int(round(x_axis_y))), (0, 255, 255), 2)
    cv2.line(overlay_all, (int(round(y_axis_x)), 0), (int(round(y_axis_x)), h - 1), (0, 255, 255), 2)
    dsave('11_axes_plus_asym_overlay.png', overlay_all)
    blue = extract_blue_mask(img)
    dsave('20_blue_mask.png', blue)
    curve_mask = remove_axes(blue, x_axis_y, y_axis_x)
    dsave('21_blue_no_axes.png', curve_mask)
    print('\n[B] Crossing check: blue curve should not cross the asymptote column (CORE)')
    cross = crossing_density(curve_mask, int(px_asym_use))
    okB = cross < CROSS_DENSITY_MAX
    print(f"  x=2: px={int(px_asym_use)}, curve_density_near_asym={cross:.4f}  {('OK' if okB else 'FAIL')}")
    core_ok = core_ok and okB
    print('\n[C] Right branch should be positive (above x-axis) (CORE, run-based)')
    right_pos = find_run_near_x(curve_mask, int(px_asym_use), 'right', x_axis_y, want_sign=+1, min_run_len=RUN_MIN_LEN, max_band=RUN_SEARCH_MAX_BAND)
    if right_pos['status'] != 'ok':
        print('  right: insufficient  FAIL')
        okC = False
    else:
        okC = right_pos['got_sign'] == +1
        print(f"  right: got={right_pos['got_sign']:+d} run_len={right_pos['run_len']} band={right_pos['band']}  {('OK' if okC else 'FAIL')}")
    core_ok = core_ok and okC
    print('\n[D] Left branch should be negative (below x-axis) (CORE, run-based)')
    left_neg = find_run_near_x(curve_mask, int(px_asym_use), 'left', x_axis_y, want_sign=-1, min_run_len=RUN_MIN_LEN, max_band=RUN_SEARCH_MAX_BAND)
    if left_neg['status'] != 'ok':
        print('  left: insufficient  FAIL')
        okD = False
    else:
        okD = left_neg['got_sign'] == -1
        print(f"  left: got={left_neg['got_sign']:+d} run_len={left_neg['run_len']} band={left_neg['band']}  {('OK' if okD else 'FAIL')}")
    core_ok = core_ok and okD
    asym_offset_units = float(px_asym_use - y_axis_x) / max(float(w) / 12.0, 1e-06)
    okA2 = 1.4 <= asym_offset_units <= 1.9
    core_ok = core_ok and okA2
    criteria = {'asymptote_present': bool(okA), 'asymptote_at_x_2': bool(okA2), 'curve_not_crossing_asymptote': bool(okB), 'right_branch_positive': bool(okC), 'left_branch_negative': bool(okD)}
    return {'id': 36, 'passed': bool(all(criteria.values())), 'criteria': criteria, 'meta': {'x_axis_y': float(x_axis_y), 'y_axis_x': float(y_axis_x), 'asymptote_x': int(px_asym_use), 'asym_offset_units': asym_offset_units, 'red_run': infoA, 'crossing_density': cross, 'right_branch': right_pos, 'left_branch': left_neg}}
    print('\n[OCR] Axis tick OCR (for A2 and optional F2)')
    x_roi, y_roi = compute_rois(h, w, x_axis_y, y_axis_x)
    img_ocr = erase_red_blue_for_ocr(img)
    dsave('29_ocr_cleaned_full.png', img_ocr)
    x0, y0, x1, y1 = x_roi
    dsave('30_x_roi.png', img_ocr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((1, 1), np.uint8))
    x0, y0, x1, y1 = y_roi
    dsave('31_y_roi.png', img_ocr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((1, 1), np.uint8))
    x_dets = ocr_numbers_robust(img_ocr, roi=x_roi, debug_prefix='32_x', debug_print=True)
    y_dets = ocr_numbers_robust(img_ocr, roi=y_roi, debug_prefix='33_y', debug_print=True)
    print(f"\n  x_dets values={sorted({d['value'] for d in x_dets})}")
    print(f"  y_dets values={sorted({d['value'] for d in y_dets})}")
    ax, bx, x_use = build_x_mapping_xy(x_dets, x_axis_y=x_axis_y, y_axis_x=y_axis_x, h=h, w=w)
    print('  [X-OCR] points used (filtered row): ' + ', '.join([f"{d['value']}@({d['center'][0]:.1f},{d['center'][1]:.1f}) c{d['conf']:.0f}" for d in sorted(x_use, key=lambda z: z['center'][0])]) if x_use else '  (none)')
    ay, by, y_use = build_y_mapping_xy(y_dets, x_axis_y=x_axis_y, y_axis_x=y_axis_x, h=h, w=w)
    print('  [Y-OCR] points used (filtered col): ' + ', '.join([f"{d['value']}@({d['center'][0]:.1f},{d['center'][1]:.1f}) c{d['conf']:.0f}" for d in sorted(y_use, key=lambda z: z['center'][1])]) if y_use else '  (none)')
    print('\n[A2] Asymptote near x=2 (CORE, OCR x mapping)')
    if ax is None:
        gotx = sorted({d['value'] for d in x_dets})
        print(f'  x-axis OCR failed: recognized {gotx} -> FAIL')
        core_ok = False
    else:
        x_asym_meas = ax * float(px_asym_use) + bx
        okA2 = abs(x_asym_meas - ASYM_TARGET_X) <= ASYM_X_TOL
        print(f"  x_asym_meas={x_asym_meas:+.3f} (target=+2.000, tol={ASYM_X_TOL})  {('OK' if okA2 else 'FAIL')}")
        core_ok = core_ok and okA2
    print('\n[E] Far-end sign check (optional, band-median)')
    if ax is None:
        print('  Skipped: x-axis OCR failed')
    else:

        def px_from_x(xv: float) -> int:
            return int(round((float(xv) - bx) / ax))
        px6 = px_from_x(6.0)
        r6 = band_median_sign(curve_mask, px6, x_axis_y)
        if r6['status'] != 'ok':
            print(f"  x=+6: insufficient  N/A (n={r6.get('n', 0)})")
        else:
            print(f"  x=+6: got_sign={r6['got_sign']:+d} (n={r6['n']}, y_med={r6['y_med']:.1f})")
        pxm6 = px_from_x(-6.0)
        rm6 = band_median_sign(curve_mask, pxm6, x_axis_y)
        if rm6['status'] != 'ok':
            print(f"  x=-6: insufficient  N/A (n={rm6.get('n', 0)})")
        else:
            print(f"  x=-6: got_sign={rm6['got_sign']:+d} (n={rm6['n']}, y_med={rm6['y_med']:.1f})")
    print('\n' + '-' * 90)
    print('Final result: PASS' if core_ok else 'Final result: FAIL')
evaluate = evaluate_plot

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 36.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
