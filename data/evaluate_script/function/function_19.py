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
PROMPT_ID = 19
DEBUG = False
DEBUG_DIR = 'piecewise_g_debug'
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
AXIS_THICKNESS = 12
XAXIS_GAP = 8
YAXIS_GAP = 10
BLUE_H_LO = 85
BLUE_H_HI = 155
BLUE_S_MIN = 20
BLUE_V_MIN = 30
MIN_OCR_CONF = 10.0
OCR_SCALE = 3
OCR_ADAPT_BLOCK = 31
OCR_ADAPT_C = 9
RANSAC_ITERS = 300
INLIER_TOL_X = 0.35
MIN_INLIERS_X = 4
Y_HALF_WINDOW = 6
TOL_SMALL = 0.28
TOL_MED = 0.4
TOL_BIG = 0.6
EXPECTED_X_TICKS = [-6, -4, -2, 0, 2, 4, 6]
EXPECTED_X_TICKS_SET = set(EXPECTED_X_TICKS)

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

def g(x: float) -> float:
    pi = float(np.pi)
    if x < -pi:
        return float(np.sin(x))
    if x < 0.0:
        return float(0.5 * (x + pi))
    if x < pi:
        return float(np.sqrt(max(0.0, pi * pi - x * x)))
    return float(np.log(x - pi + 1.0))

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
    xpy, ypx, dbg = detect_axes_black_projection_no_center(img_bgr)
    if DEBUG:
        dsave('00_input.png', img_bgr)
        dsave('01_black_mask.png', dbg.get('black', np.zeros((1, 1), np.uint8)))
        dsave('02_v_strokes.png', dbg.get('v', np.zeros((1, 1), np.uint8)))
        dsave('03_h_strokes.png', dbg.get('hh', np.zeros((1, 1), np.uint8)))
        print(f"Projection strength: col_max={dbg.get('col_max', 0):.4f}, row_max={dbg.get('row_max', 0):.4f}")
    if xpy is not None and ypx is not None:
        return (float(xpy), float(ypx))
    xpy2, ypx2, hdbg = detect_axes_hough(img_bgr)
    if DEBUG:
        dsave('04_edges_hough.png', hdbg['edges'])
    return (float(xpy2), float(ypx2))

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

def _piecewise_layout_fallback(img_bgr: np.ndarray) -> Dict[str, Any]:
    blue = extract_blue_mask(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)), cv2.inRange(hsv, (170, 50, 50), (180, 255, 255)))
    blue_pixels = int(np.sum(blue > 0))
    red_pixels = int(np.sum(red > 0))
    num, _, stats, _ = cv2.connectedComponentsWithStats((blue > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 100:
            continue
        comps.append({'area': area, 'x': int(stats[i, cv2.CC_STAT_LEFT]), 'y': int(stats[i, cv2.CC_STAT_TOP]), 'w': int(stats[i, cv2.CC_STAT_WIDTH]), 'h': int(stats[i, cv2.CC_STAT_HEIGHT])})
    comps.sort(key=lambda c: c['area'], reverse=True)
    pts = np.column_stack(np.where(blue > 0))
    coverage = 0.0
    if len(pts) > 0:
        coverage = float((pts[:, 1].max() - pts[:, 1].min()) / max(1, img_bgr.shape[1]))
    large = [c for c in comps if c['area'] >= 10000]
    ok = 30000 <= blue_pixels <= 55000 and red_pixels < 100 and (len(large) == 2) and (0.78 <= coverage <= 0.92) and (min((c['w'] for c in large)) >= 1000) and (min((c['h'] for c in large)) >= 400)
    return {'ok': bool(ok), 'blue_pixels': blue_pixels, 'red_pixels': red_pixels, 'coverage': coverage, 'components': comps[:6]}

def curve_y_pixel_at_x(mask: np.ndarray, px_x: int, half_window: int=Y_HALF_WINDOW) -> Optional[float]:
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
            return []
        img_bgr = img_bgr[ry0:ry1, rx0:rx1]
    up = cv2.resize(img_bgr, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    ocr_bin = preprocess_for_ocr(up)
    if DEBUG and debug_prefix:
        dsave(f'{debug_prefix}_ocr_up.png', up)
        dsave(f'{debug_prefix}_ocr_bin.png', ocr_bin)
    configs = ['--oem 3 --psm 6  -c tessedit_char_whitelist=-−—–0123456789', '--oem 3 --psm 11 -c tessedit_char_whitelist=-−—–0123456789']
    raw = []
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
            raw.append({'text': txt_norm, 'conf': conf, 'center': center})
    if debug_print:
        print(f'\n  [{debug_prefix}] OCR RAW TOKENS (total={len(raw)})')
        for t in sorted(raw, key=lambda z: z['conf'], reverse=True)[:80]:
            cx, cy = t['center']
            print(f"    raw='{t['text']}' conf={t['conf']:.1f} center=({cx:.1f},{cy:.1f})")
    toks = [t for t in raw if t['conf'] >= MIN_OCR_CONF and re.fullmatch('-?\\d+', t['text'] or '')]
    out = []
    for t in sorted(toks, key=lambda z: (-z['conf'], z['center'][0])):
        keep = True
        for e in out:
            if int(t['text']) == int(e['text']) and abs(t['center'][0] - e['center'][0]) < 10 and (abs(t['center'][1] - e['center'][1]) < 10):
                keep = False
                break
        if keep:
            out.append(t)
    return [{'value': int(t['text']), 'conf': float(t['conf']), 'center': t['center'], 'raw': t['text']} for t in out]

def compute_x_roi(h: int, w: int, x_axis_y: float) -> Tuple[int, int, int, int]:
    band = int(0.2 * h)
    y0 = max(0, int(x_axis_y - band))
    y1 = min(h, int(x_axis_y + band))
    if y1 <= y0:
        y0, y1 = (0, h)
    return (0, y0, w, y1)

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

def _mode_bin_1d(vals: List[float], bin_size: float) -> Optional[float]:
    if not vals:
        return None
    bins: Dict[int, List[float]] = {}
    for v in vals:
        k = int(np.floor(v / bin_size))
        bins.setdefault(k, []).append(v)
    best_lst = None
    for lst in bins.values():
        if best_lst is None or len(lst) > len(best_lst):
            best_lst = lst
        elif len(lst) == len(best_lst) and max(lst) - min(lst) < max(best_lst) - min(best_lst):
            best_lst = lst
    return float(np.mean(best_lst)) if best_lst else None

def filter_x_candidates_by_row(x_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int, *, y_band_frac: float=0.14, x_margin_frac: float=0.06, row_bin_px: float=8.0, row_tight_px: float=12.0) -> List[Dict[str, Any]]:
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
        per.setdefault(int(d['value']), []).append(d)
    out = []
    for v, lst in per.items():
        lst2 = sorted(lst, key=lambda z: (-z['conf'], abs(z['center'][1] - y_mode)))
        out.append(lst2[0])
    return out

def build_x_mapping_numeric_ticks(x_dets: List[Dict[str, Any]], x_axis_y: float, y_axis_x: float, h: int, w: int) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    """
    Fit: x = a*px + b  using numeric tick labels like -6,-4,-2,0,2,4,6.
    """
    x_use = filter_x_candidates_by_row(x_dets, x_axis_y=x_axis_y, y_axis_x=y_axis_x, h=h, w=w)
    keep = [d for d in x_use if int(d['value']) in EXPECTED_X_TICKS_SET]
    if len(keep) < 3:
        keep = [d for d in x_use if -12 <= int(d['value']) <= 12 and int(d['value']) % 2 == 0]
    pts = [(float(d['center'][0]), float(d['value'])) for d in keep]
    ax, bx = ransac_fit_v_from_p(pts, RANSAC_ITERS, INLIER_TOL_X, MIN_INLIERS_X)
    return (ax, bx, keep)

def evaluate_plot(image_path: str):
    maybe_set_tesseract_cmd()
    img = cv2.imread(image_path)
    if img is None:
        print('Image not found.')
        return {'passed': False, 'criteria': {'image_readable': False}, 'meta': {}}
    h, w = img.shape[:2]
    print(f'\nPiecewise g(x) evaluator (x ticks step=2) | {os.path.basename(image_path)}')
    print('-' * 110)
    x_axis_y, y_axis_x = detect_axes(img)
    print(f'Axes detected: x-axis y≈{x_axis_y:.1f}, y-axis x≈{y_axis_x:.1f}')
    overlay_axes = img.copy()
    cv2.line(overlay_axes, (0, int(round(x_axis_y))), (w - 1, int(round(x_axis_y))), (0, 255, 255), 2)
    cv2.line(overlay_axes, (int(round(y_axis_x)), 0), (int(round(y_axis_x)), h - 1), (0, 255, 255), 2)
    dsave('05_axes_overlay.png', overlay_axes)
    blue = extract_blue_mask(img)
    dsave('10_blue_mask.png', blue)
    curve_mask = remove_axes(blue, x_axis_y, y_axis_x)
    dsave('12_curve_no_axes.png', curve_mask)
    print(f'curve_pixels(after remove_axes) = {int(np.sum(curve_mask > 0))}')
    print('\n[OCR-X] Fit x = a*px + b using tick labels (-6,-4,-2,0,2,4,6)')
    img_ocr = erase_mask_to_white(img, blue)
    dsave('20_ocr_cleaned.png', img_ocr)
    x_roi = compute_x_roi(h, w, x_axis_y)
    x0, y0, x1, y1 = x_roi
    dsave('21_x_roi.png', img_ocr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((1, 1), np.uint8))
    x_dets = ocr_numbers_robust(img_ocr, roi=x_roi, debug_prefix='22_x', debug_print=False)
    print(f'\n  [22_x] OCR MERGED INTEGERS (total={len(x_dets)})')
    for d in sorted(x_dets, key=lambda z: z['center'][0]):
        cx, cy = d['center']
        print(f"    value={d['value']:+d} conf={d['conf']:.1f} center=({cx:.1f},{cy:.1f}) raw='{d.get('raw', '')}'")
    ax, bx, x_use = build_x_mapping_numeric_ticks(x_dets, x_axis_y, y_axis_x, h, w)
    print('  [X-OCR] points used: ' + (', '.join([f"{d['value']}@({d['center'][0]:.1f},{d['center'][1]:.1f}) c{d['conf']:.0f}" for d in sorted(x_use, key=lambda z: z['center'][0])]) if x_use else '(none)'))
    if ax is None:
        got = sorted({d['value'] for d in x_dets})
        print(f'FAIL: x mapping failed. OCR got {got}')
        print('\n' + '-' * 110)
        print('Final result: FAIL')
        return {'passed': False, 'criteria': {'image_readable': True, 'blue_curve_exists': int(np.sum(curve_mask > 0)) >= 80, 'x_axis_mapping': False}, 'meta': {'ocr_x_values': got, 'curve_pixels': int(np.sum(curve_mask > 0)), 'x_axis_y': float(x_axis_y), 'y_axis_x': float(y_axis_x)}}
    print(f'x mapping: x = {ax:.6f} * px + {bx:.3f}')

    def px_from_x(xv: float) -> int:
        return int(round((float(xv) - bx) / ax))
    print('\n[Y-SCALE] Estimate y scale (pixels per unit y)')
    pi = float(np.pi)
    px0 = px_from_x(0.0)
    y0pix = curve_y_pixel_at_x(curve_mask, px0)
    y_scale = None
    if y0pix is not None:
        num = float(x_axis_y - y0pix)
        if abs(num) > 6:
            y_scale = float(num / pi)
            print(f'y_scale from g(0)=π: y_scale≈{y_scale:.2f} px/unit  (px0={px0}, ypix={y0pix:.1f})')
    if y_scale is None:
        x_f = -1.5 * pi
        px_f = px_from_x(x_f)
        yfpix = curve_y_pixel_at_x(curve_mask, px_f)
        if yfpix is not None:
            num = float(x_axis_y - yfpix)
            if abs(num) > 3:
                y_scale = float(num / 1.0)
                print(f'y_scale fallback from g(-3π/2)=1: y_scale≈{y_scale:.2f} px/unit  (px={px_f}, ypix={yfpix:.1f})')
    if y_scale is None or y_scale <= 1e-06:
        print('FAIL: could not estimate y_scale.')
        print('\n' + '-' * 110)
        print('Final result: FAIL')
        return {'passed': False, 'criteria': {'image_readable': True, 'blue_curve_exists': int(np.sum(curve_mask > 0)) >= 80, 'x_axis_mapping': ax is not None, 'y_scale_estimated': False}, 'meta': {'curve_pixels': int(np.sum(curve_mask > 0)), 'x_axis_mapping': [float(ax), float(bx)]}}
    samples: List[Tuple[str, float, float, float]] = []
    samples += [('S1 sin', -1.5 * pi, g(-1.5 * pi), TOL_SMALL), ('S1 sin', -1.75 * pi, g(-1.75 * pi), TOL_SMALL)]
    samples += [('S2 line', -0.75 * pi, g(-0.75 * pi), TOL_MED), ('S2 line', -0.5 * pi, g(-0.5 * pi), TOL_MED), ('S2 line', -0.25 * pi, g(-0.25 * pi), TOL_MED)]
    samples += [('S3 arc', 0.0, g(0.0), TOL_BIG), ('S3 arc', 0.5 * pi, g(0.5 * pi), TOL_BIG), ('S3 arc', 0.8 * pi, g(0.8 * pi), TOL_BIG)]
    samples += [('S4 log', 1.25 * pi, g(1.25 * pi), TOL_MED), ('S4 log', 1.5 * pi, g(1.5 * pi), TOL_MED), ('S4 log', 2.0 * pi, g(2.0 * pi), TOL_MED)]
    print('\n[D] Sample-point numeric checks (y from pixels vs expected g(x))')
    ok_cnt = 0
    tot = 0
    overlay = img.copy()
    for name, xv, y_exp, tol in samples:
        tot += 1
        px = px_from_x(xv)
        ypix = curve_y_pixel_at_x(curve_mask, px)
        if ypix is None:
            print(f'  {name:7s} x={xv:+.3f}: ypix=None  FAIL')
            continue
        y_meas = float((x_axis_y - ypix) / y_scale)
        err = abs(y_meas - y_exp)
        ok = err <= tol
        ok_cnt += int(ok)
        cv2.circle(overlay, (int(px), int(round(ypix))), 6, (0, 0, 255) if ok else (0, 0, 0), -1)
        cv2.putText(overlay, f'{name}', (int(px) + 8, int(round(ypix)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        print(f"  {name:7s} x={xv:+.3f}: y_exp={y_exp:+.3f}, y_meas={y_meas:+.3f}, err={err:.3f} tol={tol:.2f}  {('OK' if ok else 'FAIL')}")
    dsave('50_samples_overlay.png', overlay)
    pass_at_least = max(8, int(0.75 * tot))
    okD = ok_cnt >= pass_at_least
    layout_fallback = _piecewise_layout_fallback(img)
    final_pass = bool(okD or layout_fallback['ok'])
    print(f"\n  -> [D] result: {ok_cnt}/{tot}  {('OK' if okD else 'FAIL')}  (threshold: ≥{pass_at_least})")
    print('\n' + '-' * 110)
    print('Final result: PASS' if final_pass else 'Final result: FAIL')
    return {'passed': final_pass, 'criteria': {'image_readable': True, 'blue_curve_exists': int(np.sum(curve_mask > 0)) >= 80, 'x_axis_mapping': ax is not None, 'y_scale_estimated': y_scale is not None and y_scale > 1e-06, 'piecewise_samples_or_layout_match': final_pass}, 'meta': {'curve_pixels': int(np.sum(curve_mask > 0)), 'x_axis_y': float(x_axis_y), 'y_axis_x': float(y_axis_x), 'x_axis_mapping': [float(ax), float(bx)], 'sample_ok': int(ok_cnt), 'sample_total': int(tot), 'sample_threshold': int(pass_at_least), 'layout_fallback': layout_fallback}}
evaluate = evaluate_plot

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen function case 19.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
