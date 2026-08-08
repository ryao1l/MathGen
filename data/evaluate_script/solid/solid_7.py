#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'plane'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator_utils import print_report
import solid_common
import solid_common as _MODULE
import plane_common
import plane_common as _PLANE
from solid_common import *
from plane_common import *
PROMPT_ID = 7
PID = 8
TYPE = 'solid'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJ = Path(__file__).resolve().parent.parent.parent.parent.parent
IMG_ROOT = PROJ / 'data' / 'generated_img'
DEBUG_ROOT = Path(__file__).resolve().parent / 'solid_8_debug'

def evaluate(image_path: str):
    return _MODULE.judge_cone(image_path)

def _load_and_downscale(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return (None, 0, 0)
    _TARGET_LONG = 1024
    long_side = max(img.shape[:2])
    if long_side > _TARGET_LONG:
        s = _TARGET_LONG / long_side
        img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    return (img, h, w)

def draw_debug(image_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, str(out_dir / '0_original.png'))
    img, h, w = _load_and_downscale(image_path)
    if img is None:
        print(f'  [SKIP] Cannot read {image_path}')
        return None
    gray, bw, _ = _MODULE._binarize(img)
    cv2.imwrite(str(out_dir / '1_binarized.png'), bw)
    ve = cv2.Canny(gray, 50, 150)
    ve_d = cv2.dilate(ve, np.ones((5, 5), np.uint8))
    edge_vis = cv2.cvtColor(ve, cv2.COLOR_GRAY2BGR)
    cv2.putText(edge_vis, 'Validation edge map (Canny)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / '2_edges.png'), edge_vis)
    ellipse = _MODULE._detect_ellipse_contour(bw, gray=gray, color_img=img)
    ellipse_vis = img.copy()
    if ellipse is not None:
        (ecx, ecy), (ea, eb), eang = ellipse
        bnd = cv2.ellipse2Poly((int(round(ecx)), int(round(ecy))), (max(1, int(round(ea / 2))), max(1, int(round(eb / 2)))), int(round(eang)), 0, 360, 5)
        n_in = sum((1 for pt in bnd if 0 <= pt[0] < w and 0 <= pt[1] < h))
        n_edge = sum((1 for pt in bnd if 0 <= pt[0] < w and 0 <= pt[1] < h and (ve_d[pt[1], pt[0]] > 0)))
        er = n_edge / n_in if n_in else 0
        for pt in bnd:
            px, py = (int(pt[0]), int(pt[1]))
            if 0 <= px < w and 0 <= py < h:
                c = (0, 255, 0) if ve_d[py, px] > 0 else (0, 0, 255)
                cv2.circle(ellipse_vis, (px, py), 2, c, -1)
        cv2.ellipse(ellipse_vis, ellipse, (0, 255, 0), 2)
        cv2.circle(ellipse_vis, (int(ecx), int(ecy)), 5, (0, 0, 255), -1)
        cv2.putText(ellipse_vis, f'c=({int(ecx)},{int(ecy)}) ax=({int(ea)},{int(eb)}) edge={er:.0%}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(ellipse_vis, 'ellipse_base: TRUE', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        ecx, ecy = (w / 2.0, h * 0.65)
        ea = eb = 0
        cv2.putText(ellipse_vis, 'No ellipse found => ellipse_base: FALSE', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(ellipse_vis, (int(ecx), int(ecy)), 5, (0, 165, 255), -1)
    cv2.line(ellipse_vis, (0, int(ecy)), (w, int(ecy)), (200, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / '3_ellipse.png'), ellipse_vis)
    min_len = max(25, int(0.08 * h))
    solid_lines = _MODULE.detect_solid_lines(img, min_length=min_len, max_gap=10, threshold=40)
    n_lines = len(solid_lines) if solid_lines else 0
    gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
    soft_edges = cv2.Canny(gray_b, 30, 90, apertureSize=3)
    raw_soft = cv2.HoughLinesP(soft_edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=10)
    soft_lines = None
    if raw_soft is not None:
        _ss = []
        for seg in raw_soft:
            x1s, y1s, x2s, y2s = seg[0]
            if math.hypot(x2s - x1s, y2s - y1s) >= min_len:
                _ss.append((float(x1s), float(y1s), float(x2s), float(y2s)))
        if _ss:
            soft_lines = _MODULE._merge_collinear_segments(_ss)
            soft_lines.sort(key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
    apex_hard = _MODULE._detect_cone_apex(solid_lines or [], ecx, ecy, h, w)
    apex_soft = _MODULE._detect_cone_apex(soft_lines or [], ecx, ecy, h, w) if soft_lines else None
    if apex_soft is not None and (apex_hard is None or abs(apex_soft[1] - ecy) > abs(apex_hard[1] - ecy)):
        used_lines, used_soft = (soft_lines, True)
    else:
        used_lines, used_soft = (solid_lines, False)
    lines_vis = img.copy()
    if solid_lines and (not used_soft):
        for idx, (x1, y1, x2, y2) in enumerate(solid_lines):
            cv2.line(lines_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            mx, my = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.putText(lines_vis, f'#{idx} L={int(math.hypot(x2 - x1, y2 - y1))}', (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    if soft_lines and used_soft:
        for idx, (x1, y1, x2, y2) in enumerate(soft_lines):
            cv2.line(lines_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            mx, my = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.putText(lines_vis, f's#{idx} L={int(math.hypot(x2 - x1, y2 - y1))}', (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1)
    n_disp = len(soft_lines) if used_soft else n_lines
    lbl = ' (soft)' if used_soft else ''
    cv2.putText(lines_vis, f"lines={n_disp}{lbl} (need>=2: {('OK' if n_disp >= 2 else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if n_disp >= 2 else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '4_lines.png'), lines_vis)
    apex = apex_soft if used_soft else apex_hard
    vis_lines = used_lines if used_lines else solid_lines or []
    if apex is not None:
        semi_major = max(ea, eb) if ellipse is not None else w * 0.25
        apex_x_tol = max(semi_major * 1.5, w * 0.15)
        if abs(apex[0] - ecx) > apex_x_tol:
            apex = None
    apex_vis = img.copy()
    cv2.line(apex_vis, (0, int(ecy)), (w, int(ecy)), (200, 200, 0), 1, cv2.LINE_AA)
    cv2.circle(apex_vis, (int(ecx), int(ecy)), 5, (0, 165, 255), -1)
    if vis_lines:
        for x1, y1, x2, y2 in vis_lines:
            cv2.line(apex_vis, (int(x1), int(y1)), (int(x2), int(y2)), (180, 180, 180), 1)
    if apex is not None:
        if ellipse is not None:
            angles_t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            cos_th = np.cos(np.radians(eang))
            sin_th = np.sin(np.radians(eang))
            sa, sb = (ea / 2, eb / 2)
            ex = ecx + sa * np.cos(angles_t) * cos_th - sb * np.sin(angles_t) * sin_th
            ey = ecy + sa * np.cos(angles_t) * sin_th + sb * np.sin(angles_t) * cos_th
            theta = np.arctan2(ey - apex[1], ex - apex[0])
            i_l, i_r = (int(np.argmin(theta)), int(np.argmax(theta)))
            cv2.line(apex_vis, (int(apex[0]), int(apex[1])), (int(ex[i_l]), int(ey[i_l])), (0, 255, 0), 2)
            cv2.line(apex_vis, (int(apex[0]), int(apex[1])), (int(ex[i_r]), int(ey[i_r])), (0, 255, 0), 2)
            cv2.circle(apex_vis, (int(ex[i_l]), int(ey[i_l])), 4, (255, 0, 255), -1)
            cv2.circle(apex_vis, (int(ex[i_r]), int(ey[i_r])), 4, (255, 0, 255), -1)
        cv2.circle(apex_vis, (int(apex[0]), int(apex[1])), 8, (0, 0, 255), -1)
        cv2.putText(apex_vis, f'APEX ({int(apex[0])},{int(apex[1])})', (int(apex[0]) + 10, int(apex[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        above = apex[1] != ecy
        cv2.putText(apex_vis, f"above_base: {('TRUE' if above else 'FALSE')}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if above else (0, 0, 255), 1)
    else:
        cv2.putText(apex_vis, 'APEX: NOT FOUND', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '5_apex.png'), apex_vis)
    conn_vis = img.copy()
    if ellipse is not None:
        cv2.ellipse(conn_vis, ellipse, (0, 255, 0), 1)
    if apex is not None:
        cv2.circle(conn_vis, (int(apex[0]), int(apex[1])), 6, (0, 0, 255), -1)
    any_conn = False
    if apex is not None and apex[1] != ecy and vis_lines:
        for x1, y1, x2, y2 in vis_lines:
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len < 0.15 * h:
                cv2.line(conn_vis, (int(x1), int(y1)), (int(x2), int(y2)), (180, 180, 180), 1)
                continue
            ok = _MODULE.has_line_between_points(img, (x1, y1), (x2, y2), allow_dash=False)
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.line(conn_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            mx, my = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.putText(conn_vis, 'CONN' if ok else 'DISC', (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            if ok:
                any_conn = True
    cv2.putText(conn_vis, f"apex_connected: {('TRUE' if any_conn else 'FALSE')}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if any_conn else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '6_connectivity.png'), conn_vis)
    result = _MODULE.judge_cone(image_path)
    summary_vis = img.copy()
    y_pos = 25
    for k, v in result['criteria'].items():
        color = (0, 255, 0) if v else (0, 0, 255)
        cv2.putText(summary_vis, f'{k}: {v}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 22
    verdict = 'PASSED' if result['passed'] else 'FAILED'
    color = (0, 255, 0) if result['passed'] else (0, 0, 255)
    cv2.putText(summary_vis, verdict, (10, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(str(out_dir / '7_summary.png'), summary_vis)
    with open(str(out_dir / 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result

def run_debug_all():
    models = sorted((d.name for d in IMG_ROOT.iterdir() if d.is_dir()))
    results = {}
    for model in models:
        img_path = IMG_ROOT / model / 'solid' / '8.png'
        if not img_path.exists():
            print(f'[SKIP] {model}: no image at {img_path}')
            continue
        out_dir = DEBUG_ROOT / model
        print(f'[{model}] {img_path} -> {out_dir}')
        r = draw_debug(str(img_path), out_dir)
        if r:
            results[model] = r['passed']
    print('\n=== Summary ===')
    for model, passed in sorted(results.items()):
        print(f"  {model:25s} {('PASS' if passed else 'FAIL')}")
    print(f'\n  {sum(results.values())}/{len(results)} passed')

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MathGen solid case 7.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
