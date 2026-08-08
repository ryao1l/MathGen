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
PROMPT_ID = 12
PID = 15
TYPE = 'solid'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_PLANE_DIR = Path(__file__).resolve().parent.parent.parent / 'plane'
sys.path.insert(0, str(_PLANE_DIR))
PROJ = Path(__file__).resolve().parent.parent.parent.parent.parent
IMG_ROOT = PROJ / 'data' / 'generated_img'
DEBUG_ROOT = Path(__file__).resolve().parent / 'solid_15_debug'
_REQUIRED_LABELS = set('ABCDEFGHPQR')
_CUBE_LABELS = set('ABCDEFGH')
_TRI_LABELS = set('PQR')
_MIN_LINES = 8
_MIN_ANGLE_GROUPS = 3
_CUBE_EDGES = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'E'), ('A', 'E'), ('B', 'F'), ('C', 'G'), ('D', 'H')]
_TRI_EDGES = [('P', 'Q'), ('Q', 'R'), ('P', 'R')]
_POINT_ON_EDGE = [('P', 'A', 'B'), ('Q', 'B', 'C'), ('R', 'C', 'G')]

def _detect_lines(img, h):
    min_len = max(20, int(0.06 * h))
    lines = _MODULE.detect_solid_lines(img, min_length=min_len, max_gap=12, threshold=35)
    return lines if lines else []

def _find_vertices_from_lines(lines, h, w):
    if not lines:
        return []
    endpoints = []
    for x1, y1, x2, y2 in lines:
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    ep_tol = max(25, int(0.04 * math.hypot(w, h)))
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = (lines[i], lines[j])
            ang_a = math.atan2(a[3] - a[1], a[2] - a[0])
            ang_b = math.atan2(b[3] - b[1], b[2] - b[0])
            da = abs(ang_a - ang_b)
            da = min(da, math.pi - da)
            if da < math.radians(10):
                continue
            pt = _MODULE.get_intersection_of_lines(((a[0], a[1]), (a[2], a[3])), ((b[0], b[1]), (b[2], b[3])))
            if pt is None:
                continue
            px, py = pt
            if px < -10 or px > w + 10 or py < -10 or (py > h + 10):
                continue

            def _near_ep(seg, px, py, tol):
                d1 = math.hypot(px - seg[0], py - seg[1])
                d2 = math.hypot(px - seg[2], py - seg[3])
                return min(d1, d2) <= tol
            if _near_ep(a, px, py, ep_tol) and _near_ep(b, px, py, ep_tol):
                endpoints.append((px, py))
    return endpoints

def _cluster_points(points, radius=20.0):
    if not points:
        return []
    remaining = list(points)
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        still_remaining = []
        for p in remaining:
            if math.hypot(p[0] - seed[0], p[1] - seed[1]) <= radius:
                group.append(p)
            else:
                still_remaining.append(p)
        remaining = still_remaining
        cx = sum((p[0] for p in group)) / len(group)
        cy = sum((p[1] for p in group)) / len(group)
        clusters.append((cx, cy, len(group)))
    clusters.sort(key=lambda c: -c[2])
    return [(c[0], c[1]) for c in clusters]

def _assign_labels_to_vertices(img, vertices, min_conf=0.08):
    """Global OCR -> nearest vertex assignment, with rescue pass."""
    if not vertices:
        return ({}, [])
    h, w = img.shape[:2]
    diag = math.hypot(w, h)
    max_dist = max(60, int(0.08 * diag))
    all_tokens = _PLANE.extract_global_letter_tokens(img, whitelist='ABCDEFGHPQR', min_conf=min_conf)
    label_best = {}
    for t in all_tokens:
        ch = t['char'].upper()
        if ch not in _REQUIRED_LABELS:
            continue
        tx, ty = t['center']
        conf = t['conf']
        best_vi, best_d = (-1, float('inf'))
        for vi, (vx, vy) in enumerate(vertices):
            d = math.hypot(tx - vx, ty - vy)
            if d < best_d:
                best_d = d
                best_vi = vi
        if best_d > max_dist:
            continue
        if ch not in label_best or conf > label_best[ch][0]['conf']:
            label_best[ch] = (t, best_vi, best_d)
    vertex_pos = {}
    tokens = []
    used_vertices = set()
    for ch in sorted(label_best, key=lambda c: -label_best[c][0]['conf']):
        t, vi, dist = label_best[ch]
        vx, vy = vertices[vi]
        vertex_pos[ch] = (vx, vy)
        used_vertices.add(vi)
        tokens.append({'char': ch, 'conf': t['conf'], 'center': (vx, vy), 'ocr_center': t['center'], 'dist_to_vertex': round(dist, 1)})
    missing = _REQUIRED_LABELS - set(vertex_pos.keys())
    if missing and len(used_vertices) < len(vertices):
        reader = _PLANE._get_easyocr_reader()
        if reader is not None:
            search_r = max(40, int(0.06 * diag))
            allowlist_ocr = 'ABCDEFGHPQRabcdefghpqr'
            for vi, (vx, vy) in enumerate(vertices):
                if vi in used_vertices:
                    continue
                if not missing:
                    break
                best_ch, best_conf = (None, 0.0)
                for r_mul in (0.6, 1.0, 1.4):
                    r = int(search_r * r_mul)
                    cx1 = max(0, int(vx) - r)
                    cy1 = max(0, int(vy) - r)
                    cx2 = min(w, int(vx) + r)
                    cy2 = min(h, int(vy) + r)
                    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                        continue
                    crop = img[cy1:cy2, cx1:cx2]
                    short = min(crop.shape[:2])
                    sc = max(2.0, 140.0 / max(1, short))
                    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                    for variant in (crop, 255 - crop):
                        try:
                            res = reader.readtext(variant, detail=1, paragraph=False, allowlist=allowlist_ocr)
                        except Exception:
                            continue
                        for item in res if isinstance(res, list) else []:
                            if not (isinstance(item, (list, tuple)) and len(item) >= 3):
                                continue
                            txt = str(item[1]).strip().upper()
                            try:
                                conf = float(item[2])
                            except Exception:
                                conf = 0.0
                            for c in txt:
                                if c in missing and conf > best_conf:
                                    best_ch = c
                                    best_conf = conf
                if best_ch and best_conf >= min_conf:
                    vertex_pos[best_ch] = (vx, vy)
                    used_vertices.add(vi)
                    tokens.append({'char': best_ch, 'conf': best_conf, 'center': (vx, vy), 'ocr_center': (vx, vy), 'dist_to_vertex': 0.0, 'rescue': True})
                    missing.discard(best_ch)
    return (vertex_pos, tokens)

def _check_point_on_edge(vertex_pos, pt_label, ep1_label, ep2_label, t_margin=0.02, perp_ratio=0.1):
    """Check that pt_label lies between ep1_label and ep2_label.

    Returns (ok, t_value, perp_dist_ratio, detail).
    t_value in [0,1] means between endpoints; perp_dist_ratio is
    perpendicular distance / edge_length.
    """
    for lbl in (pt_label, ep1_label, ep2_label):
        if lbl not in vertex_pos:
            return (False, None, None, f'missing label {lbl}')
    px, py = vertex_pos[pt_label]
    ax, ay = vertex_pos[ep1_label]
    bx, by = vertex_pos[ep2_label]
    edge_len = math.hypot(bx - ax, by - ay)
    if edge_len < 5:
        return (False, None, None, 'degenerate edge')
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / edge_len ** 2
    perp = abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / edge_len
    perp_r = perp / edge_len
    ok = t_margin <= t <= 1.0 - t_margin and perp_r <= perp_ratio
    detail = '' if ok else f'{pt_label} not on {ep1_label}{ep2_label}: t={t:.2f} perp_r={perp_r:.3f}'
    return (ok, round(t, 3), round(perp_r, 4), detail)

def _fix_duplicate_cube_vertices(vertex_pos, vertices, tokens):
    """Fix F when it shares a vertex with E using depth vector prediction."""
    if 'E' not in vertex_pos or 'F' not in vertex_pos:
        return
    ex, ey = vertex_pos['E']
    fx, fy = vertex_pos['F']
    if math.hypot(fx - ex, fy - ey) > 15:
        return
    needed = {'A', 'B', 'D', 'H'}
    if not needed.issubset(vertex_pos.keys()):
        return
    ax, ay = vertex_pos['A']
    bx, by = vertex_pos['B']
    dx, dy = vertex_pos['D']
    hx, hy = vertex_pos['H']
    dv_ae = (ex - ax, ey - ay)
    dv_dh = (hx - dx, hy - dy)
    len_ae = math.hypot(*dv_ae)
    len_dh = math.hypot(*dv_dh)
    if min(len_ae, len_dh) < 20:
        return
    if max(len_ae, len_dh) / min(len_ae, len_dh) > 2.0:
        return
    angle_ae = math.atan2(dv_ae[1], dv_ae[0])
    angle_dh = math.atan2(dv_dh[1], dv_dh[0])
    angle_diff = abs(angle_ae - angle_dh)
    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
    if angle_diff > math.radians(30):
        return
    dvx = (dv_ae[0] + dv_dh[0]) / 2
    dvy = (dv_ae[1] + dv_dh[1]) / 2
    pred_fx = bx + dvx
    pred_fy = by + dvy
    used = set()
    for ch, (cx, cy) in vertex_pos.items():
        for vi, (vx, vy) in enumerate(vertices):
            if abs(cx - vx) < 1 and abs(cy - vy) < 1:
                used.add(vi)
    best_vi, best_d = (-1, float('inf'))
    for vi, (vx, vy) in enumerate(vertices):
        if vi in used:
            continue
        d = math.hypot(vx - pred_fx, vy - pred_fy)
        if d < best_d:
            best_d = d
            best_vi = vi
    if best_vi >= 0 and best_d < 60:
        vertex_pos['F'] = vertices[best_vi]
        for t in tokens:
            if t['char'] == 'F':
                t['center'] = vertices[best_vi]
                t['note'] = f'F reassigned via depth vector (d={best_d:.1f})'

def _load_and_downscale(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return (None, 0, 0)
    _TARGET_LONG = 1024
    long_side = max(img.shape[:2])
    if long_side > _TARGET_LONG:
        s = _TARGET_LONG / long_side
        img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return (img, img.shape[0], img.shape[1])

def _run_analysis(img, h, w):
    """Shared analysis pipeline."""
    gray, bw, ink_ratio = _MODULE._binarize(img)
    lines = _detect_lines(img, h)
    n_lines = len(lines)
    diverse, n_groups = _MODULE._has_line_diversity(bw, min_angle_groups=_MIN_ANGLE_GROUPS)
    has_dashed = _MODULE._has_dashed_lines(bw)
    raw_pts = _find_vertices_from_lines(lines, h, w)
    cluster_rad = max(15, int(0.02 * math.hypot(w, h)))
    vertices = _cluster_points(raw_pts, radius=cluster_rad)
    vertex_pos, tokens = _assign_labels_to_vertices(img, vertices, min_conf=0.08)
    _fix_duplicate_cube_vertices(vertex_pos, vertices, tokens)
    found = set(vertex_pos.keys())
    missing = _REQUIRED_LABELS - found
    cube_edge_pass_ok, cube_edge_n_pass, cube_edge_n_total, cube_edge_failed = (False, 0, 12, [])
    if len(found & _CUBE_LABELS) >= 5:
        cube_edge_pass_ok, cube_edge_n_pass, cube_edge_n_total, cube_edge_failed = _MODULE.check_edge_connectivity(img, vertex_pos, _CUBE_EDGES, allow_dash=True, min_pass=8)
    sep_ok, sep_msg = (True, '')
    cube_found = sorted(found & _CUBE_LABELS)
    if len(cube_found) >= 5:
        sep_ok, sep_msg = _MODULE.check_vertices_separated(vertex_pos, cube_found, (w, h), min_sep_ratio=0.02)
    topo_ok = cube_edge_pass_ok and sep_ok
    p_on_ab_ok, p_t, p_perp_r, p_detail = _check_point_on_edge(vertex_pos, 'P', 'A', 'B')
    q_not_misplaced = True
    q_detail = ''
    q_token = None
    for t in tokens:
        if t['char'] == 'Q':
            if q_token is None or t['conf'] > q_token['conf']:
                q_token = t
    if q_token is not None and q_token['conf'] >= 0.5:
        qx, qy = q_token['ocr_center']
        if 'B' in vertex_pos and 'C' in vertex_pos:
            bx, by = vertex_pos['B']
            cx, cy = vertex_pos['C']
            edge_len = math.hypot(cx - bx, cy - by)
            if edge_len > 5:
                qt = ((qx - bx) * (cx - bx) + (qy - by) * (cy - by)) / edge_len ** 2
                qperp = abs((cx - bx) * (by - qy) - (bx - qx) * (cy - by)) / edge_len
                qperp_r = qperp / edge_len
                if not (0.05 <= qt <= 0.95 and qperp_r <= 0.2):
                    q_not_misplaced = False
                    q_detail = f"Q conf={q_token['conf']:.2f} at ({qx:.0f},{qy:.0f}) not on BC: t={qt:.2f} perp_r={qperp_r:.3f}"
    return {'gray': gray, 'bw': bw, 'ink_ratio': ink_ratio, 'lines': lines, 'n_lines': n_lines, 'diverse': diverse, 'n_groups': n_groups, 'has_dashed': has_dashed, 'raw_pts': raw_pts, 'vertices': vertices, 'vertex_pos': vertex_pos, 'tokens': tokens, 'found': found, 'missing': missing, 'cube_edge_n_pass': cube_edge_n_pass, 'cube_edge_n_total': cube_edge_n_total, 'cube_edge_failed': cube_edge_failed, 'sep_ok': sep_ok, 'sep_msg': sep_msg, 'topo_ok': topo_ok, 'p_on_ab_ok': p_on_ab_ok, 'p_t': p_t, 'p_perp_r': p_perp_r, 'p_detail': p_detail, 'q_not_misplaced': q_not_misplaced, 'q_detail': q_detail}

def judge_cube_triangle(image_path: str):
    """Evaluate whether image depicts a cube with triangle PQR."""
    _CRITERIA_KEYS = ['image_readable', 'foreground_present', 'sufficient_structure', 'line_angle_diversity', 'dashed_lines_present', 'vertices_detected', 'labels_found', 'cube_topology_ok', 'p_on_ab_ok', 'q_not_misplaced']
    _META_KEYS = ['case_id', 'ink_ratio', 'line_count', 'angle_groups', 'n_vertices', 'labels_detected', 'labels_missing', 'cube_edge_connectivity', 'cube_edge_failed', 'p_on_ab_detail', 'q_detail']

    def _result(criteria, meta):
        full_c = {k: criteria.get(k) for k in _CRITERIA_KEYS}
        full_m = {k: meta.get(k) for k in _META_KEYS}
        passed = all((v is True for v in full_c.values()))
        return {'passed': passed, 'criteria': full_c, 'meta': full_m}
    img, h, w = _load_and_downscale(image_path)
    if img is None:
        return _result({'image_readable': False}, {'case_id': PID})
    a = _run_analysis(img, h, w)
    criteria = {'image_readable': True}
    meta = {'case_id': PID, 'ink_ratio': round(a['ink_ratio'], 4)}
    criteria['foreground_present'] = a['ink_ratio'] > 0.003
    if not criteria['foreground_present']:
        return _result(criteria, meta)
    meta['line_count'] = a['n_lines']
    criteria['sufficient_structure'] = a['n_lines'] >= _MIN_LINES
    criteria['line_angle_diversity'] = a['diverse']
    meta['angle_groups'] = a['n_groups']
    criteria['dashed_lines_present'] = a['has_dashed']
    meta['n_vertices'] = len(a['vertices'])
    criteria['vertices_detected'] = len(a['vertices']) >= 8
    meta['labels_detected'] = sorted(a['found'])
    meta['labels_missing'] = sorted(a['missing'])
    criteria['labels_found'] = len(a['missing']) == 0
    meta['cube_edge_connectivity'] = f"{a['cube_edge_n_pass']}/{a['cube_edge_n_total']}"
    meta['cube_edge_failed'] = a['cube_edge_failed']
    criteria['cube_topology_ok'] = a['topo_ok']
    meta['p_on_ab_detail'] = a['p_detail'] if a['p_detail'] else 'OK'
    criteria['p_on_ab_ok'] = a['p_on_ab_ok']
    meta['q_detail'] = a['q_detail'] if a['q_detail'] else 'OK'
    criteria['q_not_misplaced'] = a['q_not_misplaced']
    return _result(criteria, meta)

def evaluate(image_path: str):
    return judge_cube_triangle(image_path)

def draw_debug(image_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, str(out_dir / '0_original.png'))
    img, h, w = _load_and_downscale(image_path)
    if img is None:
        print(f'  [SKIP] Cannot read {image_path}')
        return None
    a = _run_analysis(img, h, w)
    cv2.imwrite(str(out_dir / '1_binarized.png'), a['bw'])
    ve = cv2.Canny(a['gray'], 50, 150)
    edge_vis = cv2.cvtColor(ve, cv2.COLOR_GRAY2BGR)
    cv2.putText(edge_vis, 'Validation edge map (Canny)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / '2_edges.png'), edge_vis)
    lines_vis = img.copy()
    for idx, (x1, y1, x2, y2) in enumerate(a['lines']):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        cv2.line(lines_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        mx, my = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.putText(lines_vis, f'#{idx} L={int(seg_len)}', (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    ok_lines = a['n_lines'] >= _MIN_LINES
    cv2.putText(lines_vis, f"lines={a['n_lines']} (need>={_MIN_LINES}: {('OK' if ok_lines else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ok_lines else (0, 0, 255), 1)
    cv2.putText(lines_vis, f"angle_groups={a['n_groups']} (need>={_MIN_ANGLE_GROUPS}: {('OK' if a['diverse'] else 'FAIL')})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if a['diverse'] else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '3_lines.png'), lines_vis)
    vtx_vis = img.copy()
    for px, py in a['raw_pts']:
        cv2.circle(vtx_vis, (int(px), int(py)), 3, (200, 200, 200), -1)
    for idx, (vx, vy) in enumerate(a['vertices']):
        cv2.circle(vtx_vis, (int(vx), int(vy)), 8, (0, 0, 255), 2)
        cv2.putText(vtx_vis, f'v{idx}', (int(vx) + 10, int(vy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    n_vtx = len(a['vertices'])
    cv2.putText(vtx_vis, f"vertices={n_vtx} (need>=8: {('OK' if n_vtx >= 8 else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if n_vtx >= 8 else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '4_vertices.png'), vtx_vis)
    dashed_vis = img.copy()
    short_lines = cv2.HoughLinesP(a['bw'], 1, np.pi / 180.0, 15, minLineLength=3, maxLineGap=2)
    short_count = 0
    if short_lines is not None:
        short_threshold = max(20, int(min(h, w) * 0.03))
        for l in short_lines:
            x1, y1, x2, y2 = l[0]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if 3 <= seg_len <= short_threshold:
                cv2.line(dashed_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                short_count += 1
    cv2.putText(dashed_vis, f"dashed: short_segs={short_count} ({('OK' if a['has_dashed'] else 'FAIL')})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if a['has_dashed'] else (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / '5_dashed.png'), dashed_vis)
    label_vis = img.copy()
    for vx, vy in a['vertices']:
        cv2.circle(label_vis, (int(vx), int(vy)), 5, (180, 180, 180), 1)
    vp = a['vertex_pos']
    for u, v in _CUBE_EDGES:
        if u in vp and v in vp:
            p1 = (int(vp[u][0]), int(vp[u][1]))
            p2 = (int(vp[v][0]), int(vp[v][1]))
            cv2.line(label_vis, p1, p2, (255, 200, 0), 1)
    for t in a['tokens']:
        ch = t['char']
        cx, cy = t['center']
        conf = t['conf']
        color = (0, 255, 0) if ch in _CUBE_LABELS else (0, 255, 255)
        cv2.circle(label_vis, (int(cx), int(cy)), 8, color, -1)
        cv2.putText(label_vis, f'{ch} ({conf:.2f})', (int(cx) + 10, int(cy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(label_vis, f"labels: {sorted(a['found'])} missing={sorted(a['missing'])}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if not a['missing'] else (0, 0, 255), 1)
    cv2.putText(label_vis, f"cube edges: {a['cube_edge_n_pass']}/{a['cube_edge_n_total']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if a['topo_ok'] else (0, 0, 255), 1)
    p_color = (0, 255, 0) if a['p_on_ab_ok'] else (0, 0, 255)
    p_text = f"P on AB: {('OK' if a['p_on_ab_ok'] else a['p_detail'])}"
    cv2.putText(label_vis, p_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, p_color, 1)
    q_color = (0, 255, 0) if a['q_not_misplaced'] else (0, 0, 255)
    q_text = f"Q placement: {('OK' if a['q_not_misplaced'] else a['q_detail'])}"
    cv2.putText(label_vis, q_text, (10, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.4, q_color, 1)
    cv2.imwrite(str(out_dir / '6_labels.png'), label_vis)
    result = judge_cube_triangle(image_path)
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
        img_path = IMG_ROOT / model / 'solid' / f'{PID}.png'
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
    parser = argparse.ArgumentParser(description='Evaluate MathGen solid case 12.')
    parser.add_argument('--image', type=str, required=True, help='Path to the generated image.')
    args = parser.parse_args()
    print_report(evaluate(args.image))
if __name__ == '__main__':
    main()
