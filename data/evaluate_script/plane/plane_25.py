import argparse
import itertools
import math

PID = 51
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _circle_ink_coverage(img, center, radius, band_px):
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    if r <= 1.0:
        return 0.0, 0.0

    _, bw = _gray_and_ink_mask(img)
    h, w = bw.shape[:2]
    n_full = max(8, int(round(2.0 * math.pi * r / 3.0)))
    hit = 0
    vis = 0

    for i in range(n_full):
        th = (2.0 * math.pi * float(i)) / float(n_full)
        x = int(round(cx + r * math.cos(th)))
        y = int(round(cy + r * math.sin(th)))
        if not (0 <= x < w and 0 <= y < h):
            continue
        vis += 1
        ok = False
        for dr in range(-int(band_px), int(band_px) + 1):
            rr = r + float(dr)
            xx = int(round(cx + rr * math.cos(th)))
            yy = int(round(cy + rr * math.sin(th)))
            if 0 <= xx < w and 0 <= yy < h and bw[yy, xx] > 0:
                ok = True
                break
        if ok:
            hit += 1

    cov = float(hit) / float(max(1, vis))
    vis_ratio = float(vis) / float(max(1, n_full))
    return cov, vis_ratio


def _select_main_circle(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    band = int(max(1, round(scale_px(min_hw, 0.010, floor_px=0.0))))

    candidates = [detect_largest_circle(img), detect_second_largest_circle(img), detect_third_largest_circle(img)]
    valid = []
    for c in candidates:
        if c is None:
            continue
        cx, cy, r = [float(v) for v in c]
        if r < scale_px(min_hw, 0.08, floor_px=0.0):
            continue
        if r > 0.52 * min_hw:
            continue
        inside = min(cx - r, cy - r, float(w) - (cx + r), float(h) - (cy + r))
        if inside < -scale_px(min_hw, 0.02, floor_px=0.0):
            continue
        cov, vis = _circle_ink_coverage(img, (cx, cy), r, band_px=band)
        if vis < 0.45 or cov < 0.28:
            continue
        score = 6.0 * float(cov) + 3.0 * float(vis) + 0.015 * float(r) + 0.004 * max(0.0, float(inside))
        valid.append((score, (cx, cy, r)))

    if not valid:
        return None
    valid.sort(key=lambda t: t[0], reverse=True)
    return valid[0][1]


def _circle_equivalent(ca, cb, min_hw):
    center_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    radius_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    if _dist((ca[0], ca[1]), (cb[0], cb[1])) > center_tol:
        return False
    return abs(float(ca[2]) - float(cb[2])) <= radius_tol


def _dedup_dot_candidates(cands, tol):
    if not cands:
        return []
    ordered = sorted(cands, key=lambda t: (-float(t[3]), float(t[2])))
    keep = []
    for x, y, rr, fill in ordered:
        merged = False
        for qx, qy, qr, _ in keep:
            if math.hypot(float(x) - float(qx), float(y) - float(qy)) <= max(float(tol), 0.6 * max(float(rr), float(qr))):
                merged = True
                break
        if not merged:
            keep.append((float(x), float(y), float(rr), float(fill)))
    return keep


def _detect_filled_dot_candidates(img, circle):
    if img is None:
        return []

    h, w = img.shape[:2]
    m = float(min(h, w))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    blur = cv2.medianBlur(gray, 5)

    min_r = max(1, int(round(scale_px(m, 0.0025))))
    max_r = max(min_r + 1, int(round(scale_px(m, 0.014))))
    min_dist = max(1, int(round(scale_px(m, 0.03))))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=100,
        param2=12,
        minRadius=min_r,
        maxRadius=max_r,
    )

    _, bw = _gray_and_ink_mask(img)
    cands = []
    if circles is not None:
        for cc in circles[0]:
            x, y, rr = float(cc[0]), float(cc[1]), float(cc[2])
            sample_r = max(1.0, 0.80 * rr)
            x1 = max(0, int(x - sample_r - 1))
            x2 = min(w, int(x + sample_r + 2))
            y1 = max(0, int(y - sample_r - 1))
            y2 = min(h, int(y + sample_r + 2))
            if x2 <= x1 or y2 <= y1:
                continue
            yy, xx = np.ogrid[y1:y2, x1:x2]
            disk = ((xx - x) ** 2 + (yy - y) ** 2) <= (sample_r * sample_r)
            den = int(disk.sum())
            if den <= 0:
                continue
            ink = int(((bw[y1:y2, x1:x2] > 0) & disk).sum())
            fill_ratio = float(ink) / float(den)
            if fill_ratio < 0.68:
                continue
            cands.append((x, y, rr, fill_ratio))

    marker_rr = scale_px(m, 0.007)
    for px, py in detect_marker_points(img):
        if not has_point_at_point(img, (px, py)):
            continue
        cands.append((float(px), float(py), float(marker_rr), 1.0))

    if circle is not None:
        _, _, cr = [float(v) for v in circle]
        cands = [t for t in cands if float(t[2]) <= 0.16 * cr]

    return _dedup_dot_candidates(cands, tol=scale_px(m, 0.012))


def _enumerate_label_dot_assignments(best_tokens, dot_candidates, labels, max_dist, top_k=24):
    dots = [(float(d[0]), float(d[1])) for d in dot_candidates]
    if len(dots) < len(labels):
        return []

    assignments = []
    for idxs in itertools.permutations(range(len(dots)), len(labels)):
        cost = 0.0
        mapping = {}
        ok = True
        for ch, idx in zip(labels, idxs):
            tok = best_tokens[ch]
            pt = dots[idx]
            d = token_edge_distance_to_point(tok, pt)
            if d > float(max_dist):
                ok = False
                break
            mapping[ch] = pt
            cost += float(d)
        if ok:
            assignments.append((float(cost), mapping))

    assignments.sort(key=lambda t: t[0])
    if top_k is not None and int(top_k) > 0:
        return assignments[: int(top_k)]
    return assignments


def judge_plane_51(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    circle = _select_main_circle(img)
    if circle is None:
        return [False, "Failed to detect a valid main circle. "]
    cx, cy, r = [float(v) for v in circle]

    for extra in [detect_second_largest_circle(img), detect_third_largest_circle(img)]:
        if extra is None:
            continue
        ex, ey, er = [float(v) for v in extra]
        if _circle_equivalent((cx, cy, r), (ex, ey, er), min_hw=min_hw):
            continue
        inside = min(ex - er, ey - er, float(w) - (ex + er), float(h) - (ey + er))
        if inside < -scale_px(min_hw, 0.01, floor_px=0.0):
            continue
        if er < max(scale_px(min_hw, 0.07, floor_px=0.0), 0.60 * r):
            continue
        if _dist((ex, ey), (cx, cy)) <= scale_px(min_hw, 0.06, floor_px=0.0):
            continue
        return [False, "Detected extra prominent circle structure. "]

    lines = detect_line_segments(img, min_len_ratio=0.15)
    long_th = max(scale_px(min_hw, 0.24, floor_px=0.0), 0.45 * r)
    dominant = [ln for ln in lines if float(ln["len"]) >= long_th]
    if dominant:
        return [False, f"Detected extra dominant line(s): {len(dominant)}. "]

    tokens = extract_global_letter_tokens(img, whitelist="OPQ", min_conf=0.08)
    best = pick_best_tokens_by_char(tokens, ["O", "P", "Q"], min_conf=0.08)
    missing = [ch for ch in ["O", "P", "Q"] if ch not in best]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    dots = _detect_filled_dot_candidates(img, circle)
    if len(dots) < 3:
        return [False, f"Failed to detect enough point markers (found={len(dots)}). "]

    label_dot_tol = max(scale_px(min_hw, 0.03, floor_px=0.0), 0.28 * r)
    assignments = _enumerate_label_dot_assignments(
        best_tokens=best,
        dot_candidates=dots,
        labels=["O", "P", "Q"],
        max_dist=label_dot_tol,
        top_k=24,
    )
    if not assignments:
        return [False, "Failed to align labels O/P/Q with point markers. "]

    center_tol = max(scale_px(min_hw, 0.03, floor_px=0.0), 0.14 * r)
    inout_margin = max(scale_px(min_hw, 0.006, floor_px=0.0), 0.025 * r)
    sep_tol = max(scale_px(min_hw, 0.012, floor_px=0.0), 0.05 * r)

    best_fail_stage = -1
    best_fail_reason = "No O/P/Q assignment satisfies circle-center and inside/outside constraints. "

    def _record_fail(stage, reason):
        nonlocal best_fail_stage, best_fail_reason
        if int(stage) > best_fail_stage:
            best_fail_stage = int(stage)
            best_fail_reason = str(reason)

    for _, mapping in assignments:
        o_geo = mapping["O"]
        p_geo = mapping["P"]
        q_geo = mapping["Q"]

        o_dist = _dist(o_geo, (cx, cy))
        p_margin = _dist(p_geo, (cx, cy)) - r
        q_margin = _dist(q_geo, (cx, cy)) - r

        if o_dist > center_tol:
            _record_fail(1, f"Label O is not near the circle center (dist={o_dist:.1f}). ")
            continue
        if p_margin >= -inout_margin:
            _record_fail(2, f"Point P is not inside the circle (margin={p_margin:.1f}). ")
            continue
        if q_margin <= inout_margin:
            _record_fail(3, f"Point Q is not outside the circle (margin={q_margin:.1f}). ")
            continue

        if _dist(o_geo, p_geo) <= sep_tol or _dist(o_geo, q_geo) <= sep_tol or _dist(p_geo, q_geo) <= sep_tol:
            _record_fail(4, "Detected overlapping point markers among O/P/Q. ")
            continue

        return [True, ""]

    return [False, best_fail_reason]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_51,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
