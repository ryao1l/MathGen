import argparse
import itertools
import math

PID = 60
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


def _circle_equivalent(ca, cb, min_hw):
    center_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    radius_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    if _dist((ca[0], ca[1]), (cb[0], cb[1])) > center_tol:
        return False
    return abs(float(ca[2]) - float(cb[2])) <= radius_tol


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
        if r > 0.54 * min_hw:
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


def _dedup_dot_candidates(cands, tol):
    if not cands:
        return []
    ordered = sorted(cands, key=lambda t: (-float(t[3]), float(t[2])))
    keep = []
    for x, y, rr, fill in ordered:
        merged = False
        for qx, qy, qr, _ in keep:
            merge_tol = max(float(tol), 0.6 * max(float(rr), float(qr)))
            if math.hypot(float(x) - float(qx), float(y) - float(qy)) <= merge_tol:
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


def _project_to_circle(point, circle):
    px, py = float(point[0]), float(point[1])
    cx, cy, r = [float(v) for v in circle]
    vx = px - cx
    vy = py - cy
    norm = math.hypot(vx, vy)
    if norm <= 1e-6:
        return (cx, cy - r)
    return (cx + r * vx / norm, cy + r * vy / norm)


def _merge_candidate_items(items, tol, max_keep=8):
    if not items:
        return []
    ordered = sorted(items, key=lambda it: float(it["cost"]))
    keep = []
    for it in ordered:
        p = it["point"]
        if any(_dist(p, jt["point"]) <= float(tol) for jt in keep):
            continue
        keep.append(it)
        if len(keep) >= int(max_keep):
            break
    return keep


def judge_plane_60(img):
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
        if inside < -scale_px(min_hw, 0.02, floor_px=0.0):
            continue
        if er < max(scale_px(min_hw, 0.07, floor_px=0.0), 0.62 * r):
            continue
        if _dist((ex, ey), (cx, cy)) <= scale_px(min_hw, 0.06, floor_px=0.0):
            continue
        return [False, "Detected extra prominent circle structure. "]

    lines = detect_line_segments(img, min_len_ratio=0.16)
    dominant_len_th = max(scale_px(min_hw, 0.24, floor_px=0.0), 0.48 * r)
    dominant = [ln for ln in lines if float(ln.get("len", 0.0)) >= dominant_len_th]
    if dominant:
        return [False, f"Detected extra dominant line(s): {len(dominant)}. "]

    tokens = extract_global_letter_tokens(img, whitelist="ABOP", min_conf=0.08)
    best = pick_best_tokens_by_char(tokens, ["A", "B", "O", "P"], min_conf=0.08)
    missing = [ch for ch in ["A", "B", "O", "P"] if ch not in best]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    dots = _detect_filled_dot_candidates(img, circle)
    dot_points = [(float(d[0]), float(d[1])) for d in dots]
    marker_points = [(float(m[0]), float(m[1])) for m in detect_marker_points(img)]

    center = (cx, cy)
    dedup_tol = scale_px(min_hw, 0.010, floor_px=2.5)

    on_tol = max(scale_px(min_hw, 0.026, floor_px=0.0), 0.08 * r)
    center_tol = max(scale_px(min_hw, 0.028, floor_px=0.0), 0.12 * r)
    p_inside_soft_tol = max(scale_px(min_hw, 0.005, floor_px=0.0), 0.015 * r)
    p_strict_margin = max(scale_px(min_hw, 0.009, floor_px=0.0), 0.030 * r)
    point_sep_tol = max(scale_px(min_hw, 0.012, floor_px=0.0), 0.05 * r)
    ab_sep_tol = max(scale_px(min_hw, 0.055, floor_px=0.0), 0.16 * r)

    label_attach_ab = max(scale_px(min_hw, 0.14, floor_px=0.0), 0.32 * r)
    label_attach_o = max(scale_px(min_hw, 0.10, floor_px=0.0), 0.24 * r)
    label_attach_p = max(scale_px(min_hw, 0.10, floor_px=0.0), 0.24 * r)

    label_max_a = max(scale_px(min_hw, 0.12, floor_px=0.0), 0.24 * r)
    label_max_b = label_max_a
    label_max_o = max(scale_px(min_hw, 0.10, floor_px=0.0), 0.20 * r)
    label_max_p = max(scale_px(min_hw, 0.10, floor_px=0.0), 0.22 * r)

    cand = {"A": [], "B": [], "O": [], "P": []}

    for ch in ["A", "B"]:
        tok = best[ch]
        t_center = (float(tok["center"][0]), float(tok["center"][1]))
        local_items = []

        for p in dot_points:
            if _dist(p, t_center) > label_attach_ab:
                continue
            radial_err = abs(_dist(p, center) - r)
            if radial_err > 1.35 * on_tol:
                continue
            cst = token_edge_distance_to_point(tok, p) + 2.0 * float(radial_err)
            local_items.append({"point": p, "cost": float(cst), "kind": "dot"})

        for p in marker_points:
            if _dist(p, t_center) > label_attach_ab:
                continue
            radial_err = abs(_dist(p, center) - r)
            if radial_err > 1.35 * on_tol:
                continue
            cst = token_edge_distance_to_point(tok, p) + 2.2 * float(radial_err) + 4.0
            local_items.append({"point": p, "cost": float(cst), "kind": "marker"})

        proj = _project_to_circle(t_center, circle)
        proj_cost = token_edge_distance_to_point(tok, proj) + 0.35 * float(label_attach_ab) + 2.0
        local_items.append({"point": proj, "cost": float(proj_cost), "kind": "projection"})

        cand[ch] = _merge_candidate_items(local_items, tol=dedup_tol, max_keep=8)
        if not cand[ch]:
            return [False, f"Failed to build geometric candidates for point {ch}. "]

    tok_o = best["O"]
    o_center = (float(tok_o["center"][0]), float(tok_o["center"][1]))
    o_items = [
        {
            "point": center,
            "cost": float(token_edge_distance_to_point(tok_o, center) + 1.5),
            "kind": "center_fallback",
        }
    ]
    for p in dot_points:
        if _dist(p, o_center) > label_attach_o:
            continue
        d_ctr = _dist(p, center)
        cst = token_edge_distance_to_point(tok_o, p) + 0.55 * float(d_ctr)
        o_items.append({"point": p, "cost": float(cst), "kind": "dot"})
    for p in marker_points:
        if _dist(p, o_center) > label_attach_o:
            continue
        d_ctr = _dist(p, center)
        cst = token_edge_distance_to_point(tok_o, p) + 0.65 * float(d_ctr) + 2.0
        o_items.append({"point": p, "cost": float(cst), "kind": "marker"})

    cand["O"] = _merge_candidate_items(o_items, tol=dedup_tol, max_keep=8)
    if not cand["O"]:
        return [False, "Failed to build geometric candidates for point O. "]

    tok_p = best["P"]
    p_center = (float(tok_p["center"][0]), float(tok_p["center"][1]))
    p_items = [
        {
            "point": p_center,
            "cost": float(2.0),
            "kind": "token_fallback",
        }
    ]
    for p in dot_points:
        if _dist(p, p_center) > label_attach_p:
            continue
        d = _dist(p, center)
        if d > r + p_inside_soft_tol:
            continue
        margin_in = r - d
        boundary_pen = max(0.0, p_strict_margin - margin_in)
        cst = token_edge_distance_to_point(tok_p, p) + 4.5 * float(boundary_pen)
        p_items.append({"point": p, "cost": float(cst), "kind": "dot"})
    for p in marker_points:
        if _dist(p, p_center) > label_attach_p:
            continue
        d = _dist(p, center)
        if d > r + p_inside_soft_tol:
            continue
        margin_in = r - d
        boundary_pen = max(0.0, p_strict_margin - margin_in)
        cst = token_edge_distance_to_point(tok_p, p) + 5.0 * float(boundary_pen) + 1.5
        p_items.append({"point": p, "cost": float(cst), "kind": "marker"})

    cand["P"] = _merge_candidate_items(p_items, tol=dedup_tol, max_keep=10)
    if not cand["P"]:
        return [False, "Failed to build geometric candidates for point P. "]

    best_score = None
    best_fail_stage = -1
    best_fail_reason = "No A/B/O/P assignment satisfies center, on-circle and strict-inside-circle constraints. "

    def _record_fail(stage, reason):
        nonlocal best_fail_stage, best_fail_reason
        if int(stage) > best_fail_stage:
            best_fail_stage = int(stage)
            best_fail_reason = str(reason)

    for a_it, b_it, o_it, p_it in itertools.product(cand["A"], cand["B"], cand["O"], cand["P"]):
        a_geo = a_it["point"]
        b_geo = b_it["point"]
        o_geo = o_it["point"]
        p_geo = p_it["point"]

        if (
            _dist(a_geo, b_geo) <= point_sep_tol
            or _dist(a_geo, o_geo) <= point_sep_tol
            or _dist(a_geo, p_geo) <= point_sep_tol
            or _dist(b_geo, o_geo) <= point_sep_tol
            or _dist(b_geo, p_geo) <= point_sep_tol
            or _dist(o_geo, p_geo) <= point_sep_tol
        ):
            _record_fail(1, "Detected overlapping or indistinguishable points among A/B/O/P. ")
            continue

        o_dist = _dist(o_geo, center)
        if o_dist > center_tol:
            _record_fail(2, f"Point O is not near the circle center (dist={o_dist:.1f}). ")
            continue

        a_rad_err = abs(_dist(a_geo, center) - r)
        b_rad_err = abs(_dist(b_geo, center) - r)
        if a_rad_err > on_tol:
            _record_fail(3, f"Point A is not on the circle (radial_err={a_rad_err:.1f}). ")
            continue
        if b_rad_err > on_tol:
            _record_fail(3, f"Point B is not on the circle (radial_err={b_rad_err:.1f}). ")
            continue

        ab_len = _dist(a_geo, b_geo)
        if ab_len < ab_sep_tol:
            _record_fail(4, f"Points A and B are too close on the circle (dist={ab_len:.1f}). ")
            continue

        p_margin = _dist(p_geo, center) - r
        if p_margin > p_inside_soft_tol:
            _record_fail(5, f"Point P is outside the circle (margin={p_margin:.1f}). ")
            continue
        if p_margin > -p_strict_margin:
            _record_fail(6, f"Point P is not strictly inside the circle (margin={p_margin:.1f}). ")
            continue

        d_a = token_edge_distance_to_point(best["A"], a_geo)
        d_b = token_edge_distance_to_point(best["B"], b_geo)
        d_o = token_edge_distance_to_point(best["O"], o_geo)
        d_p = token_edge_distance_to_point(best["P"], p_geo)

        if d_a > label_max_a:
            _record_fail(7, f"Label A is too far from point A (dist={d_a:.1f}). ")
            continue
        if d_b > label_max_b:
            _record_fail(7, f"Label B is too far from point B (dist={d_b:.1f}). ")
            continue
        if d_o > label_max_o:
            _record_fail(7, f"Label O is too far from point O (dist={d_o:.1f}). ")
            continue
        if d_p > label_max_p:
            _record_fail(7, f"Label P is too far from point P (dist={d_p:.1f}). ")
            continue

        score = (
            float(a_it["cost"] + b_it["cost"] + o_it["cost"] + p_it["cost"])
            + 2.4 * float(o_dist)
            + 2.8 * float(a_rad_err + b_rad_err)
            + 0.06 * float(d_a + d_b + d_o + d_p)
            - 0.26 * float(-p_margin)
        )

        if best_score is None or score < best_score:
            best_score = float(score)

    if best_score is None:
        return [False, best_fail_reason]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_60,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
