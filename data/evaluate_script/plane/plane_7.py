import argparse

PID = 29
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _segment_item(p, q):
    seg = (float(p[0]), float(p[1]), float(q[0]), float(q[1]))
    abc = segment_to_abc(seg)
    if abc is None:
        raise RuntimeError("Failed to build segment line.")
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _pin_to_marker(p, markers, tol):
    if not markers:
        return (float(p[0]), float(p[1]))
    q = nearest_point(markers, p)
    if q is None:
        return (float(p[0]), float(p[1]))
    if _dist(q, p) <= float(tol):
        return (float(q[0]), float(q[1]))
    return (float(p[0]), float(p[1]))


def _line_endpoints_lr(ln):
    x1, y1, x2, y2 = [float(v) for v in ln["seg"]]
    pts = [(x1, y1), (x2, y2)]
    pts.sort(key=lambda p: p[0])
    return pts[0], pts[1]


def _local_label_ink(img, point, min_hw):
    if img is None or point is None:
        return 0
    h, w = img.shape[:2]
    px, py = float(point[0]), float(point[1])
    _, bw = _gray_and_ink_mask(img)
    half = scale_px(min_hw, 0.055, floor_px=6.0)
    x1 = int(max(0, px - half))
    x2 = int(min(w, px + half))
    y1 = int(max(0, py - half))
    y2 = int(min(h, py + half))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    # Suppress the expected side line through the midpoint; remaining ink is
    # usually the E/F label or a small midpoint mark.
    line_band = np.abs(yy.astype(np.float32) - py) <= max(1.5, 0.010 * float(min_hw))
    return int((roi & (~line_band)).sum())


def _endpoint_label_contradiction(tokens, point, expected, min_hw):
    if not tokens or point is None:
        return False
    px, py = float(point[0]), float(point[1])
    max_d = scale_px(min_hw, 0.13, floor_px=0.0)
    for t in tokens:
        letters = str(t.get("letters", "") or t.get("char", "")).upper()
        if not letters:
            continue
        if float(t.get("conf", 0.0)) < 0.50:
            continue
        cx, cy = float(t["center"][0]), float(t["center"][1])
        if math.hypot(cx - px, cy - py) > max_d:
            continue
        if expected not in letters:
            return True
    return False


def _judge_plane_29_geometric(img, lines, min_hw):
    horiz = [ln for ln in lines if horizontal_error_deg(float(ln["ang"])) <= 8.0 and float(ln["len"]) >= 0.16 * float(min_hw)]
    if len(horiz) < 2:
        return False, "Geometric fallback found fewer than two horizontal bases. "

    max_len = max(float(ln["len"]) for ln in horiz)
    bottom_pool = [ln for ln in horiz if float(ln["len"]) >= 0.45 * max_len]
    if not bottom_pool:
        return False, "Geometric fallback found no plausible bottom base. "
    bottom = max(bottom_pool, key=lambda ln: (0.5 * (float(ln["seg"][1]) + float(ln["seg"][3])), float(ln["len"])))
    by = 0.5 * (float(bottom["seg"][1]) + float(bottom["seg"][3]))
    blen = float(bottom["len"])

    top_pool = []
    for ln in horiz:
        if ln is bottom:
            continue
        ty = 0.5 * (float(ln["seg"][1]) + float(ln["seg"][3]))
        tlen = float(ln["len"])
        if ty >= by - 0.18 * float(min_hw):
            continue
        if not (0.20 * blen <= tlen <= 0.90 * blen):
            continue
        top_pool.append(ln)
    if not top_pool:
        return False, "Geometric fallback found no shorter upper base. "

    best_reason = "Geometric fallback failed to validate trapezoid details. "
    for top in sorted(top_pool, key=lambda ln: 0.5 * (float(ln["seg"][1]) + float(ln["seg"][3]))):
        A, B = _line_endpoints_lr(bottom)
        D, C = _line_endpoints_lr(top)
        ty = 0.5 * (float(top["seg"][1]) + float(top["seg"][3]))
        if by <= ty + 0.18 * float(min_hw):
            continue

        AD = _segment_item(A, D)
        BC = _segment_item(B, C)
        if angle_diff_deg(AD["ang"], BC["ang"]) <= 8.0:
            best_reason = "Geometric fallback rejected parallelogram/rectangle-like sides. "
            continue

        side_ok = (
            has_support_line(lines, A, D, min_hw=min_hw, ang_tol_deg=14.0, dist_ratio=0.10)
            and has_support_line(lines, B, C, min_hw=min_hw, ang_tol_deg=14.0, dist_ratio=0.10)
        )
        if not side_ok:
            best_reason = "Geometric fallback could not confirm both non-parallel sides. "
            continue

        diag_ok = (
            has_support_line(lines, A, C, min_hw=min_hw, ang_tol_deg=16.0, dist_ratio=0.12)
            and has_support_line(lines, B, D, min_hw=min_hw, ang_tol_deg=16.0, dist_ratio=0.12)
        )
        if not diag_ok:
            best_reason = "Geometric fallback could not confirm both diagonals. "
            continue

        expected_refs = [
            _segment_item(A, B),
            _segment_item(C, D),
            _segment_item(A, D),
            _segment_item(B, C),
            _segment_item(A, C),
            _segment_item(B, D),
        ]
        extra_lines = 0
        for ln in lines:
            if float(ln.get("len", 0.0)) < 0.22 * float(min_hw):
                continue
            if any(line_equivalent(ln, ref, min_hw=min_hw, angle_tol_deg=6.0) for ref in expected_refs):
                continue
            extra_lines += 1
        if extra_lines > 8:
            best_reason = f"Geometric fallback rejected cluttered extra line structure ({extra_lines}). "
            continue

        mid_e = (0.5 * (float(A[0]) + float(D[0])), 0.5 * (float(A[1]) + float(D[1])))
        mid_f = (0.5 * (float(B[0]) + float(C[0])), 0.5 * (float(B[1]) + float(C[1])))
        gap_y = by - ty
        median_lines = []
        for ln in horiz:
            my = 0.5 * (float(ln["seg"][1]) + float(ln["seg"][3]))
            llen = float(ln["len"])
            if my <= ty + 0.20 * gap_y or my >= by - 0.20 * gap_y:
                continue
            if 0.20 * blen <= llen <= 0.98 * blen:
                median_lines.append(ln)

        median_ok = False
        if median_lines:
            median_ok = True
        else:
            ink_e = _local_label_ink(img, mid_e, min_hw)
            ink_f = _local_label_ink(img, mid_f, min_hw)
            ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
            median_ok = ink_e >= ink_th and ink_f >= ink_th
        if not median_ok:
            best_reason = "Geometric fallback could not confirm median EF or midpoint labels. "
            continue

        tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.05)
        if (
            _endpoint_label_contradiction(tokens, A, "A", min_hw)
            or _endpoint_label_contradiction(tokens, B, "B", min_hw)
            or _endpoint_label_contradiction(tokens, C, "C", min_hw)
            or _endpoint_label_contradiction(tokens, D, "D", min_hw)
        ):
            best_reason = "Geometric fallback found contradictory vertex labels. "
            continue

        return True, ""

    return False, best_reason


def judge_plane_29(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 6:
        return [False, "Insufficient line structure for hard case 29. "]

    geom_ok, geom_reason = _judge_plane_29_geometric(img, lines, min_hw)

    tokens = extract_global_letter_tokens(img, whitelist="ABCDEFH", min_conf=0.10)
    best = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D", "E", "F", "H"], min_conf=0.10)
    missing = [ch for ch in ["A", "B", "C", "D", "E", "F", "H"] if ch not in best]
    if missing:
        if geom_ok:
            return [True, ""]
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    a_geo = (float(best["A"]["center"][0]), float(best["A"]["center"][1]))
    b_geo = (float(best["B"]["center"][0]), float(best["B"]["center"][1]))
    c_geo = (float(best["C"]["center"][0]), float(best["C"]["center"][1]))
    d_geo = (float(best["D"]["center"][0]), float(best["D"]["center"][1]))
    e_geo = (float(best["E"]["center"][0]), float(best["E"]["center"][1]))
    f_geo = (float(best["F"]["center"][0]), float(best["F"]["center"][1]))
    h_lbl = (float(best["H"]["center"][0]), float(best["H"]["center"][1]))

    markers = detect_marker_points(img)
    snap_tol = scale_px(min_hw, 0.06, floor_px=0.0)
    a_geo = _pin_to_marker(a_geo, markers, snap_tol)
    b_geo = _pin_to_marker(b_geo, markers, snap_tol)
    c_geo = _pin_to_marker(c_geo, markers, snap_tol)
    d_geo = _pin_to_marker(d_geo, markers, snap_tol)
    e_geo = _pin_to_marker(e_geo, markers, snap_tol)
    f_geo = _pin_to_marker(f_geo, markers, snap_tol)
    h_lbl = _pin_to_marker(h_lbl, markers, snap_tol)

    if not (float(a_geo[0]) < float(b_geo[0]) and float(d_geo[0]) < float(c_geo[0])):
        return [False, "Left-right ordering of trapezoid labels is incorrect (expect A-D on left, B-C on right). "]

    ab = _segment_item(a_geo, b_geo)
    cd = _segment_item(c_geo, d_geo)
    ad = _segment_item(a_geo, d_geo)
    bc = _segment_item(b_geo, c_geo)

    base_gap = scale_px(min_hw, 0.03, floor_px=0.0)
    if 0.5 * (float(a_geo[1]) + float(b_geo[1])) <= 0.5 * (float(c_geo[1]) + float(d_geo[1])) + base_gap:
        return [False, "Base AB is not below top side CD. "]

    par_err = angle_diff_deg(ab["ang"], cd["ang"])
    if par_err > 10.0:
        return [False, f"AB and CD are not parallel enough (err={par_err:.1f} deg). "]
    if angle_diff_deg(ad["ang"], bc["ang"]) <= 9.0:
        return [False, "Both opposite side pairs are parallel (parallelogram-like). "]

    if float(ab["len"]) <= float(cd["len"]) + scale_px(min_hw, 0.05, floor_px=0.0):
        return [False, f"Bottom base AB is not longer than top base CD (AB={ab['len']:.1f}, CD={cd['len']:.1f}). "]

    if not has_support_line(lines, a_geo, c_geo, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08):
        return [False, "Missing diagonal AC support line. "]
    if not has_support_line(lines, b_geo, d_geo, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08):
        return [False, "Missing diagonal BD support line. "]

    l_ac = segment_to_abc((a_geo[0], a_geo[1], c_geo[0], c_geo[1]))
    l_bd = segment_to_abc((b_geo[0], b_geo[1], d_geo[0], d_geo[1]))
    if l_ac is None or l_bd is None:
        return [False, "Failed to build diagonal lines. "]
    h_geo = line_intersection_from_abc(l_ac, l_bd)
    if h_geo is None:
        return [False, "Failed to intersect AC and BD. "]
    if _dist(h_lbl, h_geo) > scale_px(min_hw, 0.09, floor_px=0.0):
        return [False, "H label is not near the diagonal intersection. "]

    line_tol = scale_px(min_hw, 0.06, floor_px=0.0)
    if point_line_distance(e_geo, ad["abc"]) > line_tol:
        return [False, "E is not on leg AD. "]
    if point_line_distance(f_geo, bc["abc"]) > line_tol:
        return [False, "F is not on leg BC. "]

    rel_e = abs(_dist(e_geo, a_geo) - _dist(e_geo, d_geo)) / max(1e-6, 0.5 * (_dist(e_geo, a_geo) + _dist(e_geo, d_geo)))
    rel_f = abs(_dist(f_geo, b_geo) - _dist(f_geo, c_geo)) / max(1e-6, 0.5 * (_dist(f_geo, b_geo) + _dist(f_geo, c_geo)))
    if max(rel_e, rel_f) > 0.20:
        return [False, f"E/F are not close enough to leg midpoints (E={rel_e:.3f}, F={rel_f:.3f}). "]

    if not has_support_line(lines, e_geo, f_geo, min_hw=min_hw, ang_tol_deg=10.0, dist_ratio=0.08):
        if geom_ok:
            return [True, ""]
        return [False, "Median segment EF support line is missing. "]
    ef_ang = segment_angle_deg((e_geo[0], e_geo[1], f_geo[0], f_geo[1]))
    if angle_diff_deg(ef_ang, ab["ang"]) > 11.0:
        if geom_ok:
            return [True, ""]
        return [False, "Median EF is not parallel to bases AB/CD. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_29,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
