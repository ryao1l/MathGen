import argparse

PID = 25
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _pin_to_marker(p, markers, tol):
    if not markers:
        return (float(p[0]), float(p[1]))
    q = nearest_point(markers, p)
    if q is None:
        return (float(p[0]), float(p[1]))
    if _dist(q, p) <= float(tol):
        return (float(q[0]), float(q[1]))
    return (float(p[0]), float(p[1]))


def _cyclic_label_sequence(points):
    cx = sum(float(pt[0]) for pt in points.values()) / float(len(points))
    cy = sum(float(pt[1]) for pt in points.values()) / float(len(points))
    arr = []
    for ch, pt in points.items():
        ang = math.atan2(float(pt[1]) - cy, float(pt[0]) - cx)
        arr.append((ang, ch))
    arr.sort(key=lambda t: t[0])
    seq = [ch for _, ch in arr]
    return seq, (cx, cy)


def _rotate_to_start(seq, ch):
    if ch not in seq:
        return None
    i = seq.index(ch)
    return seq[i:] + seq[:i]


def _find_support_line_strict(
    lines,
    p1,
    p2,
    min_hw,
    ang_tol_deg=11.0,
    dist_ratio=0.07,
    dist_floor_px=0.0,
    t_min=-0.25,
    t_max=1.25,
):
    if not lines or p1 is None or p2 is None:
        return None
    target_ang = segment_angle_deg((float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])))
    dist_tol = scale_px(min_hw, dist_ratio, floor_px=dist_floor_px)
    best = None
    best_key = None
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if "abc" not in ln or "ang" not in ln or "len" not in ln or "seg" not in ln:
            continue
        ad = angle_diff_deg(target_ang, ln["ang"])
        if ad > float(ang_tol_deg):
            continue
        d1 = point_line_distance(p1, ln["abc"])
        d2 = point_line_distance(p2, ln["abc"])
        if d1 > dist_tol or d2 > dist_tol:
            continue
        t1 = segment_projection_t(ln["seg"], p1)
        t2 = segment_projection_t(ln["seg"], p2)
        if not (float(t_min) <= float(t1) <= float(t_max)):
            continue
        if not (float(t_min) <= float(t2) <= float(t_max)):
            continue
        key = (float(d1 + d2 + 0.25 * ad), -float(ln["len"]))
        if best_key is None or key < best_key:
            best_key = key
            best = ln
    return best


def _find_required_line(lines, p1, p2, min_hw):
    ln = _find_support_line_strict(
        lines,
        p1,
        p2,
        min_hw=min_hw,
        ang_tol_deg=11.0,
        dist_ratio=0.07,
        t_min=-0.25,
        t_max=1.25,
    )
    if ln is not None:
        return ln
    return find_support_line(
        lines,
        p1,
        p2,
        min_hw=min_hw,
        ang_tol_deg=11.0,
        dist_ratio=0.07,
    )


def judge_plane_25(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 8:
        return [False, "Insufficient line structure for hard case 25. "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDEFH", min_conf=0.10)
    best = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D", "E", "F", "H"], min_conf=0.10)
    missing = [ch for ch in ["A", "B", "C", "D", "E", "F", "H"] if ch not in best]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    markers = detect_marker_points(img)
    pin_tol = scale_px(min_hw, 0.10, floor_px=0.0)

    pts = {}
    for ch in ["A", "B", "C", "D", "E", "F"]:
        pts[ch] = _pin_to_marker(best[ch]["center"], markers, pin_tol)
    h_lbl = _pin_to_marker(best["H"]["center"], markers, pin_tol)

    y_min_label = min(pts.keys(), key=lambda k: float(pts[k][1]))
    if y_min_label != "A":
        return [False, "A is not at the top vertex of the hexagon. "]

    seq_ccw, center = _cyclic_label_sequence(pts)
    seq1 = _rotate_to_start(seq_ccw, "A")
    seq2 = _rotate_to_start(list(reversed(seq_ccw)), "A")
    ok_order = (seq1 == ["A", "B", "C", "D", "E", "F"]) or (seq2 == ["A", "B", "C", "D", "E", "F"])
    if not ok_order:
        return [False, f"Hexagon labels are not in A-B-C-D-E-F cyclic order (got {seq1}). "]

    radii = [_dist(pts[ch], center) for ch in ["A", "B", "C", "D", "E", "F"]]
    r_mean = sum(radii) / 6.0
    r_rel = max(abs(r - r_mean) for r in radii) / max(1e-6, r_mean)
    if r_rel > 0.18:
        return [False, f"Hexagon is not regular enough by radial symmetry (rel={r_rel:.3f}). "]

    order = ["A", "B", "C", "D", "E", "F"]
    side_lens = [_dist(pts[order[i]], pts[order[(i + 1) % 6]]) for i in range(6)]
    s_mean = sum(side_lens) / 6.0
    s_rel = max(abs(s - s_mean) for s in side_lens) / max(1e-6, s_mean)
    if s_rel > 0.24:
        return [False, f"Hexagon side lengths are not regular enough (rel={s_rel:.3f}). "]

    for i in range(6):
        p1 = pts[order[i]]
        p2 = pts[order[(i + 1) % 6]]
        if find_support_line(lines, p1, p2, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.07) is None:
            return [False, f"Missing boundary edge {order[i]}{order[(i + 1) % 6]}. "]

    a, b, c, d, e, f = pts["A"], pts["B"], pts["C"], pts["D"], pts["E"], pts["F"]
    if not has_support_line(lines, a, d, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing main diagonal AD. "]
    if not has_support_line(lines, b, e, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing main diagonal BE. "]
    if not has_support_line(lines, c, f, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing main diagonal CF. "]

    l_ad = segment_to_abc((a[0], a[1], d[0], d[1]))
    l_be = segment_to_abc((b[0], b[1], e[0], e[1]))
    l_cf = segment_to_abc((c[0], c[1], f[0], f[1]))
    if l_ad is None or l_be is None or l_cf is None:
        return [False, "Failed to build main diagonal lines. "]

    h_geo = line_intersection_from_abc(l_ad, l_be)
    if h_geo is None:
        return [False, "Failed to intersect AD and BE. "]
    if point_line_distance(h_geo, l_cf) > scale_px(min_hw, 0.05, floor_px=0.0):
        return [False, "AD/BE/CF are not concurrent enough. "]
    if _dist(h_lbl, h_geo) > scale_px(min_hw, 0.14, floor_px=0.0):
        return [False, "H label is not near the diagonal intersection center. "]

    if not has_support_line(lines, a, c, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing triangle edge AC. "]
    if not has_support_line(lines, c, e, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing triangle edge CE. "]
    if not has_support_line(lines, e, a, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing triangle edge EA. "]

    required_pairs = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
        ("F", "A"),
        ("A", "D"),
        ("B", "E"),
        ("C", "F"),
        ("A", "C"),
        ("C", "E"),
        ("E", "A"),
    ]

    required_refs = []
    for u, v in required_pairs:
        ln = _find_required_line(lines, pts[u], pts[v], min_hw=min_hw)
        if ln is None:
            return [False, f"Missing required segment {u}{v}. "]
        if not any(line_equivalent(ln, ref, min_hw) for ref in required_refs):
            required_refs.append(ln)

    extra_th = scale_px(min_hw, 0.22, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in required_refs):
            continue
        extras.append(ln)
    if extras:
        return [False, f"Detected extra dominant line(s) outside target hexagon structure: {len(extras)}. "]

    violated, info = has_excess_outside_ink(
        img=img,
        allowed_lines=required_refs,
        anchor_points=[pts[ch] for ch in ["A", "B", "C", "D", "E", "F"]] + [h_geo],
        max_outside_ratio=0.10,
        max_outside_px_ratio=0.00008,
        max_outside_px_floor=0,
    )
    if violated:
        return [
            False,
            (
                "Detected extra drawing content outside target structure "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_25,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
