import argparse

PID = 24
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


def judge_plane_24(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 8:
        return [False, "Insufficient line structure for hard case 24. "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDEFGHP", min_conf=0.10)
    best = pick_best_tokens_by_char(tokens, list("ABCDEFGHP"), min_conf=0.10)
    missing = [ch for ch in "ABCDEFGHP" if ch not in best]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    raw_points = {ch: best[ch]["center"] for ch in "ABCDEFGHP"}
    anchors = collect_intersection_anchors(
        lines=lines,
        img_shape=img.shape,
        min_angle_sep_deg=10.0,
        margin_ratio=0.12,
        point_tol_ratio=0.03,
        support_dist_ratio=0.02,
        support_t=(-0.50, 1.35),
    )
    snapped = snap_points_to_anchors(raw_points, anchors, scale_px(min_hw, 0.14, floor_px=0.0))

    a = snapped["A"]
    b = snapped["B"]
    c = snapped["C"]
    d = snapped["D"]
    e = snapped["E"]
    f = snapped["F"]
    g = snapped["G"]
    h = snapped["H"]
    p_lbl = snapped["P"]

    if not (float(a[0]) < min(float(b[0]), float(d[0])) and float(c[0]) > max(float(b[0]), float(d[0]))):
        return [False, "A/C are not at left/right vertices as required. "]
    if not (float(b[1]) < min(float(a[1]), float(c[1])) and float(d[1]) > max(float(a[1]), float(c[1]))):
        return [False, "B/D are not at top/bottom vertices as required. "]

    ab = _segment_item(a, b)
    bc = _segment_item(b, c)
    cd = _segment_item(c, d)
    da = _segment_item(d, a)

    sides = [ab["len"], bc["len"], cd["len"], da["len"]]
    s_mean = sum(sides) / 4.0
    rel_side = max(abs(s - s_mean) for s in sides) / max(1e-6, s_mean)
    if rel_side > 0.22:
        return [False, f"Outer quadrilateral is not rhombus-like (rel_side={rel_side:.3f}). "]

    if not has_support_line(lines, a, c, min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing diagonal AC support line. "]
    if not has_support_line(lines, b, d, min_hw, ang_tol_deg=11.0, dist_ratio=0.07):
        return [False, "Missing diagonal BD support line. "]

    l_ac = segment_to_abc((a[0], a[1], c[0], c[1]))
    l_bd = segment_to_abc((b[0], b[1], d[0], d[1]))
    if l_ac is None or l_bd is None:
        return [False, "Failed to build diagonal lines. "]
    p_geo = line_intersection_from_abc(l_ac, l_bd)
    if p_geo is None:
        return [False, "Failed to intersect diagonals AC and BD. "]
    if _dist(p_lbl, p_geo) > scale_px(min_hw, 0.12, floor_px=0.0):
        return [False, "P label is not near diagonal intersection. "]

    line_tol = scale_px(min_hw, 0.06, floor_px=0.0)
    if point_line_distance(e, ab["abc"]) > line_tol:
        return [False, "E is not on side AB. "]
    if point_line_distance(f, bc["abc"]) > line_tol:
        return [False, "F is not on side BC. "]
    if point_line_distance(g, cd["abc"]) > line_tol:
        return [False, "G is not on side CD. "]
    if point_line_distance(h, da["abc"]) > line_tol:
        return [False, "H is not on side DA. "]

    rel_e = abs(_dist(e, a) - _dist(e, b)) / max(1e-6, 0.5 * (_dist(e, a) + _dist(e, b)))
    rel_f = abs(_dist(f, b) - _dist(f, c)) / max(1e-6, 0.5 * (_dist(f, b) + _dist(f, c)))
    rel_g = abs(_dist(g, c) - _dist(g, d)) / max(1e-6, 0.5 * (_dist(g, c) + _dist(g, d)))
    rel_h = abs(_dist(h, d) - _dist(h, a)) / max(1e-6, 0.5 * (_dist(h, d) + _dist(h, a)))
    if max(rel_e, rel_f, rel_g, rel_h) > 0.30:
        return [
            False,
            (
                "At least one midpoint label is too far from side midpoint "
                f"(E={rel_e:.3f}, F={rel_f:.3f}, G={rel_g:.3f}, H={rel_h:.3f}). "
            ),
        ]

    if not has_support_line(lines, e, f, min_hw, ang_tol_deg=10.0, dist_ratio=0.07):
        return [False, "Missing inner edge EF. "]
    if not has_support_line(lines, f, g, min_hw, ang_tol_deg=10.0, dist_ratio=0.07):
        return [False, "Missing inner edge FG. "]
    if not has_support_line(lines, g, h, min_hw, ang_tol_deg=10.0, dist_ratio=0.07):
        return [False, "Missing inner edge GH. "]
    if not has_support_line(lines, h, e, min_hw, ang_tol_deg=10.0, dist_ratio=0.07):
        return [False, "Missing inner edge HE. "]

    e_ef = _segment_item(e, f)
    e_fg = _segment_item(f, g)
    e_gh = _segment_item(g, h)
    e_he = _segment_item(h, e)

    if angle_diff_deg(e_ef["ang"], e_gh["ang"]) > 10.0 or angle_diff_deg(e_fg["ang"], e_he["ang"]) > 10.0:
        return [False, "EFGH is not rectangle-like (opposite sides not parallel). "]

    perp = abs(angle_diff_deg(e_ef["ang"], e_fg["ang"]) - 90.0)
    if perp > 16.0:
        return [False, f"EFGH is not rectangle-like (adjacent angle error={perp:.1f}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_24,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
