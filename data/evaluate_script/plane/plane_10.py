import argparse
import math

PID = 81
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
        return None
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


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
        if r > 0.55 * min_hw:
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


def _dedup_candidate_points(items, tol):
    if not items:
        return []
    ordered = sorted(items, key=lambda it: float(it["cost"]))
    keep = []
    for it in ordered:
        p = it["point"]
        if any(_dist(p, jt["point"]) <= float(tol) for jt in keep):
            continue
        keep.append(it)
    return keep


def _build_p_candidates(p_tok, marker_points, line_item, circle, min_hw):
    if p_tok is None or line_item is None or circle is None:
        return []
    if "abc" not in line_item or "seg" not in line_item:
        return []

    cx, cy, r = [float(v) for v in circle]
    center = (cx, cy)
    p_hint = (float(p_tok["center"][0]), float(p_tok["center"][1]))

    attach_p = max(scale_px(min_hw, 0.16, floor_px=0.0), 0.35 * r)
    line_attach_tol = max(scale_px(min_hw, 0.04, floor_px=0.0), 0.12 * r)
    outside_req = max(scale_px(min_hw, 0.010, floor_px=0.0), 0.03 * r)

    items = []

    p_proj = project_point_to_line(p_hint, line_item["abc"])
    if p_proj is not None:
        d_line = point_line_distance(p_proj, line_item["abc"])
        if d_line <= line_attach_tol:
            margin = _dist(p_proj, center) - r
            if margin >= -outside_req:
                cost = (
                    1.2 * token_edge_distance_to_point(p_tok, p_proj)
                    + 0.45 * _dist(p_proj, p_hint)
                    - 0.30 * margin
                )
                items.append({"point": (float(p_proj[0]), float(p_proj[1])), "cost": float(cost), "kind": "projection"})

    x1, y1, x2, y2 = [float(v) for v in line_item["seg"]]
    for p in [(x1, y1), (x2, y2)]:
        if _dist(p, p_hint) > 1.8 * attach_p:
            continue
        margin = _dist(p, center) - r
        if margin < -outside_req:
            continue
        cost = (
            1.0 * token_edge_distance_to_point(p_tok, p)
            + 0.25 * _dist(p, p_hint)
            - 0.35 * margin
            - 2.0
        )
        items.append({"point": (float(p[0]), float(p[1])), "cost": float(cost), "kind": "endpoint"})

    for p in marker_points:
        if _dist(p, p_hint) > attach_p:
            continue
        if point_line_distance(p, line_item["abc"]) > line_attach_tol:
            continue
        margin = _dist(p, center) - r
        if margin < -outside_req:
            continue
        cost = (
            1.0 * token_edge_distance_to_point(p_tok, p)
            + 0.35 * _dist(p, p_hint)
            - 0.30 * margin
            + 0.8
        )
        items.append({"point": (float(p[0]), float(p[1])), "cost": float(cost), "kind": "marker"})

    tol = scale_px(min_hw, 0.012, floor_px=2.5)
    deduped = _dedup_candidate_points(items, tol=tol)
    return deduped[:8]


def _segment_presence(img, lines, p1, p2, min_hw, ratio_th=0.11, trim_ratio=0.03, dist_ratio=0.07):
    ok_seg, ratio = has_segment_between_points(
        img,
        p1,
        p2,
        ratio_th=float(ratio_th),
        thickness=2,
        trim_ratio=float(trim_ratio),
    )
    line_ref = find_support_line(
        lines,
        p1,
        p2,
        min_hw=min_hw,
        ang_tol_deg=11.0,
        dist_ratio=float(dist_ratio),
        dist_floor_px=0.0,
    )
    ratio = float(ratio)
    present = bool(ratio >= float(ratio_th) or (line_ref is not None and ratio >= max(0.08, 0.75 * float(ratio_th))))
    return present, ratio, line_ref


def _assign_ab_labels(a_tok, b_tok, q1, q2):
    if a_tok is None and b_tok is None:
        qs = sorted([(float(q1[0]), float(q1[1])), (float(q2[0]), float(q2[1]))], key=lambda p: (p[0], p[1]))
        return {
            "A": qs[0],
            "B": qs[1],
            "d_a": 0.0,
            "d_b": 0.0,
            "label_cost": 0.0,
            "mode": "none",
        }
    if a_tok is None:
        d_b1 = token_edge_distance_to_point(b_tok, q1)
        d_b2 = token_edge_distance_to_point(b_tok, q2)
        if d_b1 <= d_b2:
            return {
                "A": (float(q2[0]), float(q2[1])),
                "B": (float(q1[0]), float(q1[1])),
                "d_a": 0.0,
                "d_b": float(d_b1),
                "label_cost": float(d_b1),
                "mode": "B_only",
            }
        return {
            "A": (float(q1[0]), float(q1[1])),
            "B": (float(q2[0]), float(q2[1])),
            "d_a": 0.0,
            "d_b": float(d_b2),
            "label_cost": float(d_b2),
            "mode": "B_only",
        }
    if b_tok is None:
        d_a1 = token_edge_distance_to_point(a_tok, q1)
        d_a2 = token_edge_distance_to_point(a_tok, q2)
        if d_a1 <= d_a2:
            return {
                "A": (float(q1[0]), float(q1[1])),
                "B": (float(q2[0]), float(q2[1])),
                "d_a": float(d_a1),
                "d_b": 0.0,
                "label_cost": float(d_a1),
                "mode": "A_only",
            }
        return {
            "A": (float(q2[0]), float(q2[1])),
            "B": (float(q1[0]), float(q1[1])),
            "d_a": float(d_a2),
            "d_b": 0.0,
            "label_cost": float(d_a2),
            "mode": "A_only",
        }

    d_a1 = token_edge_distance_to_point(a_tok, q1)
    d_b2 = token_edge_distance_to_point(b_tok, q2)
    d_a2 = token_edge_distance_to_point(a_tok, q2)
    d_b1 = token_edge_distance_to_point(b_tok, q1)

    if float(d_a1 + d_b2) <= float(d_a2 + d_b1):
        return {
            "A": (float(q1[0]), float(q1[1])),
            "B": (float(q2[0]), float(q2[1])),
            "d_a": float(d_a1),
            "d_b": float(d_b2),
            "label_cost": float(d_a1 + d_b2),
            "mode": "both",
        }
    return {
        "A": (float(q2[0]), float(q2[1])),
        "B": (float(q1[0]), float(q1[1])),
        "d_a": float(d_a2),
        "d_b": float(d_b1),
        "label_cost": float(d_a2 + d_b1),
        "mode": "both",
    }


def _select_secant_solution(img, circle, lines, raw_lines, best_tokens, min_hw):
    cx, cy, r = [float(v) for v in circle]
    center = (cx, cy)
    p_tok = best_tokens["P"]
    a_tok = best_tokens.get("A")
    b_tok = best_tokens.get("B")
    markers = [(float(p[0]), float(p[1])) for p in detect_marker_points(img)]

    line_len_min = max(scale_px(min_hw, 0.09, floor_px=0.0), 0.22 * r)
    p_line_tol = max(scale_px(min_hw, 0.12, floor_px=0.0), 0.26 * r)
    outside_req = max(scale_px(min_hw, 0.012, floor_px=0.0), 0.04 * r)
    on_tol = max(scale_px(min_hw, 0.030, floor_px=0.0), 0.08 * r)
    p_label_max = max(scale_px(min_hw, 0.13, floor_px=0.0), 0.28 * r)
    ab_label_max = max(scale_px(min_hw, 0.12, floor_px=0.0), 0.24 * r)
    pair_sep_min = max(scale_px(min_hw, 0.05, floor_px=0.0), 0.14 * r)

    best = None
    best_score = None
    best_fail_stage = -1
    best_fail_reason = "Failed to detect a valid secant through P intersecting the circle at A and B. "

    def _record_fail(stage, reason):
        nonlocal best_fail_stage, best_fail_reason
        if int(stage) > best_fail_stage:
            best_fail_stage = int(stage)
            best_fail_reason = str(reason)

    for ln in lines:
        if float(ln.get("len", 0.0)) < line_len_min:
            continue
        if token_edge_distance_to_line(p_tok, ln["abc"]) > p_line_tol:
            _record_fail(1, "Failed to find a long secant candidate passing near label P. ")
            continue

        inters = circle_line_intersections(circle, ln["abc"])
        if len(inters) < 2:
            _record_fail(2, "No line through P intersects the circle at two points. ")
            continue
        q1 = (float(inters[0][0]), float(inters[0][1]))
        q2 = (float(inters[1][0]), float(inters[1][1]))
        if _dist(q1, q2) < pair_sep_min:
            _record_fail(2, "Secant-circle intersections are not sufficiently distinct. ")
            continue

        ab_seg = (q1[0], q1[1], q2[0], q2[1])
        support = collinear_support_stats_on_segment(
            lines=raw_lines,
            reference_line=ln,
            base_seg=ab_seg,
            min_hw=min_hw,
            angle_tol_deg=3.0,
            offset_ratio=0.025,
            offset_floor_px=8.0,
            reach_left_t=0.10,
            reach_right_t=0.90,
        )
        ok_ab, ratio_ab = has_segment_between_points(img, q1, q2, ratio_th=0.12, thickness=2, trim_ratio=0.03)
        if not ok_ab and support["coverage"] < 0.42:
            _record_fail(3, f"Missing secant support across A-B (ratio={ratio_ab:.2f}, cov={support['coverage']:.2f}). ")
            continue

        p_candidates = _build_p_candidates(
            p_tok=p_tok,
            marker_points=markers,
            line_item=ln,
            circle=circle,
            min_hw=min_hw,
        )
        if not p_candidates:
            _record_fail(4, "Failed to build point-P candidates on secant line. ")
            continue

        labels = _assign_ab_labels(a_tok, b_tok, q1, q2)
        if labels is None:
            _record_fail(5, "Failed to assign A/B labels to secant-circle intersections. ")
            continue
        if labels["mode"] == "both" and max(labels["d_a"], labels["d_b"]) > ab_label_max:
            _record_fail(5, "Labels A/B are too far from secant-circle intersections. ")
            continue
        if labels["mode"] == "A_only" and labels["d_a"] > 1.15 * ab_label_max:
            _record_fail(5, "Label A is too far from secant-circle intersections. ")
            continue
        if labels["mode"] == "B_only" and labels["d_b"] > 1.15 * ab_label_max:
            _record_fail(5, "Label B is too far from secant-circle intersections. ")
            continue

        for pit in p_candidates:
            p_geo = pit["point"]
            p_label_dist = token_edge_distance_to_point(p_tok, p_geo)
            if p_label_dist > p_label_max:
                _record_fail(6, "Point P candidate is too far from label P. ")
                continue

            outside_margin = _dist(p_geo, center) - r
            if outside_margin < outside_req:
                _record_fail(7, f"Point P is not outside the circle (margin={outside_margin:.1f}). ")
                continue

            t_p = segment_projection_t(ab_seg, p_geo)
            if -0.05 <= float(t_p) <= 1.05:
                _record_fail(8, "Point P lies between A and B, not on secant extension. ")
                continue

            ok_pa, ratio_pa, _ = _segment_presence(
                img,
                lines,
                p_geo,
                labels["A"],
                min_hw=min_hw,
                ratio_th=0.10,
                trim_ratio=0.03,
                dist_ratio=0.07,
            )
            ok_pb, ratio_pb, _ = _segment_presence(
                img,
                lines,
                p_geo,
                labels["B"],
                min_hw=min_hw,
                ratio_th=0.10,
                trim_ratio=0.03,
                dist_ratio=0.07,
            )
            if not (ok_pa and ok_pb):
                _record_fail(
                    9,
                    f"Missing secant support from P to A/B (PA={ratio_pa:.2f}, PB={ratio_pb:.2f}). ",
                )
                continue

            a_rad_err = abs(_dist(labels["A"], center) - r)
            b_rad_err = abs(_dist(labels["B"], center) - r)
            if a_rad_err > on_tol or b_rad_err > on_tol:
                _record_fail(10, "A/B are not stable on the circle. ")
                continue

            score = (
                2.2 * float(pit["cost"])
                + 1.4 * float(labels["label_cost"])
                + 1.6 * float(p_label_dist)
                + 3.2 * float(a_rad_err + b_rad_err)
                - 0.80 * float(ratio_ab)
                - 0.75 * float(ratio_pa + ratio_pb)
                - 0.08 * float(ln["len"])
                - 0.20 * float(outside_margin)
                + 2.2 * abs(min(0.0, float(t_p), float(1.0 - t_p)))
            )
            if labels["mode"] == "none":
                score += 6.0
            elif labels["mode"] in {"A_only", "B_only"}:
                score += 2.0
            if best_score is None or score < best_score:
                best_score = float(score)
                best = {
                    "line": ln,
                    "P": (float(p_geo[0]), float(p_geo[1])),
                    "A": (float(labels["A"][0]), float(labels["A"][1])),
                    "B": (float(labels["B"][0]), float(labels["B"][1])),
                    "ratio_ab": float(ratio_ab),
                    "ratio_pa": float(ratio_pa),
                    "ratio_pb": float(ratio_pb),
                    "label_mode": labels["mode"],
                }

    return best, best_fail_reason


def judge_plane_81(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))

    circle = _select_main_circle(img)
    if circle is None:
        return [False, "Failed to detect a valid main circle. "]
    cx, cy, r = [float(v) for v in circle]
    center = (cx, cy)

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
        if _dist((ex, ey), center) <= scale_px(min_hw, 0.06, floor_px=0.0):
            continue
        return [False, "Detected extra prominent circle structure. "]

    tokens = extract_global_letter_tokens(img, whitelist="OPAB", min_conf=0.08)
    best = pick_best_tokens_by_char(tokens, ["O", "P", "A", "B"], min_conf=0.08)
    missing = [ch for ch in ["O", "P"] if ch not in best]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    o_lbl = select_token_near_point(
        tokens,
        expected_char="O",
        point=center,
        max_dist=max(scale_px(min_hw, 0.14, floor_px=0.0), 0.26 * r),
    )
    if o_lbl is None:
        return [False, "Failed to detect label O near the circle center. "]

    lines = detect_line_segments(img, min_len_ratio=0.07)
    raw_lines = detect_line_segments_raw(img, min_len_ratio=0.06)
    if not lines:
        return [False, "Failed to detect line structure for secant PAB. "]
    if not raw_lines:
        raw_lines = lines

    picked, fail_reason = _select_secant_solution(
        img=img,
        circle=circle,
        lines=lines,
        raw_lines=raw_lines,
        best_tokens=best,
        min_hw=min_hw,
    )
    if picked is None:
        return [False, fail_reason]

    p_geo = picked["P"]
    a_geo = picked["A"]
    b_geo = picked["B"]
    secant_line = picked["line"]

    outside_margin = _dist(p_geo, center) - r
    outside_req = max(scale_px(min_hw, 0.012, floor_px=0.0), 0.04 * r)
    if outside_margin < outside_req:
        return [False, f"Point P is not outside the circle (margin={outside_margin:.1f}). "]

    on_tol = max(scale_px(min_hw, 0.030, floor_px=0.0), 0.08 * r)
    a_rad_err = abs(_dist(a_geo, center) - r)
    b_rad_err = abs(_dist(b_geo, center) - r)
    if a_rad_err > on_tol:
        return [False, f"Point A is not on the circle (radial_err={a_rad_err:.1f}). "]
    if b_rad_err > on_tol:
        return [False, f"Point B is not on the circle (radial_err={b_rad_err:.1f}). "]
    if _dist(a_geo, b_geo) < max(scale_px(min_hw, 0.05, floor_px=0.0), 0.14 * r):
        return [False, "Points A and B on the circle are not sufficiently distinct. "]

    p_lbl = select_token_near_point(
        tokens,
        expected_char="P",
        point=p_geo,
        max_dist=max(scale_px(min_hw, 0.13, floor_px=0.0), 0.28 * r),
    )
    if p_lbl is None:
        return [False, "Failed to detect label P near the outside point. "]

    ab_seg = (a_geo[0], a_geo[1], b_geo[0], b_geo[1])
    t_p = segment_projection_t(ab_seg, p_geo)
    if -0.05 <= float(t_p) <= 1.05:
        return [False, "Point P lies between A and B, not on secant extension. "]

    secant_ref = secant_line if secant_line is not None else _segment_item(a_geo, b_geo)
    if secant_ref is None:
        return [False, "Failed to build secant line reference. "]

    major_th = max(scale_px(min_hw, 0.24, floor_px=0.0), 0.52 * float(secant_ref["len"]))
    extras = 0
    for ln in lines:
        if float(ln.get("len", 0.0)) < major_th:
            continue
        if line_equivalent(ln, secant_ref, min_hw=min_hw, angle_tol_deg=5.0):
            continue
        extras += 1
    if extras > 0:
        return [False, f"Detected extra dominant line(s) outside secant PAB: {extras}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_81,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
