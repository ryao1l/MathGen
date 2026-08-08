import argparse
import math

PID = 65
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


def _point_to_segment_distance(point, p1, p2):
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    vx, vy = x2 - x1, y2 - y1
    ll = vx * vx + vy * vy
    if ll <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * vx + (py - y1) * vy) / ll
    t = min(1.0, max(0.0, float(t)))
    qx = x1 + t * vx
    qy = y1 + t * vy
    return math.hypot(px - qx, py - qy)


def _point_inside_convex_soft(point, vertices, tol_dist):
    if vertices is None or len(vertices) < 3:
        return False
    px, py = float(point[0]), float(point[1])
    area2 = 0.0
    for i in range(len(vertices)):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % len(vertices)][0]), float(vertices[(i + 1) % len(vertices)][1])
        area2 += x1 * y2 - x2 * y1
    orient = 1.0 if area2 >= 0.0 else -1.0

    for i in range(len(vertices)):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % len(vertices)][0]), float(vertices[(i + 1) % len(vertices)][1])
        ex, ey = x2 - x1, y2 - y1
        cross = ex * (py - y1) - ey * (px - x1)
        edge_len = math.hypot(ex, ey)
        if edge_len <= 1e-6:
            return False
        if orient * cross < -float(edge_len) * float(tol_dist):
            return False
    return True


def _point_strictly_inside_convex(point, vertices, edge_clearance):
    if vertices is None or len(vertices) < 3:
        return False, 0.0
    dmin = min(
        _point_to_segment_distance(point, vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    )
    if dmin <= float(edge_clearance):
        return False, float(dmin)
    inside = _point_inside_convex_soft(
        point=point,
        vertices=vertices,
        tol_dist=max(1e-6, 0.35 * float(edge_clearance)),
    )
    return bool(inside), float(dmin)


def _square_metrics(vertices):
    if not isinstance(vertices, list) or len(vertices) != 4:
        return None

    edges = [_dist(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    if any(float(e) <= 1e-6 for e in edges):
        return None

    cos_vals = []
    for i in range(4):
        a = vertices[i]
        b = vertices[(i + 1) % 4]
        c = vertices[(i + 2) % 4]
        v1 = (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
        v2 = (float(c[0]) - float(b[0]), float(c[1]) - float(b[1]))
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 <= 1e-6 or n2 <= 1e-6:
            return None
        cos_vals.append(abs((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))

    mean_edge = sum(edges) / 4.0
    edge_rel = (max(edges) - min(edges)) / max(1e-6, mean_edge)
    area = float(_polygon_area(vertices))
    diag = max(_dist(vertices[0], vertices[2]), _dist(vertices[1], vertices[3]))
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "mean_edge": float(mean_edge),
        "edge_rel": float(edge_rel),
        "min_edge": float(min(edges)),
        "area": area,
        "diag": float(diag),
    }


def _extract_square_candidate(img, lines, min_hw):
    h, w = img.shape[:2]
    area_norm = max(1.0, float(h) * float(w))
    configs = [
        (0.18, 10, 0.010, 12.0),
        (0.15, 12, 0.008, 10.0),
        (0.12, 14, 0.006, 8.0),
    ]

    base_lines = [ln for ln in lines if isinstance(ln, dict)] if isinstance(lines, list) else []
    if len(base_lines) < 4:
        return None

    pools = [base_lines]
    drop_candidates = list(range(min(len(base_lines), 8)))
    for i in drop_candidates:
        pool_1 = [ln for j, ln in enumerate(base_lines) if j != i]
        if len(pool_1) >= 4:
            pools.append(pool_1)
    for i in range(len(drop_candidates)):
        for j in range(i + 1, len(drop_candidates)):
            ii = drop_candidates[i]
            jj = drop_candidates[j]
            pool_2 = [ln for k, ln in enumerate(base_lines) if k != ii and k != jj]
            if len(pool_2) >= 4:
                pools.append(pool_2)

    best = None
    best_key = None
    for min_len_ratio, top_k, min_area_ratio, min_angle_sep in configs:
        for pool in pools:
            if len(pool) < 4:
                continue
            quad = extract_polygon_from_lines(
                lines=pool,
                img_shape=img.shape,
                sides=4,
                min_len_ratio=min_len_ratio,
                top_k=min(int(top_k), len(pool)),
                min_angle_sep_deg=min_angle_sep,
                margin_ratio=0.18,
                point_tol_ratio=0.045,
                min_area_ratio=min_area_ratio,
                support_t=(-0.55, 1.40),
            )
            if quad is None:
                continue

            vertices = quad["vertices"]
            metrics = _square_metrics(vertices)
            if metrics is None:
                continue
            if max(metrics["cos_vals"]) > 0.26:
                continue
            if metrics["edge_rel"] > 0.22:
                continue
            if metrics["min_edge"] < scale_px(min_hw, 0.10, floor_px=0.0):
                continue

            side_ratios = []
            side_ok = True
            for i in range(4):
                p = vertices[i]
                q = vertices[(i + 1) % 4]
                ok_side, ratio = has_segment_between_points(img, p, q, ratio_th=0.10, thickness=2, trim_ratio=0.02)
                if not ok_side:
                    side_line = find_support_line(base_lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08)
                    if side_line is None:
                        side_ok = False
                        break
                side_ratios.append(float(ratio))
            if not side_ok:
                continue

            key = (
                float(metrics["area"]) / area_norm,
                min(side_ratios) if side_ratios else 0.0,
                -max(metrics["cos_vals"]),
                -float(metrics["edge_rel"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "vertices": vertices,
                    "metrics": metrics,
                    "lines": quad["lines"],
                }
    return best


def _locate_p_geo(vertices, p_token, markers, anchors, min_hw):
    p_center = (float(p_token["center"][0]), float(p_token["center"][1]))
    diag = max(_dist(vertices[0], vertices[2]), _dist(vertices[1], vertices[3]))
    edge_clear = scale_px(min_hw, 0.010, floor_px=0.0)
    vertex_excl = max(scale_px(min_hw, 0.025, floor_px=0.0), 0.06 * diag)
    marker_attach = max(scale_px(min_hw, 0.10, floor_px=0.0), 0.12 * diag)
    anchor_attach = max(scale_px(min_hw, 0.14, floor_px=0.0), 0.16 * diag)

    cands = []
    for m in markers if isinstance(markers, list) else []:
        mp = (float(m[0]), float(m[1]))
        if min(_dist(mp, v) for v in vertices) <= vertex_excl:
            continue
        inside, clear = _point_strictly_inside_convex(mp, vertices, edge_clearance=edge_clear)
        if not inside:
            continue
        d = float(token_edge_distance_to_point(p_token, mp))
        cands.append((d - 0.10 * float(clear), d, "marker", marker_attach, mp))

    for a in anchors if isinstance(anchors, list) else []:
        ap = (float(a[0]), float(a[1]))
        if min(_dist(ap, v) for v in vertices) <= vertex_excl:
            continue
        inside, clear = _point_strictly_inside_convex(ap, vertices, edge_clearance=edge_clear)
        if not inside:
            continue
        d = float(token_edge_distance_to_point(p_token, ap))
        cands.append((d + 2.0 - 0.08 * float(clear), d, "anchor", anchor_attach, ap))

    cands.sort(key=lambda t: float(t[0]))
    if cands:
        _, d_best, kind_best, attach_best, p_best = cands[0]
        if float(d_best) <= float(attach_best):
            return p_best, ""

    center_inside, clear_center = _point_strictly_inside_convex(p_center, vertices, edge_clearance=edge_clear)
    if center_inside:
        return p_center, ""

    if cands:
        _, d_best, kind_best, attach_best, _ = cands[0]
        if float(d_best) <= 1.35 * float(attach_best):
            return p_center, ""
        return None, f"P label is too far from interior {kind_best} candidate (dist={d_best:.1f}). "

    return None, f"P is not strictly inside square ABCD (clearance={clear_center:.1f}). "


def _line_matches_any(line_item, refs, min_hw):
    return any(line_equivalent(line_item, ref, min_hw=min_hw, angle_tol_deg=5.0) for ref in refs)


def _connector_intersection_candidates(lines, b_geo, d_geo, vertices, square_refs, min_hw):
    if not isinstance(lines, list) or not lines:
        return []

    pass_tol = scale_px(min_hw, 0.028, floor_px=0.0)
    endpoint_t_max = 0.34
    edge_clear = scale_px(min_hw, 0.010, floor_px=0.0)
    vertex_excl = scale_px(min_hw, 0.028, floor_px=0.0)
    b_lines = []
    d_lines = []

    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if "abc" not in ln or "seg" not in ln or "ang" not in ln or "len" not in ln:
            continue
        if _line_matches_any(ln, square_refs, min_hw=min_hw):
            continue

        tb = float(segment_projection_t(ln["seg"], b_geo))
        td = float(segment_projection_t(ln["seg"], d_geo))
        db = float(point_line_distance(b_geo, ln["abc"]))
        dd = float(point_line_distance(d_geo, ln["abc"]))

        if db <= pass_tol and (tb <= endpoint_t_max or tb >= (1.0 - endpoint_t_max)):
            b_lines.append(ln)
        if dd <= pass_tol and (td <= endpoint_t_max or td >= (1.0 - endpoint_t_max)):
            d_lines.append(ln)

    raw = []
    for lb in b_lines:
        for ld in d_lines:
            if angle_diff_deg(lb["ang"], ld["ang"]) < 3.0:
                continue
            p = line_intersection_from_abc(lb["abc"], ld["abc"])
            if p is None:
                continue
            p = (float(p[0]), float(p[1]))
            inside, clear = _point_strictly_inside_convex(p, vertices, edge_clearance=edge_clear)
            if not inside:
                continue
            if min(_dist(p, v) for v in vertices) < vertex_excl:
                continue
            t_b = float(segment_projection_t(lb["seg"], p))
            t_d = float(segment_projection_t(ld["seg"], p))
            if not (-0.15 <= t_b <= 1.15 and -0.15 <= t_d <= 1.15):
                continue
            off = abs(t_b - 0.50) + abs(t_d - 0.50)
            score = float(lb["len"] + ld["len"]) - 60.0 * float(off) + 8.0 * float(clear)
            raw.append((score, p))

    raw.sort(key=lambda t: float(t[0]), reverse=True)
    pts = [p for _, p in raw]
    return dedup_points(pts, tol=scale_px(min_hw, 0.020, floor_px=4.0))


def _resolve_required_connector(img, lines, p1, p2, square_refs, min_hw, name):
    ok_seg, ratio = has_segment_between_points(img, p1, p2, ratio_th=0.11, thickness=2, trim_ratio=0.02)
    line_ref = find_support_line(lines, p1, p2, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.035)
    if line_ref is not None:
        if any(line_equivalent(line_ref, edge, min_hw=min_hw, angle_tol_deg=5.0) for edge in square_refs):
            line_ref = None
    if not ok_seg and line_ref is None:
        return None, f"Missing segment {name} (ratio={ratio:.2f}). "
    return line_ref if line_ref is not None else _segment_item(p1, p2), ""


def _detect_unexpected_connector(img, lines, p1, p2, allowed_refs, square_refs, min_hw):
    ok_strict, ratio = has_segment_between_points(img, p1, p2, ratio_th=0.16, thickness=2, trim_ratio=0.02)
    if not ok_strict:
        return False, float(ratio)
    line_ref = find_support_line(lines, p1, p2, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.03)
    if line_ref is None:
        return False, float(ratio)
    all_refs = list(square_refs) + list(allowed_refs)
    if any(line_equivalent(line_ref, ref, min_hw=min_hw, angle_tol_deg=5.0) for ref in all_refs):
        return False, float(ratio)
    return True, float(ratio)


def _count_extra_dominant_lines(lines, expected_refs, min_hw):
    long_th = scale_px(min_hw, 0.20, floor_px=0.0)
    extras = 0
    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        if float(ln.get("len", 0.0)) < long_th:
            continue
        matched = any(line_equivalent(ln, ref, min_hw=min_hw, angle_tol_deg=5.0) for ref in expected_refs)
        if not matched:
            extras += 1
    return int(extras)

def judge_plane_65(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 5:
        return [False, "Insufficient line structure for square with PB and PD. "]

    square = _extract_square_candidate(img, lines=lines, min_hw=min_hw)
    if square is None:
        return [False, "Failed to reconstruct a valid square boundary. "]
    vertices = square["vertices"]
    sq_metrics = square["metrics"]
    if sq_metrics["area"] < scale_area(img.shape[0], img.shape[1], 0.008, floor_px=0):
        return [False, "Square area is too small or degenerate. "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDP", min_conf=0.08)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D", "P"], min_conf=0.08)
    missing = [ch for ch in ["A", "B", "C", "D"] if ch not in best_by_char]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    label_radius = max(scale_px(min_hw, 0.08, floor_px=0.0), 0.36 * float(sq_metrics["mean_edge"]))
    ok_cycle, detected, cyc_vertices = match_labels_in_cycle(
        tokens=tokens,
        vertices=vertices,
        target_labels=["A", "B", "C", "D"],
        max_dist=label_radius,
        allow_reversed=True,
        min_conf=0.08,
        single_char_only=False,
    )
    if not ok_cycle:
        hits = sum(1 for x in detected if x is not None)
        return [False, f"Failed to match A/B/C/D labels to square vertices (hits={hits}). "]

    a_geo, b_geo, c_geo, d_geo = cyc_vertices
    labeled_metrics = _square_metrics([a_geo, b_geo, c_geo, d_geo])
    if labeled_metrics is None:
        return [False, "Failed to derive labeled square geometry. "]
    if max(labeled_metrics["cos_vals"]) > 0.28:
        return [False, "Labeled A/B/C/D do not form a square-like right-angle shape. "]
    if labeled_metrics["edge_rel"] > 0.24:
        return [False, f"Labeled A/B/C/D are not equal-sided enough (edge_rel={labeled_metrics['edge_rel']:.3f}). "]

    label_far_th = max(scale_px(min_hw, 0.05, floor_px=0.0), 0.34 * float(labeled_metrics["mean_edge"]))
    far = []
    for ch, p in [("A", a_geo), ("B", b_geo), ("C", c_geo), ("D", d_geo)]:
        d = float(token_edge_distance_to_point(best_by_char[ch], p))
        if d > label_far_th:
            far.append((ch, d))
    if far:
        detail = ",".join(f"{ch}:{d:.1f}" for ch, d in far)
        return [False, f"Some vertex labels are too far from square corners ({detail}). "]

    ab = _segment_item(a_geo, b_geo)
    bc = _segment_item(b_geo, c_geo)
    cd = _segment_item(c_geo, d_geo)
    da = _segment_item(d_geo, a_geo)
    square_refs = [ab, bc, cd, da]

    for name, p, q in [("AB", a_geo, b_geo), ("BC", b_geo, c_geo), ("CD", c_geo, d_geo), ("DA", d_geo, a_geo)]:
        ok_side, ratio = has_segment_between_points(img, p, q, ratio_th=0.10, thickness=2, trim_ratio=0.02)
        if ok_side:
            continue
        fallback = find_support_line(lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08)
        if fallback is None:
            return [False, f"Missing square side {name} (ratio={ratio:.2f}). "]

    anchors = collect_intersection_anchors(
        lines=lines,
        img_shape=img.shape,
        min_angle_sep_deg=8.0,
        margin_ratio=0.16,
        point_tol_ratio=0.03,
        support_dist_ratio=0.02,
        support_t=(-0.60, 1.40),
    )
    markers = detect_marker_points(img)
    p_tok = best_by_char.get("P")
    p_candidates = []
    p_fail_reason = "Failed to locate a valid interior point connected to B and D. "

    if p_tok is not None:
        p_geo_from_label, p_reason = _locate_p_geo(
            vertices=[a_geo, b_geo, c_geo, d_geo],
            p_token=p_tok,
            markers=markers,
            anchors=anchors,
            min_hw=min_hw,
        )
        if p_geo_from_label is not None:
            p_candidates.append((float(p_geo_from_label[0]), float(p_geo_from_label[1])))
        elif p_reason:
            p_fail_reason = p_reason

        p_center = (float(p_tok["center"][0]), float(p_tok["center"][1]))
        inside_center, _ = _point_strictly_inside_convex(
            point=p_center,
            vertices=[a_geo, b_geo, c_geo, d_geo],
            edge_clearance=scale_px(min_hw, 0.010, floor_px=0.0),
        )
        if inside_center:
            p_candidates.append(p_center)

    p_candidates.extend(
        _connector_intersection_candidates(
            lines=lines,
            b_geo=b_geo,
            d_geo=d_geo,
            vertices=[a_geo, b_geo, c_geo, d_geo],
            square_refs=square_refs,
            min_hw=min_hw,
        )
    )
    p_candidates.extend(
        (float(p[0]), float(p[1]))
        for p in (markers if isinstance(markers, list) else [])
    )
    p_candidates.extend(
        (float(p[0]), float(p[1]))
        for p in (anchors if isinstance(anchors, list) else [])
    )
    p_candidates = dedup_points(p_candidates, tol=scale_px(min_hw, 0.020, floor_px=4.0))

    p_geo = None
    bp_ref = None
    dp_ref = None
    for cand in p_candidates:
        inside_ok, clear = _point_strictly_inside_convex(
            point=cand,
            vertices=[a_geo, b_geo, c_geo, d_geo],
            edge_clearance=scale_px(min_hw, 0.010, floor_px=0.0),
        )
        if not inside_ok:
            continue
        if min(_dist(cand, v) for v in [a_geo, b_geo, c_geo, d_geo]) < scale_px(min_hw, 0.028, floor_px=0.0):
            continue

        bp_try, bp_reason = _resolve_required_connector(
            img=img,
            lines=lines,
            p1=b_geo,
            p2=cand,
            square_refs=square_refs,
            min_hw=min_hw,
            name="BP",
        )
        if bp_try is None:
            p_fail_reason = bp_reason
            continue
        dp_try, dp_reason = _resolve_required_connector(
            img=img,
            lines=lines,
            p1=d_geo,
            p2=cand,
            square_refs=square_refs,
            min_hw=min_hw,
            name="DP",
        )
        if dp_try is None:
            p_fail_reason = dp_reason
            continue
        p_geo = cand
        bp_ref = bp_try
        dp_ref = dp_try
        break

    if p_geo is None or bp_ref is None or dp_ref is None:
        return [False, p_fail_reason]

    if p_tok is not None:
        p_label_far_th = max(scale_px(min_hw, 0.14, floor_px=0.0), 0.18 * float(labeled_metrics["diag"]))
        p_label_dist = float(token_edge_distance_to_point(p_tok, p_geo))
        if p_label_dist > p_label_far_th:
            return [False, f"P label is too far from interior point marker (dist={p_label_dist:.1f}). "]

    bad_pa, ratio_pa = _detect_unexpected_connector(
        img=img,
        lines=lines,
        p1=p_geo,
        p2=a_geo,
        allowed_refs=[bp_ref, dp_ref],
        square_refs=square_refs,
        min_hw=min_hw,
    )
    if bad_pa:
        return [False, f"Detected unexpected connector PA (ratio={ratio_pa:.2f}). "]
    bad_pc, ratio_pc = _detect_unexpected_connector(
        img=img,
        lines=lines,
        p1=p_geo,
        p2=c_geo,
        allowed_refs=[bp_ref, dp_ref],
        square_refs=square_refs,
        min_hw=min_hw,
    )
    if bad_pc:
        return [False, f"Detected unexpected connector PC (ratio={ratio_pc:.2f}). "]

    expected_refs = square_refs + [bp_ref, dp_ref]
    extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
    if extra_lines > 2:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_65,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
