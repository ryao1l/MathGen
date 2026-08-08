import argparse
import math

PID = 61
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


def _rectangle_metrics(vertices):
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

    opp_rel_1 = abs(edges[0] - edges[2]) / max(1e-6, 0.5 * (edges[0] + edges[2]))
    opp_rel_2 = abs(edges[1] - edges[3]) / max(1e-6, 0.5 * (edges[1] + edges[3]))
    pair_1 = 0.5 * (edges[0] + edges[2])
    pair_2 = 0.5 * (edges[1] + edges[3])
    width = max(pair_1, pair_2)
    height = min(pair_1, pair_2)
    diag = max(_dist(vertices[0], vertices[2]), _dist(vertices[1], vertices[3]))
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "opp_rel_1": float(opp_rel_1),
        "opp_rel_2": float(opp_rel_2),
        "width": float(width),
        "height": float(height),
        "diag": float(diag),
        "min_edge": float(min(edges)),
    }


def _extract_rectangle_candidate(img, lines, min_hw):
    h, w = img.shape[:2]
    area_norm = max(1.0, float(h) * float(w))
    configs = [
        (0.18, 12, 0.010, 12.0),
        (0.15, 14, 0.008, 10.0),
        (0.12, 16, 0.006, 8.0),
    ]
    base_lines = [ln for ln in lines if isinstance(ln, dict)] if isinstance(lines, list) else []
    if len(base_lines) < 4:
        return None

    best = None
    best_key = None
    pool_drop_cap = min(8, max(0, len(base_lines) - 4))

    for min_len_ratio, top_k, min_area_ratio, min_angle_sep in configs:
        pools = [base_lines]
        for drop_idx in range(pool_drop_cap):
            pools.append([ln for j, ln in enumerate(base_lines) if j != drop_idx])

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
            metrics = _rectangle_metrics(vertices)
            if metrics is None:
                continue
            if max(metrics["cos_vals"]) > 0.25:
                continue
            if metrics["opp_rel_1"] > 0.28 or metrics["opp_rel_2"] > 0.28:
                continue
            if metrics["min_edge"] < scale_px(min_hw, 0.10, floor_px=0.0):
                continue

            edge_ratios = []
            sides_ok = True
            for i in range(4):
                p = vertices[i]
                q = vertices[(i + 1) % 4]
                ok_side, ratio = has_segment_between_points(img, p, q, ratio_th=0.10, thickness=2, trim_ratio=0.02)
                if not ok_side:
                    side_line = find_support_line(base_lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08)
                    if side_line is None:
                        sides_ok = False
                        break
                edge_ratios.append(float(ratio))
            if not sides_ok:
                continue

            key = (
                float(quad.get("area", 0.0)) / area_norm,
                min(edge_ratios) if edge_ratios else 0.0,
                -max(metrics["cos_vals"]),
                -(metrics["opp_rel_1"] + metrics["opp_rel_2"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "vertices": vertices,
                    "lines": quad["lines"],
                    "metrics": metrics,
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
        _, d_best, kind_best, attach_best, p_best = cands[0]
        if float(d_best) <= 1.35 * float(attach_best):
            return p_best, ""
        return None, f"P label is too far from interior {kind_best} candidate (dist={d_best:.1f}). "

    return None, f"P is not strictly inside the rectangle (clearance={clear_center:.1f}). "


def _resolve_required_connector(img, lines, p1, p2, rect_refs, min_hw, name):
    ok_seg, ratio = has_segment_between_points(img, p1, p2, ratio_th=0.11, thickness=2, trim_ratio=0.02)
    line_ref = find_support_line(lines, p1, p2, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.035)
    if line_ref is not None:
        if any(line_equivalent(line_ref, edge, min_hw=min_hw, angle_tol_deg=5.0) for edge in rect_refs):
            line_ref = None
    if not ok_seg and line_ref is None:
        return None, f"Missing segment {name} (ratio={ratio:.2f}). "
    return line_ref if line_ref is not None else _segment_item(p1, p2), ""


def _detect_unexpected_connector(img, lines, p1, p2, allowed_refs, rect_refs, min_hw):
    ok_strict, ratio = has_segment_between_points(img, p1, p2, ratio_th=0.16, thickness=2, trim_ratio=0.02)
    if not ok_strict:
        return False, float(ratio)
    line_ref = find_support_line(lines, p1, p2, min_hw=min_hw, ang_tol_deg=11.0, dist_ratio=0.03)
    if line_ref is None:
        return False, float(ratio)
    all_refs = list(rect_refs) + list(allowed_refs)
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


def _label_ink_near_point(img, point, min_hw, refs):
    if img is None or point is None:
        return 0
    h, w = img.shape[:2]
    px, py = float(point[0]), float(point[1])
    _, bw = _gray_and_ink_mask(img)
    win = scale_px(min_hw, 0.11, floor_px=0.0)
    x1 = int(max(0, px - win))
    x2 = int(min(w, px + win))
    y1 = int(max(0, py - win))
    y2 = int(min(h, py + win))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    point_disk = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.020 * min_hw) ** 2
    line_band = np.zeros_like(roi, dtype=bool)
    for ln in refs if isinstance(refs, list) else []:
        a, b, c = [float(v) for v in ln["abc"]]
        den = max(1e-6, math.hypot(a, b))
        line_band |= np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den <= max(3.0, 0.010 * min_hw)
    label_mask = roi & (~point_disk) & (~line_band)
    if int(label_mask.sum()) <= 0:
        return 0
    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    min_area = max(8, int(0.000005 * h * w))
    max_area = max(350, int(0.0040 * h * w))
    areas = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw_box > 0.12 * min_hw or bh_box > 0.12 * min_hw:
            continue
        areas.append(area)
    return max(areas) if areas else 0


def _try_geometry_fallback_without_ocr(img, lines, vertices, min_hw):
    metrics = _rectangle_metrics(vertices)
    if metrics is None:
        return None, "Failed to derive rectangle geometry for OCR fallback. "
    if max(metrics["cos_vals"]) > 0.28:
        return None, "Fallback rectangle is not rectangle-like. "
    if metrics["opp_rel_1"] > 0.30 or metrics["opp_rel_2"] > 0.30:
        return None, "Fallback rectangle opposite sides are unstable. "

    rect_refs = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    edge_clearance = scale_px(min_hw, 0.012, floor_px=0.0)
    vertex_excl = max(scale_px(min_hw, 0.025, floor_px=0.0), 0.06 * float(metrics["diag"]))

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
    candidates = []
    for p in list(anchors or []) + list(markers or []):
        pp = (float(p[0]), float(p[1]))
        if min(_dist(pp, v) for v in vertices) <= vertex_excl:
            continue
        inside, clear = _point_strictly_inside_convex(pp, vertices, edge_clearance=edge_clearance)
        if not inside:
            continue
        ink = _label_ink_near_point(img, pp, min_hw, refs=rect_refs)
        if ink < max(24, int(0.000008 * img.shape[0] * img.shape[1])):
            continue
        candidates.append((float(clear), int(ink), pp))

    if not candidates:
        return None, "Fallback could not find an interior P marker/label. "
    candidates.sort(key=lambda t: (-t[0], -t[1]))

    opposite_pairs = [(0, 2), (1, 3)]
    for _, _, p_geo in candidates[:8]:
        for i, j in opposite_pairs:
            ref_i, reason_i = _resolve_required_connector(
                img=img,
                lines=lines,
                p1=vertices[i],
                p2=p_geo,
                rect_refs=rect_refs,
                min_hw=min_hw,
                name=f"V{i}P",
            )
            ref_j, reason_j = _resolve_required_connector(
                img=img,
                lines=lines,
                p1=vertices[j],
                p2=p_geo,
                rect_refs=rect_refs,
                min_hw=min_hw,
                name=f"V{j}P",
            )
            if ref_i is None or ref_j is None:
                continue
            expected_refs = rect_refs + [ref_i, ref_j]
            extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
            if extra_lines > 8:
                continue
            return True, ""

    return None, "Fallback found P but not two required diagonal-corner connectors. "


def judge_plane_61(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 5:
        return [False, "Insufficient line structure for rectangle with AP and CP. "]

    rect = _extract_rectangle_candidate(img, lines=lines, min_hw=min_hw)
    if rect is None:
        return [False, "Failed to reconstruct a valid rectangle boundary. "]
    vertices = rect["vertices"]
    rect_metrics = rect["metrics"]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDP", min_conf=0.08)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D", "P"], min_conf=0.08)
    missing = [ch for ch in ["A", "B", "C", "D", "P"] if ch not in best_by_char]
    if missing:
        ok_fallback, fallback_reason = _try_geometry_fallback_without_ocr(
            img=img,
            lines=lines,
            vertices=vertices,
            min_hw=min_hw,
        )
        if ok_fallback:
            return [True, ""]
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    label_radius = max(scale_px(min_hw, 0.08, floor_px=0.0), 0.34 * float(rect_metrics["height"]))
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
        return [False, f"Failed to match A/B/C/D labels to rectangle vertices (hits={hits}). "]

    a_geo, b_geo, c_geo, d_geo = cyc_vertices
    labeled_metrics = _rectangle_metrics([a_geo, b_geo, c_geo, d_geo])
    if labeled_metrics is None:
        return [False, "Failed to derive labeled rectangle geometry. "]
    if max(labeled_metrics["cos_vals"]) > 0.28:
        return [False, "Labeled A/B/C/D do not form a rectangle-like shape. "]
    if labeled_metrics["opp_rel_1"] > 0.30 or labeled_metrics["opp_rel_2"] > 0.30:
        return [False, "Labeled A/B/C/D do not form stable opposite sides. "]

    label_far_th = max(scale_px(min_hw, 0.05, floor_px=0.0), 0.33 * float(labeled_metrics["height"]))
    far = []
    for ch, p in [("A", a_geo), ("B", b_geo), ("C", c_geo), ("D", d_geo)]:
        d = float(token_edge_distance_to_point(best_by_char[ch], p))
        if d > label_far_th:
            far.append((ch, d))
    if far:
        detail = ",".join(f"{ch}:{d:.1f}" for ch, d in far)
        return [False, f"Some vertex labels are too far from rectangle corners ({detail}). "]

    ab = _segment_item(a_geo, b_geo)
    bc = _segment_item(b_geo, c_geo)
    cd = _segment_item(c_geo, d_geo)
    da = _segment_item(d_geo, a_geo)
    rect_refs = [ab, bc, cd, da]

    for name, p, q in [("AB", a_geo, b_geo), ("BC", b_geo, c_geo), ("CD", c_geo, d_geo), ("DA", d_geo, a_geo)]:
        ok_side, ratio = has_segment_between_points(img, p, q, ratio_th=0.10, thickness=2, trim_ratio=0.02)
        if ok_side:
            continue
        fallback = find_support_line(lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08)
        if fallback is None:
            return [False, f"Missing rectangle side {name} (ratio={ratio:.2f}). "]

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
    p_tok = best_by_char["P"]
    p_geo, p_reason = _locate_p_geo(vertices=[a_geo, b_geo, c_geo, d_geo], p_token=p_tok, markers=markers, anchors=anchors, min_hw=min_hw)
    if p_geo is None:
        return [False, p_reason]

    inside_ok, clear = _point_strictly_inside_convex(
        point=p_geo,
        vertices=[a_geo, b_geo, c_geo, d_geo],
        edge_clearance=scale_px(min_hw, 0.010, floor_px=0.0),
    )
    if not inside_ok:
        return [False, f"P is not strictly inside rectangle ABCD (clearance={clear:.1f}). "]

    if min(_dist(p_geo, v) for v in [a_geo, b_geo, c_geo, d_geo]) < scale_px(min_hw, 0.028, floor_px=0.0):
        return [False, "P is too close to a rectangle vertex. "]

    def _try_connectors(point):
        ap_ref_local, ap_reason_local = _resolve_required_connector(
            img=img,
            lines=lines,
            p1=a_geo,
            p2=point,
            rect_refs=rect_refs,
            min_hw=min_hw,
            name="AP",
        )
        if ap_ref_local is None:
            return None, None, ap_reason_local
        cp_ref_local, cp_reason_local = _resolve_required_connector(
            img=img,
            lines=lines,
            p1=c_geo,
            p2=point,
            rect_refs=rect_refs,
            min_hw=min_hw,
            name="CP",
        )
        if cp_ref_local is None:
            return None, None, cp_reason_local
        return ap_ref_local, cp_ref_local, ""

    ap_ref, cp_ref, conn_reason = _try_connectors(p_geo)
    if ap_ref is None or cp_ref is None:
        p_center = (float(p_tok["center"][0]), float(p_tok["center"][1]))
        ac_ref = _segment_item(a_geo, c_geo)
        proj = project_point_to_line(p_center, ac_ref["abc"])
        if proj is not None:
            p_proj = (float(proj[0]), float(proj[1]))
            p_proj_inside, _ = _point_strictly_inside_convex(
                point=p_proj,
                vertices=[a_geo, b_geo, c_geo, d_geo],
                edge_clearance=scale_px(min_hw, 0.010, floor_px=0.0),
            )
            p_proj_t = float(segment_projection_t(ac_ref["seg"], p_proj))
            p_proj_dist = float(token_edge_distance_to_point(p_tok, p_proj))
            p_proj_max_dist = max(scale_px(min_hw, 0.08, floor_px=0.0), 0.10 * float(labeled_metrics["diag"]))
            if p_proj_inside and 0.02 <= p_proj_t <= 0.98 and p_proj_dist <= p_proj_max_dist:
                ap_try, cp_try, conn_reason_try = _try_connectors(p_proj)
                if ap_try is not None and cp_try is not None:
                    p_geo = p_proj
                    ap_ref, cp_ref = ap_try, cp_try
                else:
                    return [False, conn_reason_try]
            else:
                return [False, conn_reason]
        else:
            return [False, conn_reason]

    p_label_far_th = max(scale_px(min_hw, 0.14, floor_px=0.0), 0.18 * float(labeled_metrics["diag"]))
    p_label_dist = float(token_edge_distance_to_point(p_tok, p_geo))
    if p_label_dist > p_label_far_th:
        return [False, f"P label is too far from interior point marker (dist={p_label_dist:.1f}). "]

    bad_pb, ratio_pb = _detect_unexpected_connector(
        img=img,
        lines=lines,
        p1=p_geo,
        p2=b_geo,
        allowed_refs=[ap_ref, cp_ref],
        rect_refs=rect_refs,
        min_hw=min_hw,
    )
    if bad_pb:
        return [False, f"Detected unexpected connector PB (ratio={ratio_pb:.2f}). "]
    bad_pd, ratio_pd = _detect_unexpected_connector(
        img=img,
        lines=lines,
        p1=p_geo,
        p2=d_geo,
        allowed_refs=[ap_ref, cp_ref],
        rect_refs=rect_refs,
        min_hw=min_hw,
    )
    if bad_pd:
        return [False, f"Detected unexpected connector PD (ratio={ratio_pd:.2f}). "]

    expected_refs = rect_refs + [ap_ref, cp_ref]
    extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
    if extra_lines > 2:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_61,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
