import argparse
import itertools

PID = 27
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
        raise RuntimeError("Failed to build line from segment points.")
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _labels_on_three_points(tokens, points, tol):
    labels = ["D", "E", "F"]
    for perm in itertools.permutations(range(3), 3):
        ok = True
        for i, ch in enumerate(labels):
            tok = select_token_near_point(tokens, expected_char=ch, point=points[perm[i]], max_dist=tol)
            if tok is None:
                ok = False
                break
        if ok:
            return True
    return False


def _lerp_point(p, q, t):
    tt = float(t)
    return (
        float(p[0]) + (float(q[0]) - float(p[0])) * tt,
        float(p[1]) + (float(q[1]) - float(p[1])) * tt,
    )


def _segment_window_ink_ratio(img, p1, p2, t0, t1, thickness=2):
    a = _lerp_point(p1, p2, t0)
    b = _lerp_point(p1, p2, t1)
    return float(segment_ink_ratio(img, a, b, thickness=thickness, trim_ratio=0.0))


def _triangle_area(a, b, c):
    return abs(
        float(a[0]) * (float(b[1]) - float(c[1]))
        + float(b[0]) * (float(c[1]) - float(a[1]))
        + float(c[0]) * (float(a[1]) - float(b[1]))
    ) * 0.5


def _build_anchor_pool(lines, img_shape, min_hw):
    anchors = collect_intersection_anchors(
        lines=lines,
        img_shape=img_shape,
        min_angle_sep_deg=8.0,
        margin_ratio=0.12,
        point_tol_ratio=0.02,
        support_t=(-0.75, 1.35),
    )
    pool = []
    for p in anchors:
        pool.append((float(p[0]), float(p[1])))

    long_th = scale_px(min_hw, 0.12, floor_px=0.0)
    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        if float(ln.get("len", 0.0)) < long_th:
            continue
        seg = ln.get("seg")
        if not (isinstance(seg, (list, tuple)) and len(seg) >= 4):
            continue
        p1 = (float(seg[0]), float(seg[1]))
        p2 = (float(seg[2]), float(seg[3]))
        pool.append(p1)
        pool.append(p2)

    return dedup_points(pool, tol=scale_px(min_hw, 0.015, floor_px=1.0))


def _label_anchor_candidates(token, anchors, max_dist, top_k=8):
    if not isinstance(token, dict):
        return []
    cands = []
    for p in anchors if isinstance(anchors, list) else []:
        d = token_edge_distance_to_point(token, p)
        if not math.isfinite(float(d)):
            continue
        if float(d) <= float(max_dist):
            cands.append((float(d), (float(p[0]), float(p[1]))))
    cands.sort(key=lambda z: z[0])
    return cands[: max(1, int(top_k))]


def _select_triangle_from_label_anchors(img, best_by_char, anchors, min_hw):
    if not isinstance(best_by_char, dict):
        return None
    if any(ch not in best_by_char for ch in ["A", "B", "C"]):
        return None

    h, w = img.shape[:2]
    max_dist = scale_px(min_hw, 0.20, floor_px=0.0)
    a_cands = _label_anchor_candidates(best_by_char["A"], anchors, max_dist=max_dist, top_k=10)
    b_cands = _label_anchor_candidates(best_by_char["B"], anchors, max_dist=max_dist, top_k=10)
    c_cands = _label_anchor_candidates(best_by_char["C"], anchors, max_dist=max_dist, top_k=10)
    if not a_cands or not b_cands or not c_cands:
        return None

    min_sep = scale_px(min_hw, 0.04, floor_px=0.0)
    y_margin = scale_px(min_hw, 0.012, floor_px=0.0)
    area_min = float(scale_area(h, w, 0.006, floor_px=0))
    side_min = scale_px(min_hw, 0.10, floor_px=0.0)
    best = None
    best_score = None

    for d_a, A in a_cands:
        for d_b, B in b_cands:
            if _dist(A, B) < min_sep:
                continue
            for d_c, C in c_cands:
                if _dist(A, C) < min_sep or _dist(B, C) < min_sep:
                    continue
                if float(A[1]) >= min(float(B[1]), float(C[1])) - y_margin:
                    continue
                if min(_dist(A, B), _dist(B, C), _dist(C, A)) < side_min:
                    continue
                area = _triangle_area(A, B, C)
                if area < area_min:
                    continue

                ok_ab, r_ab = has_segment_between_points(img, A, B, ratio_th=0.11, thickness=2, trim_ratio=0.02)
                ok_bc, r_bc = has_segment_between_points(img, B, C, ratio_th=0.11, thickness=2, trim_ratio=0.02)
                ok_ca, r_ca = has_segment_between_points(img, C, A, ratio_th=0.11, thickness=2, trim_ratio=0.02)
                if not (ok_ab and ok_bc and ok_ca):
                    continue

                dist_penalty = (float(d_a) + float(d_b) + float(d_c)) / max(1e-6, 3.0 * max_dist)
                side_quality = (float(r_ab) + float(r_bc) + float(r_ca)) / 3.0
                area_ratio = float(area) / float(max(1, h * w))
                score = 2.2 * side_quality + 0.8 * area_ratio - 0.9 * dist_penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best = {
                        "A": A,
                        "B": B,
                        "C": C,
                        "abc_dists": {"A": float(d_a), "B": float(d_b), "C": float(d_c)},
                        "side_ratios": {"AB": float(r_ab), "BC": float(r_bc), "CA": float(r_ca)},
                    }
    return best


def _select_triangle_from_line_candidates(img, lines, best_by_char, min_hw):
    tri_cands = extract_triangle_candidates_from_lines(
        lines=lines,
        img_shape=img.shape,
        min_len_ratio=0.16,
        top_k=14,
        min_angle_sep_deg=9.0,
        margin_ratio=0.22,
        point_tol_ratio=0.045,
        min_area_ratio=0.006,
        support_t=(-0.75, 1.35),
    )
    if not tri_cands:
        return None

    tol = scale_px(min_hw, 0.16, floor_px=0.0)
    best = None
    best_score = None
    y_margin = scale_px(min_hw, 0.012, floor_px=0.0)
    for cand in tri_cands:
        verts = cand["vertices"]
        assign, dists = assign_labels_to_vertices_min_cost(best_by_char, verts, ["A", "B", "C"])
        if assign is None or dists is None:
            continue
        if any(float(dists[ch]) > tol for ch in ["A", "B", "C"]):
            continue
        A = verts[assign["A"]]
        B = verts[assign["B"]]
        C = verts[assign["C"]]
        if float(A[1]) >= min(float(B[1]), float(C[1])) - y_margin:
            continue
        ok_ab, r_ab = has_segment_between_points(img, A, B, ratio_th=0.11, thickness=2, trim_ratio=0.02)
        ok_bc, r_bc = has_segment_between_points(img, B, C, ratio_th=0.11, thickness=2, trim_ratio=0.02)
        ok_ca, r_ca = has_segment_between_points(img, C, A, ratio_th=0.11, thickness=2, trim_ratio=0.02)
        if not (ok_ab and ok_bc and ok_ca):
            continue
        side_quality = (float(r_ab) + float(r_bc) + float(r_ca)) / 3.0
        dist_penalty = (float(dists["A"]) + float(dists["B"]) + float(dists["C"])) / max(1e-6, 3.0 * tol)
        score = 1.9 * side_quality + 0.5 * float(cand.get("area", 0.0)) / float(max(1, img.shape[0] * img.shape[1])) - 0.9 * dist_penalty
        if best_score is None or score > best_score:
            best_score = score
            best = {
                "A": (float(A[0]), float(A[1])),
                "B": (float(B[0]), float(B[1])),
                "C": (float(C[0]), float(C[1])),
                "abc_dists": {"A": float(dists["A"]), "B": float(dists["B"]), "C": float(dists["C"])},
                "side_ratios": {"AB": float(r_ab), "BC": float(r_bc), "CA": float(r_ca)},
            }
    return best


def _select_triangle_geometry_only(img, lines, min_hw):
    tri_cands = extract_triangle_candidates_from_lines(
        lines=lines,
        img_shape=img.shape,
        min_len_ratio=0.14,
        top_k=16,
        min_angle_sep_deg=8.0,
        margin_ratio=0.22,
        point_tol_ratio=0.050,
        min_area_ratio=0.006,
        support_t=(-0.75, 1.35),
    )
    if not tri_cands:
        return None

    best = None
    best_score = None
    for cand in tri_cands[:12]:
        verts = [(float(x), float(y)) for x, y in cand["vertices"]]
        top_idx = min(range(3), key=lambda i: verts[i][1])
        A = verts[top_idx]
        base = [verts[i] for i in range(3) if i != top_idx]
        base.sort(key=lambda p: p[0])
        B, C = base[0], base[1]
        if float(A[1]) >= min(float(B[1]), float(C[1])) - scale_px(min_hw, 0.012, floor_px=0.0):
            continue
        ok_ab, r_ab = has_segment_between_points(img, A, B, ratio_th=0.10, thickness=2, trim_ratio=0.02)
        ok_bc, r_bc = has_segment_between_points(img, B, C, ratio_th=0.10, thickness=2, trim_ratio=0.02)
        ok_ca, r_ca = has_segment_between_points(img, C, A, ratio_th=0.10, thickness=2, trim_ratio=0.02)
        if not (ok_ab and ok_bc and ok_ca):
            continue
        score = float(cand.get("area", 0.0)) + 80.0 * (float(r_ab) + float(r_bc) + float(r_ca))
        if best_score is None or score > best_score:
            best_score = score
            best = {
                "A": A,
                "B": B,
                "C": C,
                "abc_dists": {"A": 0.0, "B": 0.0, "C": 0.0},
                "side_ratios": {"AB": float(r_ab), "BC": float(r_bc), "CA": float(r_ca)},
            }
    return best


def _token_bbox(token):
    if not isinstance(token, dict):
        return None
    bbox = token.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _build_token_allow_mask(img_shape, tokens, min_hw):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pad = int(round(scale_px(min_hw, 0.01, floor_px=1.0)))
    for tok in tokens if isinstance(tokens, list) else []:
        b = _token_bbox(tok)
        if b is None:
            continue
        x1 = int(round(max(0.0, float(b[0]) - pad)))
        y1 = int(round(max(0.0, float(b[1]) - pad)))
        x2 = int(round(min(float(w - 1), float(b[2]) + pad)))
        y2 = int(round(min(float(h - 1), float(b[3]) + pad)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


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


def _support_line_or_virtual(lines, p1, p2, min_hw):
    ln = find_support_line(
        lines,
        p1,
        p2,
        min_hw=min_hw,
        ang_tol_deg=11.0,
        dist_ratio=0.06,
    )
    if ln is not None:
        return ln
    return _segment_item(p1, p2)


def _best_median_from_vertex(img, lines, vertex, opposite_side, target_midpoint, min_hw):
    if opposite_side is None or "abc" not in opposite_side or "seg" not in opposite_side:
        return None

    side_len = _dist(
        (float(opposite_side["seg"][0]), float(opposite_side["seg"][1])),
        (float(opposite_side["seg"][2]), float(opposite_side["seg"][3])),
    )
    if side_len <= 1e-6:
        return None

    v_tol = scale_px(min_hw, 0.055, floor_px=0.0)
    long_th = scale_px(min_hw, 0.10, floor_px=0.0)
    t_pad = 0.12
    best = None
    best_score = None

    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        if "abc" not in ln or "seg" not in ln or "len" not in ln:
            continue
        if float(ln["len"]) < long_th:
            continue
        if point_line_distance(vertex, ln["abc"]) > v_tol:
            continue
        if line_equivalent(ln, opposite_side, min_hw=min_hw, angle_tol_deg=8.0):
            continue

        foot = line_intersection_from_abc(ln["abc"], opposite_side["abc"])
        if foot is None:
            continue
        t_foot = segment_projection_t(opposite_side["seg"], foot)
        if t_foot < -t_pad or t_foot > (1.0 + t_pad):
            continue

        mid_rel = _dist(foot, target_midpoint) / max(1e-6, side_len)
        if mid_rel > 0.12:
            continue

        full = float(segment_ink_ratio(img, vertex, foot, thickness=2, trim_ratio=0.01))
        core = _segment_window_ink_ratio(img, vertex, foot, 0.14, 0.86, thickness=2)
        end = _segment_window_ink_ratio(img, vertex, foot, 0.60, 0.90, thickness=2)
        centroid_band = _segment_window_ink_ratio(img, vertex, foot, 0.58, 0.74, thickness=2)
        if full < 0.22 or core < 0.20:
            continue

        score = (
            1.8 * max(full, core)
            + 0.7 * end
            + 0.3 * centroid_band
            - 2.1 * mid_rel
            - 0.40 * abs(float(t_foot) - 0.5)
        )
        if best_score is None or score > best_score:
            best_score = score
            best = {
                "line": ln,
                "foot": (float(foot[0]), float(foot[1])),
                "mid_rel": float(mid_rel),
                "foot_t": float(t_foot),
                "full": float(full),
                "core": float(core),
                "end": float(end),
                "centroid_band": float(centroid_band),
                "score": float(score),
            }

    if best is not None:
        return best
    return None


def _validate_midpoint_labels(tokens, foot_d, foot_e, foot_f, min_hw):
    strict_tol = scale_px(min_hw, 0.15, floor_px=0.0)
    tok_d = select_token_near_point(tokens, expected_char="D", point=foot_d, max_dist=strict_tol)
    tok_e = select_token_near_point(tokens, expected_char="E", point=foot_e, max_dist=strict_tol)
    tok_f = select_token_near_point(tokens, expected_char="F", point=foot_f, max_dist=strict_tol)
    if tok_d is not None and tok_e is not None and tok_f is not None:
        return True
    return _labels_on_three_points(
        tokens=tokens,
        points=[foot_d, foot_e, foot_f],
        tol=scale_px(min_hw, 0.17, floor_px=0.0),
    )


def _median_support_stats(lines, median_line, start, foot, min_hw):
    base_seg = (float(start[0]), float(start[1]), float(foot[0]), float(foot[1]))
    return collinear_support_stats_on_segment(
        lines=lines,
        reference_line=median_line,
        base_seg=base_seg,
        min_hw=min_hw,
        angle_tol_deg=3.5,
        offset_ratio=0.03,
        offset_floor_px=8.0,
        reach_left_t=0.10,
        reach_right_t=0.90,
    )


def judge_plane_27(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))

    lines = detect_line_segments(img, min_len_ratio=0.06)
    if len(lines) < 4:
        return [False, "Insufficient line structure for hard case 27. "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDEFG", min_conf=0.08)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D", "E", "F", "G"], min_conf=0.08)
    missing_abc = [ch for ch in ["A", "B", "C"] if ch not in best_by_char]

    anchors = _build_anchor_pool(lines, img.shape, min_hw=min_hw)
    used_geom_triangle = False
    if missing_abc:
        tri = _select_triangle_geometry_only(img, lines, min_hw=min_hw)
        if tri is None:
            return [False, f"Missing labels from OCR: {','.join(missing_abc)}. "]
        used_geom_triangle = True
    else:
        tri = _select_triangle_from_label_anchors(img, best_by_char, anchors, min_hw=min_hw)
        if tri is None:
            tri = _select_triangle_from_line_candidates(img, lines, best_by_char, min_hw=min_hw)
    if tri is None:
        return [False, "Failed to localize a valid outer triangle ABC from labels and line geometry."]

    a_geo = tri["A"]
    b_geo = tri["B"]
    c_geo = tri["C"]
    abc_dists = tri.get("abc_dists", {})

    abc_tol = scale_px(min_hw, 0.16, floor_px=0.0)
    if any(float(abc_dists.get(ch, 1e9)) > abc_tol for ch in ["A", "B", "C"]):
        return [False, "A/B/C labels are too far from triangle vertices."]

    if float(a_geo[1]) >= min(float(b_geo[1]), float(c_geo[1])) - scale_px(min_hw, 0.012, floor_px=0.0):
        return [False, "A is not the top vertex."]

    ok_ab, r_ab = has_segment_between_points(img, a_geo, b_geo, ratio_th=0.11, thickness=2, trim_ratio=0.02)
    ok_bc, r_bc = has_segment_between_points(img, b_geo, c_geo, ratio_th=0.11, thickness=2, trim_ratio=0.02)
    ok_ca, r_ca = has_segment_between_points(img, c_geo, a_geo, ratio_th=0.11, thickness=2, trim_ratio=0.02)
    if not (ok_ab and ok_bc and ok_ca):
        return [False, f"Outer triangle incomplete (AB={r_ab:.2f}, BC={r_bc:.2f}, CA={r_ca:.2f}). "]

    m_bc = ((float(b_geo[0]) + float(c_geo[0])) * 0.5, (float(b_geo[1]) + float(c_geo[1])) * 0.5)
    m_ac = ((float(a_geo[0]) + float(c_geo[0])) * 0.5, (float(a_geo[1]) + float(c_geo[1])) * 0.5)
    m_ab = ((float(a_geo[0]) + float(b_geo[0])) * 0.5, (float(a_geo[1]) + float(b_geo[1])) * 0.5)



    ab_side = _segment_item(a_geo, b_geo)
    bc_side = _segment_item(b_geo, c_geo)
    ca_side = _segment_item(c_geo, a_geo)

    med_a = _best_median_from_vertex(
        img=img,
        lines=lines,
        vertex=a_geo,
        opposite_side=bc_side,
        target_midpoint=m_bc,
        min_hw=min_hw,
    )
    med_b = _best_median_from_vertex(
        img=img,
        lines=lines,
        vertex=b_geo,
        opposite_side=ca_side,
        target_midpoint=m_ac,
        min_hw=min_hw,
    )
    med_c = _best_median_from_vertex(
        img=img,
        lines=lines,
        vertex=c_geo,
        opposite_side=ab_side,
        target_midpoint=m_ab,
        min_hw=min_hw,
    )
    if med_a is None or med_b is None or med_c is None:
        return [False, "Failed to detect all three medians from line geometry."]

    rel_a = float(med_a["mid_rel"])
    rel_b = float(med_b["mid_rel"])
    rel_c = float(med_c["mid_rel"])
    if max(rel_a, rel_b, rel_c) > 0.10:
        return [False, f"Detected median foot(s) too far from side midpoints (A={rel_a:.3f}, B={rel_b:.3f}, C={rel_c:.3f}). "]

    t_dev = max(abs(float(med_a["foot_t"]) - 0.5), abs(float(med_b["foot_t"]) - 0.5), abs(float(med_c["foot_t"]) - 0.5))
    if t_dev > 0.12:
        return [False, f"Median foot parameter deviates too much from side midpoint (max_dev={t_dev:.3f}). "]

    if min(float(med_a["core"]), float(med_b["core"]), float(med_c["core"])) < 0.20:
        return [
            False,
            "Missing/incomplete median segment(s): "
            f"AD(full={med_a['full']:.2f},core={med_a['core']:.2f},foot={med_a['end']:.2f}), "
            f"BE(full={med_b['full']:.2f},core={med_b['core']:.2f},foot={med_b['end']:.2f}), "
            f"CF(full={med_c['full']:.2f},core={med_c['core']:.2f},foot={med_c['end']:.2f}). ",
        ]

    foot_d = med_a["foot"]
    foot_e = med_b["foot"]
    foot_f = med_c["foot"]

    if not _validate_midpoint_labels(tokens, foot_d, foot_e, foot_f, min_hw=min_hw):
        if not used_geom_triangle:
            return [False, "Failed to align labels D/E/F with median feet on opposite sides."]

    g_ab = line_intersection_from_abc(med_a["line"]["abc"], med_b["line"]["abc"])
    g_ac = line_intersection_from_abc(med_a["line"]["abc"], med_c["line"]["abc"])
    g_bc = line_intersection_from_abc(med_b["line"]["abc"], med_c["line"]["abc"])
    if g_ab is None or g_ac is None or g_bc is None:
        return [False, "Failed to intersect median lines."]

    g_geo = (
        (float(g_ab[0]) + float(g_ac[0]) + float(g_bc[0])) / 3.0,
        (float(g_ab[1]) + float(g_ac[1]) + float(g_bc[1])) / 3.0,
    )
    conc_spread = max(_dist(g_geo, g_ab), _dist(g_geo, g_ac), _dist(g_geo, g_bc))
    if conc_spread > scale_px(min_hw, 0.045, floor_px=0.0):
        return [False, f"The three medians are not concurrent enough (spread={conc_spread:.1f}). "]




    g_to_d = float(segment_ink_ratio(img, g_geo, m_bc, thickness=2, trim_ratio=0.02))
    g_to_e = float(segment_ink_ratio(img, g_geo, m_ac, thickness=2, trim_ratio=0.02))
    g_to_f = float(segment_ink_ratio(img, g_geo, m_ab, thickness=2, trim_ratio=0.02))
    reach_th = 0.025 if used_geom_triangle else 0.12
    if min(g_to_d, g_to_e, g_to_f) < reach_th:
        return [
            False,
            "Median segment does not clearly reach opposite-side midpoint region: "
            f"G->D={g_to_d:.2f}, G->E={g_to_e:.2f}, G->F={g_to_f:.2f}. ",
        ]

    centroid_th = 0.025 if used_geom_triangle else 0.04
    if min(float(med_a["centroid_band"]), float(med_b["centroid_band"]), float(med_c["centroid_band"])) < centroid_th:
        return [False, "Median support around centroid is too weak."]

    g_tok = select_token_near_point(tokens, expected_char="G", point=g_geo, max_dist=scale_px(min_hw, 0.13, floor_px=0.0))
    if g_tok is None and not used_geom_triangle:
        return [False, "Failed to detect centroid label G near median intersection."]

    expected_refs = [
        ab_side,
        bc_side,
        ca_side,
        med_a["line"],
        med_b["line"],
        med_c["line"],
    ]
    extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
    if extra_lines > 2:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    ideal_allow_segments = [
        {"seg": _segment_item(a_geo, b_geo)["seg"]},
        {"seg": _segment_item(b_geo, c_geo)["seg"]},
        {"seg": _segment_item(c_geo, a_geo)["seg"]},
        {"seg": _segment_item(a_geo, foot_d)["seg"]},
        {"seg": _segment_item(b_geo, foot_e)["seg"]},
        {"seg": _segment_item(c_geo, foot_f)["seg"]},
    ]

    label_tokens = [best_by_char[ch] for ch in ["A", "B", "C", "D", "E", "F", "G"] if ch in best_by_char]
    token_mask = _build_token_allow_mask(img.shape, label_tokens, min_hw=min_hw)
    outside_ratio, outside_px, total_ink = compute_outside_ink_stats(
        img=img,
        allowed_lines=ideal_allow_segments,
        anchor_points=[a_geo, b_geo, c_geo, foot_d, foot_e, foot_f, g_geo],
        band_ratio=0.018,
        band_floor_px=1,
        anchor_radius_ratio=0.02,
        anchor_radius_floor_px=1,
        extra_allow_mask=token_mask,
    )
    outside_px_th = scale_area(img.shape[0], img.shape[1], 0.00010, floor_px=0)
    outside_ratio_th = 0.18 if used_geom_triangle else 0.14
    if outside_px > outside_px_th and outside_ratio > outside_ratio_th:
        return [
            False,
            "Detected excessive outside structure "
            f"(outside_ratio={outside_ratio:.3f}, outside_px={outside_px}, total_ink={total_ink}). ",
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_27,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
