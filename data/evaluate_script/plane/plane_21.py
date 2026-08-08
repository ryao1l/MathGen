import argparse
import math

PID = 44
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


def _point_inside_triangle_soft(point, vertices, tol_dist):
    if vertices is None or len(vertices) != 3:
        return False
    px, py = float(point[0]), float(point[1])

    area2 = 0.0
    for i in range(3):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % 3][0]), float(vertices[(i + 1) % 3][1])
        area2 += x1 * y2 - x2 * y1
    orient = 1.0 if area2 >= 0.0 else -1.0

    for i in range(3):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % 3][0]), float(vertices[(i + 1) % 3][1])
        ex, ey = x2 - x1, y2 - y1
        cross = ex * (py - y1) - ey * (px - x1)
        edge_len = math.hypot(ex, ey)
        if edge_len <= 1e-6:
            return False
        if orient * cross < -float(edge_len) * float(tol_dist):
            return False
    return True


def _point_strictly_inside_triangle(point, vertices, edge_clearance):
    if vertices is None or len(vertices) != 3:
        return False, 0.0
    dmin = min(
        _point_to_segment_distance(point, vertices[i], vertices[(i + 1) % 3])
        for i in range(3)
    )
    if dmin <= float(edge_clearance):
        return False, float(dmin)
    inside = _point_inside_triangle_soft(
        point=point,
        vertices=vertices,
        tol_dist=max(1e-6, 0.35 * float(edge_clearance)),
    )
    return bool(inside), float(dmin)


def _project_to_line_point(point, line_item):
    proj = project_point_to_line(point, line_item["abc"])
    if proj is None:
        return (float(point[0]), float(point[1]))
    return (float(proj[0]), float(proj[1]))


def _count_extra_dominant_lines(lines, expected_refs, min_hw):
    long_th = scale_px(min_hw, 0.20, floor_px=0.0)
    extras = 0
    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        if float(ln.get("len", 0.0)) < long_th:
            continue
        matched = any(
            line_equivalent(ln, ref, min_hw=min_hw, angle_tol_deg=5.0)
            for ref in (expected_refs if isinstance(expected_refs, list) else [])
        )
        if not matched:
            extras += 1
    return int(extras)


def _locate_p_geo(vertices, p_token, markers, min_hw):
    p_center = (float(p_token["center"][0]), float(p_token["center"][1]))
    edge_clear = scale_px(min_hw, 0.010, floor_px=0.0)
    vertex_excl = scale_px(min_hw, 0.05, floor_px=0.0)
    attach_tol = scale_px(min_hw, 0.11, floor_px=0.0)

    marker_cands = []
    for m in markers if isinstance(markers, list) else []:
        mp = (float(m[0]), float(m[1]))
        if min(_dist(mp, v) for v in vertices) <= vertex_excl:
            continue
        inside, clear = _point_strictly_inside_triangle(mp, vertices, edge_clear)
        if not inside:
            continue
        marker_cands.append((mp, float(clear), _dist(mp, p_center)))

    if marker_cands:
        marker_cands.sort(key=lambda t: (t[2], -t[1]))
        best_mp, _, best_d = marker_cands[0]
        if best_d <= attach_tol:
            return best_mp, ""
        return None, f"P label is not close to an interior marked point (dist={best_d:.1f}). "

    inside, clear = _point_strictly_inside_triangle(p_center, vertices, edge_clear)
    if not inside:
        return None, f"P is not strictly inside triangle ABC (clearance={clear:.1f}). "
    return p_center, ""


def judge_plane_44(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 3:
        return [False, "Insufficient line structure for case 44. "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCP", min_conf=0.08)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "P"], min_conf=0.08)
    missing_abc = [ch for ch in ["A", "B", "C"] if ch not in best_by_char]
    if missing_abc:
        return [False, f"Missing labels from OCR: {','.join(missing_abc)}. "]
    if "P" not in best_by_char:
        return [False, "Missing label P from OCR. "]

    tri_cands = extract_triangle_candidates_from_lines(
        lines=lines,
        img_shape=img.shape,
        min_len_ratio=0.14,
        top_k=14,
        min_angle_sep_deg=9.0,
        margin_ratio=0.20,
        point_tol_ratio=0.045,
        min_area_ratio=0.006,
        support_t=(-0.75, 1.35),
    )
    if not tri_cands:
        return [False, "Failed to reconstruct triangle candidates from detected lines. "]

    p_token = best_by_char["P"]
    markers = detect_marker_points(img)

    best_fail_stage = -1
    best_fail_reason = "No triangle candidate satisfies triangle-ABC, strict interior P and AP constraints."

    def _record_fail(stage, reason):
        nonlocal best_fail_stage, best_fail_reason
        if int(stage) > best_fail_stage:
            best_fail_stage = int(stage)
            best_fail_reason = str(reason)

    for cand in tri_cands:
        verts = cand["vertices"]
        assign, dists = assign_labels_to_vertices_min_cost(best_by_char, verts, ["A", "B", "C"])
        if assign is None or dists is None:
            _record_fail(1, "Failed to assign A/B/C to triangle vertices.")
            continue

        a_geo = verts[assign["A"]]
        b_geo = verts[assign["B"]]
        c_geo = verts[assign["C"]]
        tri_vertices = [a_geo, b_geo, c_geo]

        abc_tol = scale_px(min_hw, 0.16, floor_px=0.0)
        if any(float(dists[ch]) > abc_tol for ch in ["A", "B", "C"]):
            _record_fail(2, "A/B/C labels are too far from triangle vertices.")
            continue

        ok_ab, r_ab = has_segment_between_points(img, a_geo, b_geo, ratio_th=0.12, thickness=2, trim_ratio=0.02)
        ok_bc, r_bc = has_segment_between_points(img, b_geo, c_geo, ratio_th=0.12, thickness=2, trim_ratio=0.02)
        ok_ac, r_ac = has_segment_between_points(img, a_geo, c_geo, ratio_th=0.12, thickness=2, trim_ratio=0.02)
        if not (ok_ab and ok_bc and ok_ac):
            _record_fail(3, f"Outer triangle incomplete (AB={r_ab:.2f}, BC={r_bc:.2f}, AC={r_ac:.2f}). ")
            continue

        p_geo, p_reason = _locate_p_geo(
            vertices=tri_vertices,
            p_token=p_token,
            markers=markers,
            min_hw=min_hw,
        )
        if p_geo is None:
            _record_fail(4, p_reason)
            continue

        inside_ok, clear = _point_strictly_inside_triangle(
            point=p_geo,
            vertices=tri_vertices,
            edge_clearance=scale_px(min_hw, 0.010, floor_px=0.0),
        )
        if not inside_ok:
            _record_fail(5, f"P is not strictly inside triangle ABC (clearance={clear:.1f}). ")
            continue

        p_label_dist = token_edge_distance_to_point(p_token, p_geo)
        if p_label_dist > scale_px(min_hw, 0.12, floor_px=0.0):
            _record_fail(5, f"P label is too far from interior point marker (dist={p_label_dist:.1f}). ")
            continue

        ap_min_len = scale_px(min_hw, 0.06, floor_px=0.0)
        if _dist(a_geo, p_geo) < ap_min_len:
            _record_fail(5, "P is too close to A; segment AP is unstable.")
            continue

        line_ap = find_support_line(
            lines=lines,
            p1=a_geo,
            p2=p_geo,
            min_hw=min_hw,
            ang_tol_deg=13.0,
            dist_ratio=0.09,
        )
        ok_ap, r_ap = has_segment_between_points(img, a_geo, p_geo, ratio_th=0.10, thickness=2, trim_ratio=0.02)
        if not ok_ap and line_ap is not None:
            p_proj = _project_to_line_point(p_geo, line_ap)
            proj_inside, _ = _point_strictly_inside_triangle(
                point=p_proj,
                vertices=tri_vertices,
                edge_clearance=scale_px(min_hw, 0.008, floor_px=0.0),
            )
            if proj_inside:
                p_geo = p_proj
                ok_ap, r_ap = has_segment_between_points(
                    img,
                    a_geo,
                    p_geo,
                    ratio_th=0.10,
                    thickness=2,
                    trim_ratio=0.02,
                )

        if not ok_ap and line_ap is None:
            _record_fail(6, f"Missing segment AP from A to interior point P (ratio={r_ap:.2f}). ")
            continue

        ab = _segment_item(a_geo, b_geo)
        ac = _segment_item(a_geo, c_geo)
        bc = _segment_item(b_geo, c_geo)
        ap_ref = line_ap if line_ap is not None else _segment_item(a_geo, p_geo)
        if line_equivalent(ap_ref, ab, min_hw=min_hw, angle_tol_deg=5.0) or line_equivalent(
            ap_ref, ac, min_hw=min_hw, angle_tol_deg=5.0
        ):
            _record_fail(7, "Segment AP is not distinguishable from triangle side AB/AC.")
            continue

        expected_refs = [ab, ac, bc, ap_ref]
        extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
        if extra_lines > 2:
            _record_fail(8, f"Detected too many extra dominant lines ({extra_lines}). ")
            continue

        return [True, ""]

    return [False, best_fail_reason]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_44,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
