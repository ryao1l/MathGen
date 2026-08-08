import argparse
import math

PID = 48
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
        raise RuntimeError("Failed to build segment line from points.")
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _triangle_metrics(vertices):
    if not isinstance(vertices, list) or len(vertices) != 3:
        return None
    edges = [_dist(vertices[i], vertices[(i + 1) % 3]) for i in range(3)]
    if any(float(e) <= 1e-6 for e in edges):
        return None
    x1, y1 = float(vertices[0][0]), float(vertices[0][1])
    x2, y2 = float(vertices[1][0]), float(vertices[1][1])
    x3, y3 = float(vertices[2][0]), float(vertices[2][1])
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    center = ((x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0)
    return {
        "edges": edges,
        "min_edge": float(min(edges)),
        "max_edge": float(max(edges)),
        "area": float(area),
        "center": center,
    }


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
    center = (
        sum(float(v[0]) for v in vertices) / 4.0,
        sum(float(v[1]) for v in vertices) / 4.0,
    )
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "opp_rel_1": float(opp_rel_1),
        "opp_rel_2": float(opp_rel_2),
        "width": float(width),
        "height": float(height),
        "min_edge": float(min(edges)),
        "diag": float(diag),
        "center": center,
    }


def _point_inside_convex_polygon_soft(point, vertices, tol_dist):
    if vertices is None or len(vertices) < 3:
        return False
    px, py = float(point[0]), float(point[1])

    area2 = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % n][0]), float(vertices[(i + 1) % n][1])
        area2 += x1 * y2 - x2 * y1
    orient = 1.0 if area2 >= 0.0 else -1.0

    for i in range(n):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % n][0]), float(vertices[(i + 1) % n][1])
        ex, ey = x2 - x1, y2 - y1
        edge_len = math.hypot(ex, ey)
        if edge_len <= 1e-6:
            return False
        cross = ex * (py - y1) - ey * (px - x1)
        if orient * cross < -float(edge_len) * float(tol_dist):
            return False
    return True


def _point_inside_triangle_soft(point, vertices, tol_dist):
    if vertices is None or len(vertices) != 3:
        return False
    return _point_inside_convex_polygon_soft(point, vertices, tol_dist=tol_dist)


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


def _segment_intersect_or_touch(p1, p2, q1, q2, tol):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(q1[0]), float(q1[1])
    x4, y4 = float(q2[0]), float(q2[1])

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - tol <= c[0] <= max(a[0], b[0]) + tol
            and min(a[1], b[1]) - tol <= c[1] <= max(a[1], b[1]) + tol
        )

    a = (x1, y1)
    b = (x2, y2)
    c = (x3, y3)
    d = (x4, y4)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    cross_hit = (o1 * o2 < -tol * tol) and (o3 * o4 < -tol * tol)
    if cross_hit:
        return True

    if abs(o1) <= tol and on_segment(a, b, c):
        return True
    if abs(o2) <= tol and on_segment(a, b, d):
        return True
    if abs(o3) <= tol and on_segment(c, d, a):
        return True
    if abs(o4) <= tol and on_segment(c, d, b):
        return True
    return False


def _segment_distance(p1, p2, q1, q2, tol):
    if _segment_intersect_or_touch(p1, p2, q1, q2, tol=tol):
        return 0.0
    d1 = _point_to_segment_distance(p1, q1, q2)
    d2 = _point_to_segment_distance(p2, q1, q2)
    d3 = _point_to_segment_distance(q1, p1, p2)
    d4 = _point_to_segment_distance(q2, p1, p2)
    return float(min(d1, d2, d3, d4))


def _triangle_equivalent(a, b, min_hw):
    if a is None or b is None:
        return False
    center_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    if _dist(a["metrics"]["center"], b["metrics"]["center"]) > center_tol:
        return False
    area_a = float(a["metrics"]["area"])
    area_b = float(b["metrics"]["area"])
    return abs(area_a - area_b) <= 0.18 * max(1.0, area_a, area_b)


def _rectangle_equivalent(a, b, min_hw):
    if a is None or b is None:
        return False
    center_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    diag_tol = scale_px(min_hw, 0.05, floor_px=0.0)
    if _dist(a["metrics"]["center"], b["metrics"]["center"]) > center_tol:
        return False
    return abs(float(a["metrics"]["diag"]) - float(b["metrics"]["diag"])) <= diag_tol


def _extract_triangle_candidates(img, lines, min_hw):
    if not lines:
        return []

    base_lines = [ln for ln in lines if isinstance(ln, dict)]
    pool_drop_cap = min(5, max(0, len(base_lines) - 3))
    pools = [base_lines]
    for drop_idx in range(pool_drop_cap):
        pools.append([ln for j, ln in enumerate(base_lines) if j != drop_idx])

    configs = [
        (0.15, 14, 0.010, 10.0),
        (0.13, 16, 0.008, 9.0),
        (0.11, 18, 0.006, 8.0),
        (0.09, 22, 0.003, 8.0),
    ]
    cands = []
    min_edge_th = scale_px(min_hw, 0.08, floor_px=0.0)
    for pool in pools:
        for min_len_ratio, top_k, min_area_ratio, min_ang in configs:
            tri_list = extract_triangle_candidates_from_lines(
                lines=pool,
                img_shape=img.shape,
                min_len_ratio=min_len_ratio,
                top_k=min(int(top_k), len(pool)),
                min_angle_sep_deg=min_ang,
                margin_ratio=0.22,
                point_tol_ratio=0.045,
                min_area_ratio=min_area_ratio,
                support_t=(-0.75, 1.35),
            )
            for tri in tri_list[:10]:
                vertices = tri["vertices"]
                metrics = _triangle_metrics(vertices)
                if metrics is None:
                    continue
                if metrics["min_edge"] < min_edge_th:
                    continue

                edge_items = [_segment_item(vertices[i], vertices[(i + 1) % 3]) for i in range(3)]
                edge_ratios = []
                edges_ok = True
                for i in range(3):
                    p = vertices[i]
                    q = vertices[(i + 1) % 3]
                    ok, ratio = has_segment_between_points(img, p, q, ratio_th=0.10, thickness=2, trim_ratio=0.02)
                    if not ok:
                        side_line = find_support_line(lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08)
                        if side_line is None:
                            edges_ok = False
                            break
                    edge_ratios.append(float(ratio))
                if not edges_ok:
                    continue

                cand = {
                    "vertices": vertices,
                    "lines": tri["lines"],
                    "edge_items": edge_items,
                    "edge_ratios": edge_ratios,
                    "metrics": metrics,
                    "area": float(tri.get("area", 0.0)),
                    "score": float(tri.get("score", 0.0)),
                }
                if any(_triangle_equivalent(cand, ex, min_hw=min_hw) for ex in cands):
                    continue
                cands.append(cand)

    cands.sort(
        key=lambda t: (
            t["area"],
            min(t["edge_ratios"]) if t["edge_ratios"] else 0.0,
            t["score"],
        ),
        reverse=True,
    )
    return cands


def _extract_rectangle_candidates(img, lines, min_hw):
    if not lines:
        return []

    base_lines = [ln for ln in lines if isinstance(ln, dict)]
    if len(base_lines) < 4:
        return []
    pool_drop_cap = min(6, max(0, len(base_lines) - 4))
    pools = [base_lines]
    for drop_idx in range(pool_drop_cap):
        pools.append([ln for j, ln in enumerate(base_lines) if j != drop_idx])

    configs = [
        (0.18, 10, 0.010, 12.0),
        (0.15, 12, 0.008, 10.0),
        (0.12, 16, 0.005, 9.0),
        (0.09, 22, 0.002, 8.0),
    ]
    cands = []
    min_edge_th = scale_px(min_hw, 0.08, floor_px=0.0)
    for pool in pools:
        for min_len_ratio, top_k, min_area_ratio, min_ang in configs:
            quad = extract_polygon_from_lines(
                lines=pool,
                img_shape=img.shape,
                sides=4,
                min_len_ratio=min_len_ratio,
                top_k=min(int(top_k), len(pool)),
                min_angle_sep_deg=min_ang,
                margin_ratio=0.20,
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
            if max(metrics["cos_vals"]) > 0.26:
                continue
            if metrics["opp_rel_1"] > 0.26 or metrics["opp_rel_2"] > 0.26:
                continue
            if metrics["min_edge"] < min_edge_th:
                continue

            edge_items = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
            edge_ratios = []
            edges_ok = True
            for i in range(4):
                p = vertices[i]
                q = vertices[(i + 1) % 4]
                ok, ratio = has_segment_between_points(img, p, q, ratio_th=0.10, thickness=2, trim_ratio=0.02)
                if not ok:
                    side_line = find_support_line(lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.08)
                    if side_line is None:
                        edges_ok = False
                        break
                edge_ratios.append(float(ratio))
            if not edges_ok:
                continue

            cand = {
                "vertices": vertices,
                "lines": quad["lines"],
                "edge_items": edge_items,
                "edge_ratios": edge_ratios,
                "metrics": metrics,
                "area": float(quad.get("area", 0.0)),
                "score": float(quad.get("score", 0.0)),
            }
            if any(_rectangle_equivalent(cand, ex, min_hw=min_hw) for ex in cands):
                continue
            cands.append(cand)

    cands.sort(
        key=lambda t: (
            t["area"],
            min(t["edge_ratios"]) if t["edge_ratios"] else 0.0,
            t["score"],
        ),
        reverse=True,
    )
    return cands


def _triangle_rectangle_disjoint_stats(tri_vertices, rect_vertices, min_hw):
    if tri_vertices is None or rect_vertices is None:
        return False, {"reason": "Missing polygon vertices."}
    if len(tri_vertices) != 3 or len(rect_vertices) != 4:
        return False, {"reason": "Invalid polygon vertex counts."}

    inside_tol = scale_px(min_hw, 0.006, floor_px=0.0)
    for p in tri_vertices:
        if _point_inside_convex_polygon_soft(p, rect_vertices, tol_dist=inside_tol):
            return False, {"reason": "Triangle overlaps rectangle interior."}
    for p in rect_vertices:
        if _point_inside_triangle_soft(p, tri_vertices, tol_dist=inside_tol):
            return False, {"reason": "Rectangle overlaps triangle interior."}

    tri_edges = [(tri_vertices[i], tri_vertices[(i + 1) % 3]) for i in range(3)]
    rect_edges = [(rect_vertices[i], rect_vertices[(i + 1) % 4]) for i in range(4)]
    touch_tol = scale_px(min_hw, 0.003, floor_px=0.0)

    min_gap = None
    for p1, p2 in tri_edges:
        for q1, q2 in rect_edges:
            if _segment_intersect_or_touch(p1, p2, q1, q2, tol=touch_tol):
                return False, {"reason": "Triangle and rectangle are intersecting/touching."}
            d = _segment_distance(p1, p2, q1, q2, tol=touch_tol)
            if min_gap is None or d < min_gap:
                min_gap = float(d)

    if min_gap is None:
        return False, {"reason": "Failed to compute polygon separation."}
    if min_gap <= touch_tol:
        return False, {"reason": "Triangle and rectangle are too close to be disjoint."}

    return True, {"min_gap": float(min_gap), "touch_tol": float(touch_tol)}


def _count_extra_dominant_lines(lines, expected_refs, min_hw):
    long_th = scale_px(min_hw, 0.22, floor_px=0.0)
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


def judge_plane_48(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.05)
    if len(lines) < 6:
        return [False, "Insufficient line structure for triangle-rectangle task. "]

    tri_cands = _extract_triangle_candidates(img=img, lines=lines, min_hw=min_hw)
    if not tri_cands:
        return [False, "Failed to reconstruct triangle candidate. "]

    rect_cands = _extract_rectangle_candidates(img=img, lines=lines, min_hw=min_hw)
    if not rect_cands:
        return [False, "Failed to reconstruct rectangle candidate. "]

    best = None
    best_score = None
    best_reason = "No triangle-rectangle pair satisfies strict non-overlap constraints. "

    for tri in tri_cands[:20]:
        tri_vertices = tri["vertices"]
        for rect in rect_cands[:20]:
            rect_vertices = rect["vertices"]

            ok_disjoint, stats = _triangle_rectangle_disjoint_stats(
                tri_vertices=tri_vertices,
                rect_vertices=rect_vertices,
                min_hw=min_hw,
            )
            if not ok_disjoint:
                if "reason" in stats:
                    best_reason = str(stats["reason"]) + " "
                continue

            expected_refs = list(tri["edge_items"]) + list(rect["edge_items"])
            extra_lines = _count_extra_dominant_lines(lines, expected_refs=expected_refs, min_hw=min_hw)
            if extra_lines > 3:
                best_reason = f"Detected too many extra dominant lines ({extra_lines}). "
                continue

            score = (
                0.0010 * (float(tri["area"]) + float(rect["area"]))
                + 1.4 * (min(tri["edge_ratios"]) + min(rect["edge_ratios"]))
                + 0.20 * (float(stats["min_gap"]) / max(1.0, min_hw))
                - 0.15 * float(extra_lines)
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "triangle": tri,
                    "rectangle": rect,
                    "stats": stats,
                    "extra_lines": int(extra_lines),
                }

    if best is None:
        return [False, best_reason]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_48,
        require_ocr=False,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
