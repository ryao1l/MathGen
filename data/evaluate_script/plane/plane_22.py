import argparse
import math

PID = 47
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

    mean_edge = sum(float(e) for e in edges) / 4.0
    edge_rel = (max(edges) - min(edges)) / max(1e-6, mean_edge)
    diag = max(_dist(vertices[0], vertices[2]), _dist(vertices[1], vertices[3]))
    center = (
        sum(float(v[0]) for v in vertices) / 4.0,
        sum(float(v[1]) for v in vertices) / 4.0,
    )
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "mean_edge": float(mean_edge),
        "edge_rel": float(edge_rel),
        "diag": float(diag),
        "min_edge": float(min(edges)),
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


def _square_equivalent(a, b, min_hw):
    if a is None or b is None:
        return False
    if _dist(a["metrics"]["center"], b["metrics"]["center"]) > scale_px(min_hw, 0.05, floor_px=0.0):
        return False
    if abs(a["metrics"]["diag"] - b["metrics"]["diag"]) > scale_px(min_hw, 0.05, floor_px=0.0):
        return False
    return True


def _extract_square_candidates(img, lines, min_hw, min_edge_ratio):
    if not lines:
        return []

    configs = [
        (0.22, 14, 0.012, 12.0),
        (0.18, 16, 0.008, 11.0),
        (0.14, 18, 0.005, 10.0),
        (0.11, 22, 0.003, 9.0),
        (0.08, 26, 0.0015, 8.0),
        (0.06, 30, 0.0010, 8.0),
    ]
    cands = []
    for min_len_ratio, top_k, min_area_ratio, min_ang in configs:
        quad = extract_polygon_from_lines(
            lines=lines,
            img_shape=img.shape,
            sides=4,
            min_len_ratio=min_len_ratio,
            top_k=top_k,
            min_angle_sep_deg=min_ang,
            margin_ratio=0.20,
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
        if max(metrics["cos_vals"]) > 0.30:
            continue
        if metrics["edge_rel"] > 0.25:
            continue
        if metrics["min_edge"] < scale_px(min_hw, min_edge_ratio, floor_px=0.0):
            continue

        edge_items = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
        edge_ratios = []
        edge_ok = True
        for i in range(4):
            ok, ratio = has_segment_between_points(
                img,
                vertices[i],
                vertices[(i + 1) % 4],
                ratio_th=0.11,
                thickness=2,
                trim_ratio=0.02,
            )
            edge_ratios.append(float(ratio))
            if not ok:
                edge_ok = False
        if not edge_ok:
            continue

        cand = {
            "vertices": vertices,
            "lines": quad["lines"],
            "edge_items": edge_items,
            "metrics": metrics,
            "area": float(quad.get("area", 0.0)),
            "score": float(quad.get("score", 0.0)),
            "edge_ratios": edge_ratios,
        }

        duplicate = any(_square_equivalent(cand, ex, min_hw=min_hw) for ex in cands)
        if duplicate:
            continue
        cands.append(cand)

    cands.sort(
        key=lambda c: (
            c["area"],
            min(c["edge_ratios"]) if c["edge_ratios"] else 0.0,
            c["score"],
        ),
        reverse=True,
    )
    return cands


def _extract_axis_aligned_square_candidates(img, lines, min_hw, min_edge_ratio):
    if not lines:
        return []

    min_len = scale_px(min_hw, min_edge_ratio, floor_px=0.0)
    horizontals = []
    verticals = []
    for ln in lines:
        if float(ln.get("len", 0.0)) < min_len:
            continue
        if horizontal_error_deg(float(ln["ang"])) <= 12.0:
            horizontals.append(ln)
        elif angle_diff_deg(float(ln["ang"]), 90.0) <= 12.0:
            verticals.append(ln)

    horizontals = sorted(horizontals, key=lambda ln: float(ln.get("len", 0.0)), reverse=True)[:14]
    verticals = sorted(verticals, key=lambda ln: float(ln.get("len", 0.0)), reverse=True)[:14]

    out = []
    for i in range(len(horizontals)):
        for j in range(i + 1, len(horizontals)):
            h1 = horizontals[i]
            h2 = horizontals[j]
            y1 = 0.5 * (float(h1["seg"][1]) + float(h1["seg"][3]))
            y2 = 0.5 * (float(h2["seg"][1]) + float(h2["seg"][3]))
            if abs(y1 - y2) < min_len:
                continue
            y_top, y_bot = sorted([y1, y2])

            for a in range(len(verticals)):
                for b in range(a + 1, len(verticals)):
                    v1 = verticals[a]
                    v2 = verticals[b]
                    x1 = 0.5 * (float(v1["seg"][0]) + float(v1["seg"][2]))
                    x2 = 0.5 * (float(v2["seg"][0]) + float(v2["seg"][2]))
                    if abs(x1 - x2) < min_len:
                        continue
                    x_left, x_right = sorted([x1, x2])
                    width = x_right - x_left
                    height = y_bot - y_top
                    if width < min_len or height < min_len:
                        continue
                    if max(float(h1["len"]), float(h2["len"])) > 1.38 * width + scale_px(min_hw, 0.02, floor_px=0.0):
                        continue
                    if max(float(v1["len"]), float(v2["len"])) > 1.38 * height + scale_px(min_hw, 0.02, floor_px=0.0):
                        continue
                    side_rel = abs(width - height) / max(1e-6, 0.5 * (width + height))
                    if side_rel > 0.28:
                        continue

                    vertices = [(x_left, y_top), (x_right, y_top), (x_right, y_bot), (x_left, y_bot)]
                    edge_ratios = []
                    edge_ok = True
                    for k in range(4):
                        ok, ratio = has_segment_between_points(
                            img,
                            vertices[k],
                            vertices[(k + 1) % 4],
                            ratio_th=0.06,
                            thickness=2,
                            trim_ratio=0.02,
                        )
                        edge_ratios.append(float(ratio))
                        if not ok:
                            edge_ok = False
                    if not edge_ok:
                        continue

                    metrics = _square_metrics(vertices)
                    if metrics is None:
                        continue
                    area = float(width * height)
                    cand = {
                        "vertices": vertices,
                        "lines": [h1, h2, v1, v2],
                        "edge_items": [_segment_item(vertices[k], vertices[(k + 1) % 4]) for k in range(4)],
                        "metrics": metrics,
                        "area": area,
                        "score": float(sum(edge_ratios)),
                        "edge_ratios": edge_ratios,
                    }
                    if any(_square_equivalent(cand, ex, min_hw=min_hw) for ex in out):
                        continue
                    out.append(cand)

    # Some accepted renderings leave one side of the smaller square very faint.
    # Recover a square from two vertical sides plus either its top or bottom edge.
    for a in range(len(verticals)):
        for b in range(a + 1, len(verticals)):
            v1 = verticals[a]
            v2 = verticals[b]
            x1 = 0.5 * (float(v1["seg"][0]) + float(v1["seg"][2]))
            x2 = 0.5 * (float(v2["seg"][0]) + float(v2["seg"][2]))
            if abs(x1 - x2) < min_len:
                continue
            x_left, x_right = sorted([x1, x2])
            v1y = sorted([float(v1["seg"][1]), float(v1["seg"][3])])
            v2y = sorted([float(v2["seg"][1]), float(v2["seg"][3])])
            y_top = 0.5 * (v1y[0] + v2y[0])
            y_bot = 0.5 * (v1y[1] + v2y[1])
            width = x_right - x_left
            height = y_bot - y_top
            if width < min_len or height < min_len:
                continue
            side_rel = abs(width - height) / max(1e-6, 0.5 * (width + height))
            if side_rel > 0.30:
                continue
            if max(float(v1["len"]), float(v2["len"])) > 1.38 * height + scale_px(min_hw, 0.02, floor_px=0.0):
                continue

            vertices = [(x_left, y_top), (x_right, y_top), (x_right, y_bot), (x_left, y_bot)]
            ok_left, r_left = has_segment_between_points(img, vertices[3], vertices[0], ratio_th=0.06, thickness=2, trim_ratio=0.02)
            ok_right, r_right = has_segment_between_points(img, vertices[1], vertices[2], ratio_th=0.06, thickness=2, trim_ratio=0.02)
            ok_top, r_top = has_segment_between_points(img, vertices[0], vertices[1], ratio_th=0.06, thickness=2, trim_ratio=0.02)
            ok_bot, r_bot = has_segment_between_points(img, vertices[2], vertices[3], ratio_th=0.06, thickness=2, trim_ratio=0.02)
            if not (ok_left and ok_right and (ok_top or ok_bot)):
                continue
            edge_ratios = [float(r_top), float(r_right), float(r_bot), float(r_left)]
            metrics = _square_metrics(vertices)
            if metrics is None:
                continue
            cand = {
                "vertices": vertices,
                "lines": [v1, v2],
                "edge_items": [_segment_item(vertices[k], vertices[(k + 1) % 4]) for k in range(4)],
                "metrics": metrics,
                "area": float(width * height),
                "score": float(sum(edge_ratios)),
                "edge_ratios": edge_ratios,
            }
            if any(_square_equivalent(cand, ex, min_hw=min_hw) for ex in out):
                continue
            out.append(cand)

    out.sort(
        key=lambda c: (
            c["area"],
            min(c["edge_ratios"]) if c["edge_ratios"] else 0.0,
            c["score"],
        ),
        reverse=True,
    )
    return out


def _remove_equivalent_lines(lines, refs, min_hw):
    out = []
    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        matched = any(line_equivalent(ln, ref, min_hw=min_hw, angle_tol_deg=4.5) for ref in refs)
        if matched:
            continue
        out.append(ln)
    return out


def _collect_square_candidates(img, lines, min_hw):
    axis = _extract_axis_aligned_square_candidates(
        img=img,
        lines=lines,
        min_hw=min_hw,
        min_edge_ratio=0.05,
    )
    if len(lines) > 35:
        return axis

    primary = _extract_square_candidates(
        img=img,
        lines=lines,
        min_hw=min_hw,
        min_edge_ratio=0.05,
    )
    merged = list(primary)
    for cand in axis:
        if any(_square_equivalent(cand, ex, min_hw=min_hw) for ex in merged):
            continue
        merged.append(cand)
    for base in primary[:5]:
        base_refs = list(base["lines"]) + list(base["edge_items"])
        residual = _remove_equivalent_lines(lines, refs=base_refs, min_hw=min_hw)
        if len(residual) > 30:
            continue
        extras = _extract_square_candidates(
            img=img,
            lines=residual,
            min_hw=min_hw,
            min_edge_ratio=0.05,
        )
        for cand in extras:
            if any(_square_equivalent(cand, ex, min_hw=min_hw) for ex in merged):
                continue
            merged.append(cand)

    merged.sort(
        key=lambda c: (
            c["area"],
            min(c["edge_ratios"]) if c["edge_ratios"] else 0.0,
            c["score"],
        ),
        reverse=True,
    )
    return merged


def _square_pair_disjoint_stats(sq1, sq2, min_hw):
    v1 = sq1["vertices"]
    v2 = sq2["vertices"]
    if not v1 or not v2:
        return False, {"reason": "Invalid square vertices."}

    m1 = float(sq1["metrics"]["mean_edge"])
    m2 = float(sq2["metrics"]["mean_edge"])
    larger = max(m1, m2)
    smaller = min(m1, m2)
    area_ratio = min(float(sq1["area"]), float(sq2["area"])) / max(1e-6, max(float(sq1["area"]), float(sq2["area"])))
    edge_sep = abs(m1 - m2)
    edge_sep_th = scale_px(min_hw, 0.025, floor_px=0.0)
    if edge_sep < edge_sep_th and area_ratio > 0.90:
        return False, {"reason": "The two squares are not clearly different in size."}
    if smaller >= 0.95 * larger:
        return False, {"reason": "The two squares are nearly the same size."}

    inside_tol = scale_px(min_hw, 0.006, floor_px=0.0)
    for p in v1:
        if _point_inside_convex_polygon_soft(p, v2, tol_dist=inside_tol):
            return False, {"reason": "Squares overlap or one is inside the other."}
    for p in v2:
        if _point_inside_convex_polygon_soft(p, v1, tol_dist=inside_tol):
            return False, {"reason": "Squares overlap or one is inside the other."}

    touch_tol = scale_px(min_hw, 0.003, floor_px=0.0)
    min_gap = None
    for i in range(4):
        a1 = v1[i]
        a2 = v1[(i + 1) % 4]
        for j in range(4):
            b1 = v2[j]
            b2 = v2[(j + 1) % 4]
            if _segment_intersect_or_touch(a1, a2, b1, b2, tol=touch_tol):
                return False, {"reason": "The two squares are intersecting or touching."}
            d = _segment_distance(a1, a2, b1, b2, tol=touch_tol)
            if min_gap is None or d < min_gap:
                min_gap = float(d)

    if min_gap is None:
        return False, {"reason": "Failed to compute square separation."}
    if min_gap <= touch_tol:
        return False, {"reason": "The two squares are too close to be disjoint."}

    center_dist = _dist(sq1["metrics"]["center"], sq2["metrics"]["center"])
    return True, {
        "min_gap": float(min_gap),
        "center_dist": float(center_dist),
        "edge_sep": float(edge_sep),
        "area_ratio": float(area_ratio),
    }


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


def judge_plane_47(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.05)
    if len(lines) < 6:
        return [False, "Insufficient line structure for two-square task. "]
    long_lines = sum(1 for ln in lines if float(ln.get("len", 0.0)) > 0.22 * min_hw)
    if long_lines > 14:
        return [False, f"Detected too many dominant lines for exactly two squares ({long_lines}). "]
    if len(lines) > 45:
        return [False, f"Detected overly cluttered line structure for exactly two squares ({len(lines)} lines). "]

    square_candidates = _collect_square_candidates(img=img, lines=lines, min_hw=min_hw)
    if len(square_candidates) < 2:
        return [False, "Failed to reconstruct two square candidates. "]

    best_pair = None
    best_score = None
    best_reason = "Failed to find two disjoint squares with distinct sizes. "

    for i in range(len(square_candidates)):
        for j in range(i + 1, len(square_candidates)):
            sq1 = square_candidates[i]
            sq2 = square_candidates[j]
            if _square_equivalent(sq1, sq2, min_hw=min_hw):
                continue

            ok, stats = _square_pair_disjoint_stats(sq1, sq2, min_hw=min_hw)
            if not ok:
                if "reason" in stats:
                    best_reason = str(stats["reason"]) + " "
                continue

            score = (
                0.0040 * (float(sq1["area"]) + float(sq2["area"]))
                + 1.5 * (min(sq1["edge_ratios"]) + min(sq2["edge_ratios"]))
                + 0.35 * (float(stats["edge_sep"]) / max(1.0, min_hw))
                + 0.12 * (float(stats["min_gap"]) / max(1.0, min_hw))
            )
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (sq1, sq2)

    if best_pair is None:
        return [False, best_reason]

    sq1, sq2 = best_pair
    expected_refs = list(sq1["edge_items"]) + list(sq2["edge_items"])
    extra_lines = _count_extra_dominant_lines(lines, expected_refs=expected_refs, min_hw=min_hw)
    if extra_lines > 3:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    residual_lines = _remove_equivalent_lines(lines, refs=expected_refs, min_hw=min_hw)
    extra_square_candidates = _extract_square_candidates(
        img=img,
        lines=residual_lines,
        min_hw=min_hw,
        min_edge_ratio=0.05,
    )
    small_area = min(float(sq1["area"]), float(sq2["area"]))
    for ex in extra_square_candidates:
        if _square_equivalent(ex, sq1, min_hw=min_hw) or _square_equivalent(ex, sq2, min_hw=min_hw):
            continue
        if float(ex["area"]) > 0.25 * small_area:
            return [False, "Detected extra square-like boundary beyond required two squares. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_47,
        require_ocr=False,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
