import argparse
import math

PID = 34
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


def _triangle_edges(vertices):
    if vertices is None or len(vertices) != 3:
        return []
    return [_segment_item(vertices[i], vertices[(i + 1) % 3]) for i in range(3)]


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
    radius_tol = scale_px(min_hw, 0.045, floor_px=0.0)
    if _dist((ca[0], ca[1]), (cb[0], cb[1])) > center_tol:
        return False
    return abs(float(ca[2]) - float(cb[2])) <= radius_tol


def _detect_circle_candidates(img, min_hw):
    h, w = img.shape[:2]
    min_r = int(round(scale_px(min_hw, 0.028, floor_px=1.0)))
    max_r = float(scale_px(min_hw, 0.34, floor_px=0.0))
    score_th = int(round(scale_px(min_hw, 0.085, floor_px=1.0)))
    band = int(max(1, round(scale_px(min_hw, 0.010, floor_px=0.0))))

    found, scores = _find_top_k_circles(
        img,
        k=24,
        min_r=min_r,
        max_r=0,
        seed=0,
        iters=3800,
        score_th=score_th,
    )

    x_margin = 0.12 * float(w)
    y_margin = 0.12 * float(h)
    raw = []
    for (cx, cy, r), s in zip(found, scores):
        x, y, rr = float(cx), float(cy), float(r)
        if rr <= 0.0 or rr < min_r or rr > max_r:
            continue
        if x < (-x_margin) or x > (float(w) + x_margin):
            continue
        if y < (-y_margin) or y > (float(h) + y_margin):
            continue

        refined = _refine_circle_radius_by_inner_outer_edges(img, (x, y, rr))
        if refined is None:
            continue
        rx, ry, rr2 = [float(v) for v in refined]
        if rr2 <= 0.0 or rr2 < min_r or rr2 > max_r:
            continue
        cov, vis = _circle_ink_coverage(img, (rx, ry), rr2, band)
        if vis < 0.50 or cov < 0.30:
            continue
        raw.append((rx, ry, rr2, int(s), float(cov), float(vis)))

    if raw:
        raw.sort(key=lambda t: (-t[3], -t[4], -t[2]))
        merged = _merge_circle_candidates(
            [(x, y, r, s) for (x, y, r, s, _, _) in raw],
            center_tol=scale_px(min_hw, 0.03, floor_px=0.0),
            radius_tol=scale_px(min_hw, 0.03, floor_px=0.0),
        )
        out = []
        for x, y, r, s in merged:
            cov, vis = _circle_ink_coverage(img, (x, y), r, band)
            if vis < 0.50 or cov < 0.30:
                continue
            out.append((float(x), float(y), float(r), int(s), float(cov), float(vis)))
        if out:
            out.sort(key=lambda t: (-t[3], -t[4], -t[2]))
            return out

    fallback = []
    for order in (1, 2, 3, 4):
        c = detect_circle(img, order=order, min_r=min_r, max_r=0)
        if c is None:
            continue
        x, y, r = [float(v) for v in c]
        if r <= 0.0 or r < min_r or r > max_r:
            continue
        cov, vis = _circle_ink_coverage(img, (x, y), r, band)
        if vis < 0.50 or cov < 0.30:
            continue
        fallback.append((x, y, r, 0, cov, vis))

    dedup = []
    for it in fallback:
        c = (it[0], it[1], it[2])
        if any(_circle_equivalent(c, (q[0], q[1], q[2]), min_hw=min_hw) for q in dedup):
            continue
        dedup.append(it)
    dedup.sort(key=lambda t: (-t[4], -t[2]))
    return dedup


def _count_extra_dominant_lines(lines, expected_lines, min_hw):
    long_th = scale_px(min_hw, 0.23, floor_px=0.0)
    extras = 0
    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        if float(ln.get("len", 0.0)) < long_th:
            continue
        matched = any(
            line_equivalent(ln, ref, min_hw=min_hw, angle_tol_deg=5.0)
            for ref in (expected_lines if isinstance(expected_lines, list) else [])
        )
        if not matched:
            extras += 1
    return int(extras)


def _circle_inside_triangle_stats(circle, vertices, min_hw):
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    center = (cx, cy)
    edges = _triangle_edges(vertices)
    if len(edges) != 3:
        return False, 0.0, 0.0

    center_tol = scale_px(min_hw, 0.010, floor_px=0.0)
    if not _point_inside_triangle_soft(center, vertices, tol_dist=center_tol):
        return False, 0.0, 0.0

    side_dists = [point_line_distance(center, e["abc"]) for e in edges]
    dmin = float(min(side_dists))
    clearance = float(dmin - r)
    cross_tol = scale_px(min_hw, 0.014, floor_px=0.0)
    if clearance < -cross_tol:
        return False, dmin, clearance
    return True, dmin, clearance


def judge_plane_34(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 3:
        return [False, "Insufficient line structure for a triangle. "]

    tri_cands = []
    for min_len_ratio, top_k, min_area_ratio, min_ang in (
        (0.13, 14, 0.008, 9.0),
        (0.11, 18, 0.006, 8.0),
        (0.09, 22, 0.003, 8.0),
    ):
        tri_cands.extend(
            extract_triangle_candidates_from_lines(
                lines=lines,
                img_shape=img.shape,
                min_len_ratio=min_len_ratio,
                top_k=min(int(top_k), len(lines)),
                min_angle_sep_deg=min_ang,
                margin_ratio=0.22,
                point_tol_ratio=0.045,
                min_area_ratio=min_area_ratio,
                support_t=(-0.75, 1.35),
            )
        )
    if not tri_cands:
        return [False, "Failed to reconstruct triangle candidates from detected lines. "]

    circle_cands = _detect_circle_candidates(img, min_hw=min_hw)
    if not circle_cands:
        return [False, "Failed to detect a valid circle candidate. "]

    best = None
    best_score = None

    for tri in tri_cands[:10]:
        verts = tri["vertices"]
        if len(verts) != 3:
            continue

        ok_ab, r_ab = has_segment_between_points(img, verts[0], verts[1], ratio_th=0.11, thickness=2, trim_ratio=0.02)
        ok_bc, r_bc = has_segment_between_points(img, verts[1], verts[2], ratio_th=0.11, thickness=2, trim_ratio=0.02)
        ok_ca, r_ca = has_segment_between_points(img, verts[2], verts[0], ratio_th=0.11, thickness=2, trim_ratio=0.02)
        if not (ok_ab and ok_bc and ok_ca):
            continue

        tri_score = float(tri.get("score", 0.0))
        tri_area = float(tri.get("area", 0.0))

        for cx, cy, r, c_score, cov, vis in circle_cands:
            inside_ok, dmin, clearance = _circle_inside_triangle_stats(
                (cx, cy, r),
                verts,
                min_hw=min_hw,
            )
            if not inside_ok:
                continue
            rel_clear = clearance / max(1.0, r)
            score = (
                0.0010 * tri_area
                + 0.010 * tri_score
                + 18.0 * float(cov)
                + 10.0 * float(vis)
                + 0.020 * float(c_score)
                + 4.0 * float(rel_clear)
                + 0.015 * float(r)
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "triangle": tri,
                    "circle": (float(cx), float(cy), float(r), float(cov), float(vis)),
                    "clearance": float(clearance),
                    "dmin": float(dmin),
                    "side_ratios": (float(r_ab), float(r_bc), float(r_ca)),
                }

    if best is None:
        return [False, "No circle is fully inside any detected triangle. "]

    tri = best["triangle"]
    verts = tri["vertices"]
    cx, cy, r, cov, vis = best["circle"]
    clearance = float(best["clearance"])
    dmin = float(best["dmin"])

    if vis < 0.50 or cov < 0.30:
        return [False, f"Circle trace is too weak (cov={cov:.2f}, vis={vis:.2f}). "]

    if clearance < -scale_px(min_hw, 0.014, floor_px=0.0):
        return [False, f"Circle crosses triangle boundary (clearance={clearance:.1f}). "]

    if dmin <= scale_px(min_hw, 0.012, floor_px=0.0):
        return [False, "Circle center is too close to triangle boundary. "]

    extra_lines = _count_extra_dominant_lines(lines, expected_lines=tri["lines"], min_hw=min_hw)
    if extra_lines > 2:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    selected = (cx, cy, r)
    extra_circles = 0
    for cand in circle_cands:
        c = (float(cand[0]), float(cand[1]), float(cand[2]))
        if _circle_equivalent(c, selected, min_hw=min_hw):
            continue
        if c[2] < max(scale_px(min_hw, 0.030, floor_px=0.0), 0.65 * r):
            continue
        if _dist((c[0], c[1]), (cx, cy)) <= scale_px(min_hw, 0.06, floor_px=0.0):
            continue
        extra_circles += 1

    if extra_circles > 0:
        return [False, f"Detected extra prominent circle(s): {extra_circles}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_34,
        require_ocr=False,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
