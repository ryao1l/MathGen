import argparse
import math

PID = 39
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
        raise RuntimeError("Failed to build rectangle edge line.")
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _rectangle_metrics(vertices):
    if not isinstance(vertices, list) or len(vertices) != 4:
        return None

    edges = [_dist(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    if any(e <= 1e-6 for e in edges):
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
    diag = max(_dist(vertices[0], vertices[2]), _dist(vertices[1], vertices[3]))
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "opp_rel_1": float(opp_rel_1),
        "opp_rel_2": float(opp_rel_2),
        "min_edge": float(min(edges)),
        "diag": float(diag),
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
    center_tol = scale_px(min_hw, 0.04, floor_px=0.0)
    radius_tol = scale_px(min_hw, 0.04, floor_px=0.0)
    if _dist((ca[0], ca[1]), (cb[0], cb[1])) > center_tol:
        return False
    return abs(float(ca[2]) - float(cb[2])) <= radius_tol


def _detect_circle_candidates(img, min_hw):
    h, w = img.shape[:2]
    min_r = int(round(scale_px(min_hw, 0.07, floor_px=1.0)))
    max_r = float(scale_px(min_hw, 0.62, floor_px=0.0))
    score_th = int(round(scale_px(min_hw, 0.09, floor_px=1.0)))
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

    x_margin = 0.15 * float(w)
    y_margin = 0.15 * float(h)
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
    for item in fallback:
        c = (item[0], item[1], item[2])
        if any(_circle_equivalent(c, (q[0], q[1], q[2]), min_hw=min_hw) for q in dedup):
            continue
        dedup.append(item)
    dedup.sort(key=lambda t: (-t[4], -t[2]))
    return dedup


def _extract_rectangle_candidates(img, lines, min_hw):
    configs = [
        (0.17, 12, 0.010, 10.0),
        (0.14, 14, 0.008, 9.0),
        (0.12, 16, 0.006, 8.0),
        (0.10, 20, 0.0035, 7.5),
        (0.08, 24, 0.0020, 7.0),
    ]
    candidates = []

    for min_len_ratio, top_k, min_area_ratio, min_angle_sep in configs:
        quad = extract_polygon_from_lines(
            lines=lines,
            img_shape=img.shape,
            sides=4,
            min_len_ratio=min_len_ratio,
            top_k=top_k,
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
        if max(metrics["cos_vals"]) > 0.26:
            continue
        if metrics["opp_rel_1"] > 0.28 or metrics["opp_rel_2"] > 0.28:
            continue
        if metrics["min_edge"] < scale_px(min_hw, 0.10, floor_px=0.0):
            continue

        edge_ratios = []
        edges_ok = True
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
                edges_ok = False
        if not edges_ok:
            continue

        center = (
            sum(float(v[0]) for v in vertices) / 4.0,
            sum(float(v[1]) for v in vertices) / 4.0,
        )
        duplicate = False
        for ex in candidates:
            if _dist(center, ex["center"]) > scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            if abs(metrics["diag"] - ex["metrics"]["diag"]) > scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            duplicate = True
            break
        if duplicate:
            continue

        candidates.append(
            {
                "vertices": vertices,
                "lines": quad["lines"],
                "metrics": metrics,
                "area": float(quad.get("area", 0.0)),
                "score": float(quad.get("score", 0.0)),
                "edge_ratios": edge_ratios,
                "center": center,
            }
        )

    candidates.sort(
        key=lambda t: (
            min(t["edge_ratios"]) if t["edge_ratios"] else 0.0,
            t["area"],
            t["score"],
        ),
        reverse=True,
    )
    return candidates


def _select_rectangle_inside_circle(rectangles, circles, min_hw):
    if not rectangles:
        return None, "Failed to reconstruct a valid rectangle. "
    if not circles:
        return None, "Failed to detect a valid circle. "

    contain_tol = scale_px(min_hw, 0.030, floor_px=0.0)
    strict_gap = scale_px(min_hw, 0.002, floor_px=0.0)
    best = None
    best_score = None

    for rect in rectangles:
        vertices = rect["vertices"]
        diag = float(rect["metrics"]["diag"])
        area_norm = float(rect["area"]) / max(1.0, float(min_hw) * float(min_hw))
        edge_min_ratio = min(rect["edge_ratios"]) if rect["edge_ratios"] else 0.0

        for cx, cy, r, c_score, cov, vis in circles:
            if float(r) < 0.45 * diag:
                continue

            dists = [_dist(v, (cx, cy)) for v in vertices]
            max_d = max(dists)
            clearance = float(r) - float(max_d)
            avg_gap = sum(float(r) - float(d) for d in dists) / 4.0
            inside_count = sum(1 for d in dists if d <= float(r) + contain_tol)
            deep_inside_count = sum(1 for d in dists if d <= float(r) - strict_gap)

            if inside_count < 4:
                continue
            if clearance < -contain_tol:
                continue
            if avg_gap < -0.5 * contain_tol:
                continue
            if deep_inside_count < 2 and clearance < strict_gap:
                continue

            score = (
                8.0 * float(cov)
                + 4.0 * float(vis)
                + 0.018 * float(c_score)
                + 0.9 * float(area_norm)
                + 1.5 * float(edge_min_ratio)
                + 0.06 * (float(clearance) / max(1.0, float(min_hw)))
                + 0.04 * (float(avg_gap) / max(1.0, float(min_hw)))
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "rect": rect,
                    "circle": (float(cx), float(cy), float(r), float(cov), float(vis)),
                    "clearance": float(clearance),
                    "avg_gap": float(avg_gap),
                }

    if best is None:
        return None, "No circle-rectangle pair satisfies 'rectangle completely inside circle'. "
    return best, ""


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


def judge_plane_39(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 4:
        return [False, "Insufficient line structure for rectangle detection. "]

    rectangles = _extract_rectangle_candidates(img, lines=lines, min_hw=min_hw)
    if not rectangles:
        return [False, "Failed to reconstruct a valid rectangle from detected lines. "]

    circles = _detect_circle_candidates(img, min_hw=min_hw)
    if not circles:
        return [False, "Failed to detect a reliable circle boundary. "]

    picked, reason = _select_rectangle_inside_circle(rectangles, circles, min_hw=min_hw)
    if picked is None:
        return [False, reason]

    rect = picked["rect"]
    vertices = rect["vertices"]
    cx, cy, r, cov, vis = picked["circle"]
    clearance = float(picked["clearance"])

    if vis < 0.50 or cov < 0.30:
        return [False, f"Circle trace is too weak (cov={cov:.2f}, vis={vis:.2f}). "]
    if clearance < -scale_px(min_hw, 0.012, floor_px=0.0):
        return [False, f"Rectangle is not fully inside circle (clearance={clearance:.1f}). "]

    expected_refs = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
    if extra_lines > 6:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    selected = (cx, cy, r)
    extra_circles = 0
    for cand in circles:
        c = (float(cand[0]), float(cand[1]), float(cand[2]))
        if _circle_equivalent(c, selected, min_hw=min_hw):
            continue
        if float(cand[5]) < 0.50 or float(cand[4]) < 0.32:
            continue
        if c[2] < max(scale_px(min_hw, 0.07, floor_px=0.0), 0.55 * r):
            continue
        center_shift = _dist((c[0], c[1]), (cx, cy))
        if 0.72 * r <= c[2] <= 0.88 * r and center_shift <= 0.25 * r:
            continue
        extra_circles += 1
    if extra_circles > 0:
        return [False, f"Detected extra prominent circle(s): {extra_circles}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_39,
        require_ocr=False,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
