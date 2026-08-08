import argparse
import math
import os

PID = 35
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
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "opp_rel_1": float(opp_rel_1),
        "opp_rel_2": float(opp_rel_2),
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


def _point_in_convex_polygon(point, vertices, tol):
    if vertices is None or len(vertices) < 3:
        return False
    px, py = float(point[0]), float(point[1])
    sign = 0
    for i in range(len(vertices)):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % len(vertices)][0]), float(vertices[(i + 1) % len(vertices)][1])
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) <= float(tol):
            continue
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif sign != cur:
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


def _detect_circle_candidates(img):
    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    min_r = int(round(scale_px(min_hw, 0.035, floor_px=1.0)))
    max_r = scale_px(min_hw, 0.45, floor_px=0.0)
    score_th = int(round(scale_px(min_hw, 0.10, floor_px=1.0)))
    band = int(max(1, round(scale_px(min_hw, 0.010, floor_px=0.0))))

    found, scores = _find_top_k_circles(
        img,
        k=20,
        min_r=min_r,
        max_r=0,
        seed=0,
        iters=3600,
        score_th=score_th,
    )

    x_margin = 0.12 * float(w)
    y_margin = 0.12 * float(h)
    candidates = []
    for (cx, cy, r), s in zip(found, scores):
        x, y, rr = float(cx), float(cy), float(r)
        if rr <= 0.0 or rr > max_r:
            continue
        if x < (-x_margin) or x > (float(w) + x_margin):
            continue
        if y < (-y_margin) or y > (float(h) + y_margin):
            continue
        refined = _refine_circle_radius_by_inner_outer_edges(img, (x, y, rr))
        if refined is None:
            continue
        rx, ry, rr2 = [float(v) for v in refined]
        if rr2 <= 0.0 or rr2 > max_r:
            continue
        cov, vis = _circle_ink_coverage(img, (rx, ry), rr2, band)
        if vis < 0.45 or cov < 0.30:
            continue
        candidates.append((rx, ry, rr2, int(s), float(cov), float(vis)))

    if candidates:
        candidates.sort(key=lambda t: (-t[3], -t[2]))
        merged = _merge_circle_candidates(
            [(x, y, r, s) for (x, y, r, s, _, _) in candidates],
            center_tol=scale_px(min_hw, 0.03, floor_px=0.0),
            radius_tol=scale_px(min_hw, 0.03, floor_px=0.0),
        )
        merged_out = []
        for x, y, r, s in merged:
            cov, vis = _circle_ink_coverage(img, (x, y), r, band)
            if vis < 0.45 or cov < 0.30:
                continue
            merged_out.append((float(x), float(y), float(r), int(s), float(cov), float(vis)))
        merged_out.sort(key=lambda t: (-t[3], -t[4], -t[2]))
        if merged_out:
            return merged_out

    fallback = []
    for order in (1, 2, 3):
        c = detect_circle(img, order=order, min_r=min_r, max_r=0)
        if c is None:
            continue
        x, y, r = [float(v) for v in c]
        if r <= 0.0 or r > max_r:
            continue
        cov, vis = _circle_ink_coverage(img, (x, y), r, band)
        if vis < 0.45 or cov < 0.30:
            continue
        fallback.append((x, y, r, 0, cov, vis))
    return fallback


def _select_right_outside_circle(candidates, vertices, min_hw):
    if not candidates:
        return None, "Failed to detect a valid circle. "

    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    rect_w = max(1e-6, x_max - x_min)
    rect_h = max(1e-6, y_max - y_min)
    rect_center_x = 0.5 * (x_min + x_max)

    side_segs = [(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    inside_tol = scale_px(min_hw, 0.010, floor_px=0.0)
    outside_gap = scale_px(min_hw, 0.006, floor_px=0.0)
    right_gap = scale_px(min_hw, 0.004, floor_px=0.0)
    min_circle_r = max(scale_px(min_hw, 0.035, floor_px=0.0), 0.08 * min(rect_w, rect_h))

    valid = []
    for cx, cy, r, s, cov, vis in candidates:
        if float(r) < min_circle_r:
            continue
        if _point_in_convex_polygon((cx, cy), vertices, tol=inside_tol):
            continue

        edge_min = min(_point_to_segment_distance((cx, cy), p1, p2) for p1, p2 in side_segs)
        clear = float(edge_min) - float(r)
        if clear <= outside_gap:
            continue

        right_clear = (float(cx) - float(r)) - x_max
        if right_clear <= right_gap:
            continue
        if float(cx) <= rect_center_x + 0.08 * rect_w:
            continue

        overlap = max(0.0, min(float(cy) + float(r), y_max) - max(float(cy) - float(r), y_min))
        need_overlap = 0.12 * min(2.0 * float(r), rect_h)
        if overlap < need_overlap:
            continue

        score = 2.4 * right_clear + 0.7 * clear + 0.12 * float(s) + 9.0 * float(cov) + 0.06 * float(r)
        valid.append((score, (float(cx), float(cy), float(r)), right_clear, clear))

    if not valid:
        return None, "No circle satisfies right-side and fully-outside rectangle constraints. "

    valid.sort(key=lambda t: t[0], reverse=True)
    return valid[0][1], ""


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


def _validate_labels_if_ocr_ready(img, vertices, min_hw):
    os.environ.setdefault("PLANE_EASYOCR_AUTO_DOWNLOAD", "0")
    if _get_easyocr_reader() is None:
        return True, ""

    tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.10)
    max_dist = scale_px(min_hw, 0.12, floor_px=0.0)
    ok_cycle, detected, _ = match_labels_in_cycle(
        tokens=tokens,
        vertices=vertices,
        target_labels=["A", "B", "C", "D"],
        max_dist=max_dist,
        allow_reversed=True,
        min_conf=0.10,
        single_char_only=False,
    )
    hits = sum(1 for x in detected if x is not None)
    if not ok_cycle and hits < 3:
        return False, f"Failed to detect enough rectangle labels A/B/C/D (hits={hits}). "
    return True, ""


def judge_plane_35(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 4:
        return [False, "Insufficient line structure for rectangle detection. "]

    quad = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=4,
        min_len_ratio=0.15,
        top_k=14,
        min_angle_sep_deg=10.0,
        margin_ratio=0.20,
        point_tol_ratio=0.04,
        min_area_ratio=0.008,
        support_t=(-0.55, 1.40),
    )
    if quad is None:
        return [False, "Failed to reconstruct rectangle boundary. "]

    vertices = quad["vertices"]
    metrics = _rectangle_metrics(vertices)
    if metrics is None:
        return [False, "Detected quadrilateral is not rectangle-like. "]
    if max(metrics["cos_vals"]) > 0.24:
        return [False, "Detected quadrilateral is not rectangle-like. "]
    if metrics["opp_rel_1"] > 0.24 or metrics["opp_rel_2"] > 0.24:
        return [False, "Detected quadrilateral is not rectangle-like. "]

    min_edge = min(float(e) for e in metrics["edges"])
    if min_edge < scale_px(min_hw, 0.12, floor_px=0.0):
        return [False, "Detected rectangle is too small. "]

    circles = _detect_circle_candidates(img)
    circle, reason = _select_right_outside_circle(circles, vertices, min_hw=min_hw)
    if circle is None:
        return [False, reason]

    cx, cy, r = [float(v) for v in circle]
    if r < scale_px(min_hw, 0.035, floor_px=0.0):
        return [False, "Detected circle is too small. "]

    expected_refs = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
    if extra_lines > 2:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    extra_circle_count = 0
    for qx, qy, qr, _, cov, vis in circles:
        if vis < 0.48 or cov < 0.32:
            continue
        if _dist((qx, qy), (cx, cy)) <= scale_px(min_hw, 0.09, floor_px=0.0) and abs(float(qr) - r) <= scale_px(
            min_hw, 0.07, floor_px=0.0
        ):
            continue
        if float(qr) < 0.65 * r:
            continue
        extra_circle_count += 1
    if extra_circle_count > 1:
        return [False, f"Detected extra prominent circles ({extra_circle_count}). "]

    ok_label, msg = _validate_labels_if_ocr_ready(img, vertices, min_hw=min_hw)
    if not ok_label:
        return [True, f"Geometry valid; OCR label check was inconclusive: {msg}"]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_35,
        require_ocr=False,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
