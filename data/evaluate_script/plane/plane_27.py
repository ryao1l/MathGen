import argparse
import math

PID = 54
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
    center = (
        sum(float(v[0]) for v in vertices) / 4.0,
        sum(float(v[1]) for v in vertices) / 4.0,
    )
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "opp_rel_1": float(opp_rel_1),
        "opp_rel_2": float(opp_rel_2),
        "min_edge": float(min(edges)),
        "diag": float(diag),
        "center": center,
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


def _extract_rectangle_candidates(img, lines, min_hw):
    configs = [
        (0.18, 12, 0.010, 10.0),
        (0.15, 14, 0.008, 9.0),
        (0.12, 18, 0.006, 8.0),
    ]
    out = []

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
        if metrics["opp_rel_1"] > 0.30 or metrics["opp_rel_2"] > 0.30:
            continue
        if metrics["min_edge"] < scale_px(min_hw, 0.10, floor_px=0.0):
            continue

        edge_ratios = []
        edge_ok = True
        for i in range(4):
            ok, ratio = has_segment_between_points(
                img,
                vertices[i],
                vertices[(i + 1) % 4],
                ratio_th=0.10,
                thickness=2,
                trim_ratio=0.02,
            )
            edge_ratios.append(float(ratio))
            if not ok:
                edge_ok = False
        if not edge_ok:
            continue

        duplicated = False
        for ex in out:
            c1 = metrics["center"]
            c2 = ex["metrics"]["center"]
            if _dist(c1, c2) > scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            if abs(float(metrics["diag"]) - float(ex["metrics"]["diag"])) > scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            duplicated = True
            break
        if duplicated:
            continue

        out.append(
            {
                "vertices": vertices,
                "lines": quad["lines"],
                "metrics": metrics,
                "area": float(quad.get("area", 0.0)),
                "score": float(quad.get("score", 0.0)),
                "edge_ratios": edge_ratios,
            }
        )

    out.sort(
        key=lambda t: (
            min(t["edge_ratios"]) if t["edge_ratios"] else 0.0,
            t["area"],
            t["score"],
        ),
        reverse=True,
    )
    return out


def _detect_circle_candidates(img, min_hw):
    h, w = img.shape[:2]
    min_r = int(round(scale_px(min_hw, 0.025, floor_px=1.0)))
    max_r = float(scale_px(min_hw, 0.45, floor_px=0.0))
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
    for order in (1, 2, 3, 4, 5):
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


def _select_rectangle_inner_circle_pair(rectangles, circles, min_hw):
    if not rectangles:
        return None, "Failed to reconstruct a valid rectangle. "
    if not circles:
        return None, "Failed to detect a valid circle. "

    contain_tol = scale_px(min_hw, 0.015, floor_px=0.0)
    strict_gap = scale_px(min_hw, 0.003, floor_px=0.0)
    best = None
    best_score = None

    for rect in rectangles:
        vertices = rect["vertices"]
        edge_lines = rect["lines"]
        min_edge = float(rect["metrics"]["min_edge"])
        center = rect["metrics"]["center"]
        edge_min_ratio = min(rect["edge_ratios"]) if rect["edge_ratios"] else 0.0
        area_norm = float(rect["area"]) / max(1.0, float(min_hw) * float(min_hw))

        for cx, cy, r, c_score, cov, vis in circles:
            cpt = (float(cx), float(cy))
            if not _point_in_convex_polygon(cpt, vertices, tol=contain_tol):
                continue
            if float(r) < max(scale_px(min_hw, 0.025, floor_px=0.0), 0.08 * min_edge):
                continue
            if float(r) > 0.52 * min_edge + contain_tol:
                continue

            side_dists = [point_line_distance(cpt, ln["abc"]) for ln in edge_lines]
            if len(side_dists) != 4:
                continue
            dmin = float(min(side_dists))
            clearance = float(dmin - r)
            if clearance < -contain_tol:
                continue
            if dmin <= r + strict_gap and clearance < 0.0:
                continue

            center_bias = _dist(cpt, center) / max(1.0, min_edge)
            score = (
                8.0 * float(cov)
                + 4.0 * float(vis)
                + 0.018 * float(c_score)
                + 1.2 * float(edge_min_ratio)
                + 0.8 * float(area_norm)
                + 0.10 * (float(clearance) / max(1.0, float(min_hw)))
                - 0.25 * float(center_bias)
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "rect": rect,
                    "circle": (float(cx), float(cy), float(r), float(cov), float(vis)),
                    "clearance": float(clearance),
                }

    if best is None:
        return None, "No circle satisfies the 'completely inside rectangle' constraint. "
    return best, ""


def _count_extra_dominant_lines(lines, expected_refs, min_hw):
    long_th = scale_px(min_hw, 0.22, floor_px=0.0)
    extras = 0
    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        if float(ln.get("len", 0.0)) < long_th:
            continue
        if any(line_equivalent(ln, ref, min_hw=min_hw, angle_tol_deg=5.0) for ref in expected_refs):
            continue
        extras += 1
    return int(extras)


def _select_center_point_marker(img, circle_center, p_label_center, min_hw, circle_r):
    candidates = []
    for p in detect_marker_points(img):
        candidates.append((float(p[0]), float(p[1])))

    cx, cy = float(circle_center[0]), float(circle_center[1])
    step = scale_px(min_hw, 0.008, floor_px=1.0, ceil_px=max(1.0, 0.08 * float(circle_r)))
    probes = [
        (cx, cy),
        (cx - step, cy),
        (cx + step, cy),
        (cx, cy - step),
        (cx, cy + step),
    ]
    for p in probes:
        if has_point_at_point(img, p):
            candidates.append((float(p[0]), float(p[1])))

    if p_label_center is not None and has_point_at_point(img, p_label_center):
        candidates.append((float(p_label_center[0]), float(p_label_center[1])))

    candidates = dedup_points(candidates, tol=scale_px(min_hw, 0.012, floor_px=1.0))
    if not candidates:
        return None

    center_tol = max(scale_px(min_hw, 0.030, floor_px=0.0), 0.14 * float(circle_r))
    best = None
    best_key = None
    for p in candidates:
        dc = _dist(p, circle_center)
        if dc > center_tol:
            continue
        dl = _dist(p, p_label_center) if p_label_center is not None else 0.0
        key = (float(dc), float(dl))
        if best_key is None or key < best_key:
            best_key = key
            best = p
    return best


def _corner_annotation_hits(img, vertices, min_hw):
    if img is None or not vertices:
        return 0
    h, w = img.shape[:2]
    _, bw = _gray_and_ink_mask(img)
    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    cx = sum(xs) / float(len(xs))
    cy = sum(ys) / float(len(ys))
    half = scale_px(min_hw, 0.055, floor_px=7.0)
    hits = 0
    for vx, vy in vertices:
        vx, vy = float(vx), float(vy)
        sx = -1.0 if vx < cx else 1.0
        sy = -1.0 if vy < cy else 1.0
        # Look just outside the rectangle corner, where A/B/C/D labels live.
        px = vx + sx * 0.055 * float(min_hw)
        py = vy + sy * 0.045 * float(min_hw)
        x1 = int(max(0, px - half))
        x2 = int(min(w, px + half))
        y1 = int(max(0, py - half))
        y2 = int(min(h, py + half))
        if x2 <= x1 or y2 <= y1:
            continue
        ink = int((bw[y1:y2, x1:x2] > 0).sum())
        if ink >= max(8, int(0.000004 * h * w)):
            hits += 1
    return hits


def _p_label_ink_near_center(img, center, min_hw, circle_r):
    if img is None or center is None:
        return 0
    h, w = img.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    rr = float(circle_r)
    _, bw = _gray_and_ink_mask(img)
    half = max(scale_px(min_hw, 0.055, floor_px=7.0), 0.26 * rr)
    x1 = int(max(0, cx - half))
    x2 = int(min(w, cx + half))
    y1 = int(max(0, cy - half))
    y2 = int(min(h, cy + half))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    dot_disk = ((xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2) <= (0.10 * rr) ** 2
    label_mask = roi & (~dot_disk)
    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    best = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < max(6, int(0.000003 * h * w)):
            continue
        if bw_box > 0.22 * float(min_hw) or bh_box > 0.22 * float(min_hw):
            continue
        best = max(best, area)
    return best


def judge_plane_54(img):
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

    picked, reason = _select_rectangle_inner_circle_pair(rectangles, circles, min_hw=min_hw)
    if picked is None:
        return [False, reason]

    rect = picked["rect"]
    vertices = rect["vertices"]
    cx, cy, r, cov, vis = picked["circle"]
    clearance = float(picked["clearance"])

    if vis < 0.50 or cov < 0.30:
        return [False, f"Circle trace is too weak (cov={cov:.2f}, vis={vis:.2f}). "]
    if clearance < -scale_px(min_hw, 0.015, floor_px=0.0):
        return [False, f"Circle is not fully inside rectangle (clearance={clearance:.1f}). "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDP", min_conf=0.08)
    label_radius = max(scale_px(min_hw, 0.13, floor_px=0.0), 0.22 * float(rect["metrics"]["min_edge"]))
    ok_cycle, detected, _ = match_labels_in_cycle(
        tokens=tokens,
        vertices=vertices,
        target_labels=["A", "B", "C", "D"],
        max_dist=label_radius,
        allow_reversed=True,
        min_conf=0.08,
        single_char_only=False,
    )
    hits = sum(1 for x in detected if x is not None)
    if not ok_cycle and hits < 3:
        ink_hits = _corner_annotation_hits(img, vertices, min_hw)
        if ink_hits < 3:
            return [False, f"Failed to detect enough rectangle labels A/B/C/D (hits={hits}, ink_hits={ink_hits}). "]

    p_tok = select_token_near_point(
        tokens=tokens,
        expected_char="P",
        point=(cx, cy),
        max_dist=max(scale_px(min_hw, 0.16, floor_px=0.0), 0.38 * float(r)),
    )
    if p_tok is None:
        p_ink = _p_label_ink_near_center(img, (cx, cy), min_hw, r)
        if p_ink < max(8, int(0.000003 * img.shape[0] * img.shape[1])):
            return [False, f"Failed to detect label P near circle center (p_ink={p_ink}). "]
        p_label_center = (float(cx), float(cy))
    else:
        p_label_center = (float(p_tok["center"][0]), float(p_tok["center"][1]))

    p_point = _select_center_point_marker(
        img=img,
        circle_center=(cx, cy),
        p_label_center=p_label_center,
        min_hw=min_hw,
        circle_r=r,
    )
    if p_point is None:
        return [False, "Failed to detect point marker at the circle center for P. "]

    if p_tok is not None:
        label_point_dist = token_edge_distance_to_point(p_tok, p_point)
        if label_point_dist > max(scale_px(min_hw, 0.15, floor_px=0.0), 0.44 * float(r)):
            return [False, f"Label P is too far from its point marker (dist={label_point_dist:.1f}). "]

    expected_refs = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    extra_lines = _count_extra_dominant_lines(lines, expected_refs, min_hw=min_hw)
    if extra_lines > 0:
        return [False, f"Detected extra dominant line(s) outside rectangle: {extra_lines}. "]

    selected = (cx, cy, r)
    inner_tol = scale_px(min_hw, 0.015, floor_px=0.0)
    min_edge = float(rect["metrics"]["min_edge"])
    extra_circles = 0
    for cand in circles:
        c = (float(cand[0]), float(cand[1]), float(cand[2]))
        if _circle_equivalent(c, selected, min_hw=min_hw):
            continue
        if _dist((c[0], c[1]), (selected[0], selected[1])) <= scale_px(min_hw, 0.040, floor_px=0.0):
            if 0.72 * selected[2] <= c[2] <= 1.28 * selected[2]:
                continue
        if float(cand[5]) < 0.50 or float(cand[4]) < 0.32:
            continue
        if c[2] < max(scale_px(min_hw, 0.04, floor_px=0.0), 0.55 * float(r)):
            continue
        if c[2] > 0.52 * min_edge + inner_tol:
            continue
        if not _point_in_convex_polygon((c[0], c[1]), vertices, tol=inner_tol):
            continue
        side_d = [point_line_distance((c[0], c[1]), ln["abc"]) for ln in rect["lines"]]
        if len(side_d) != 4:
            continue
        if min(side_d) - c[2] < -inner_tol:
            continue
        extra_circles += 1
    if extra_circles > 0:
        return [False, f"Detected extra prominent circle(s): {extra_circles}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_54,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
