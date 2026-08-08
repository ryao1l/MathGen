import argparse
import math

PID = 52
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


def _median_scale(values):
    arr = sorted(float(v) for v in values if float(v) > 1e-6)
    if not arr:
        return 1.0
    n = len(arr)
    if n % 2 == 1:
        return float(arr[n // 2])
    return 0.5 * float(arr[n // 2 - 1] + arr[n // 2])


def _parallelogram_metrics(vertices):
    if not isinstance(vertices, list) or len(vertices) != 4:
        return None

    edge_items = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    edges = [float(e["len"]) for e in edge_items]
    if any(e <= 1e-6 for e in edges):
        return None

    par_err_1 = float(angle_diff_deg(edge_items[0]["ang"], edge_items[2]["ang"]))
    par_err_2 = float(angle_diff_deg(edge_items[1]["ang"], edge_items[3]["ang"]))
    adj = [float(angle_diff_deg(edge_items[i]["ang"], edge_items[(i + 1) % 4]["ang"])) for i in range(4)]

    mid_ac = get_mid_point(vertices[0], vertices[2])
    mid_bd = get_mid_point(vertices[1], vertices[3])
    mid_gap = float(_dist(mid_ac, mid_bd))
    diag = float(max(_dist(vertices[0], vertices[2]), _dist(vertices[1], vertices[3])))
    center = (
        sum(float(v[0]) for v in vertices) / 4.0,
        sum(float(v[1]) for v in vertices) / 4.0,
    )

    return {
        "edge_items": edge_items,
        "edges": edges,
        "min_edge": float(min(edges)),
        "edge_scale": float(_median_scale(edges)),
        "par_err_1": float(par_err_1),
        "par_err_2": float(par_err_2),
        "adj_min": float(min(adj)),
        "mid_gap": float(mid_gap),
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
    min_r = int(round(scale_px(min_hw, 0.030, floor_px=1.0)))
    max_r = float(scale_px(min_hw, 0.42, floor_px=0.0))
    score_th = int(round(scale_px(min_hw, 0.090, floor_px=1.0)))
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
    for item in fallback:
        c = (item[0], item[1], item[2])
        if any(_circle_equivalent(c, (q[0], q[1], q[2]), min_hw=min_hw) for q in dedup):
            continue
        dedup.append(item)
    dedup.sort(key=lambda t: (-t[4], -t[2]))
    return dedup


def _extract_parallelogram_candidates(img, lines, min_hw):
    configs = [
        (0.18, 12, 0.010, 10.0),
        (0.14, 16, 0.007, 9.0),
        (0.11, 20, 0.004, 8.0),
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
            support_t=(-0.60, 1.40),
        )
        if quad is None:
            continue

        vertices = quad["vertices"]
        metrics = _parallelogram_metrics(vertices)
        if metrics is None:
            continue
        if metrics["par_err_1"] > 12.0 or metrics["par_err_2"] > 12.0:
            continue
        if metrics["adj_min"] < 9.0:
            continue
        if metrics["min_edge"] < scale_px(min_hw, 0.10, floor_px=0.0):
            continue
        mid_tol = max(scale_px(min_hw, 0.04, floor_px=0.0), 0.14 * metrics["edge_scale"])
        if metrics["mid_gap"] > mid_tol:
            continue

        edge_ratios = []
        edges_ok = True
        for i in range(4):
            p = vertices[i]
            q = vertices[(i + 1) % 4]
            ok, ratio = has_segment_between_points(img, p, q, ratio_th=0.10, thickness=2, trim_ratio=0.02)
            if not ok:
                side_line = find_support_line(lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.07)
                if side_line is None:
                    edges_ok = False
                    break
            edge_ratios.append(float(ratio))
        if not edges_ok:
            continue

        duplicate = False
        for ex in candidates:
            if _dist(metrics["center"], ex["metrics"]["center"]) > scale_px(min_hw, 0.05, floor_px=0.0):
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
                "edge_items": metrics["edge_items"],
                "metrics": metrics,
                "area": float(quad.get("area", 0.0)),
                "score": float(quad.get("score", 0.0)),
                "edge_ratios": edge_ratios,
            }
        )

    candidates.sort(
        key=lambda t: (
            t["area"],
            min(t["edge_ratios"]) if t["edge_ratios"] else 0.0,
            t["score"],
        ),
        reverse=True,
    )
    return candidates


def _circle_inside_parallelogram_stats(circle, vertices, min_hw):
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    center = (cx, cy)
    if vertices is None or len(vertices) != 4:
        return False, {}

    inside_tol = scale_px(min_hw, 0.010, floor_px=0.0)
    if not _point_inside_convex_polygon_soft(center, vertices, tol_dist=inside_tol):
        return False, {}

    edge_items = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    dists = [float(point_line_distance(center, e["abc"])) for e in edge_items]
    dmin = float(min(dists))
    clearance = float(dmin - r)
    cross_tol = scale_px(min_hw, 0.035, floor_px=0.0)
    if clearance < -cross_tol:
        return False, {}

    strict_gap = -scale_px(min_hw, 0.020, floor_px=0.0)
    return True, {
        "dmin": float(dmin),
        "clearance": float(clearance),
        "strict_gap": float(strict_gap),
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


def _validate_abcd_labels(img, vertices, min_hw, edge_scale):
    tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.08)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D"], min_conf=0.08)
    missing = [ch for ch in ["A", "B", "C", "D"] if ch not in best_by_char]
    if missing:
        edge_lines = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
        h, w = img.shape[:2]
        ink_th = max(10, int(0.000006 * h * w))
        ink_hits = []
        _, bw = _gray_and_ink_mask(img)
        for point in vertices:
            px, py = float(point[0]), float(point[1])
            win = scale_px(min_hw, 0.13, floor_px=0.0)
            x1 = int(max(0, px - win))
            x2 = int(min(w, px + win))
            y1 = int(max(0, py - win))
            y2 = int(min(h, py + win))
            if x2 <= x1 or y2 <= y1:
                ink_hits.append(0)
                continue
            roi = bw[y1:y2, x1:x2] > 0
            yy, xx = np.ogrid[y1:y2, x1:x2]
            vertex_disk = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.025 * min_hw) ** 2
            edge_band = np.zeros_like(roi, dtype=bool)
            for ln in edge_lines:
                a, b, c = [float(v) for v in ln["abc"]]
                den = max(1e-6, math.hypot(a, b))
                edge_band |= np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den <= max(3.0, 0.012 * min_hw)
            label_mask = roi & (~vertex_disk) & (~edge_band)
            if int(label_mask.sum()) <= 0:
                ink_hits.append(0)
                continue
            num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
            areas = []
            for i in range(1, num):
                area = int(stats[i, cv2.CC_STAT_AREA])
                bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
                bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
                if area < ink_th or area > max(350, int(0.0040 * h * w)):
                    continue
                if bw_box > 0.12 * min_hw or bh_box > 0.12 * min_hw:
                    continue
                areas.append(area)
            ink_hits.append(max(areas) if areas else 0)
        if len(ink_hits) == 4 and all(val >= ink_th for val in ink_hits):
            return True, ""
        return False, f"Missing labels from OCR: {','.join(missing)} (label_ink={ink_hits}). "

    max_dist = max(scale_px(min_hw, 0.08, floor_px=0.0), 0.40 * float(edge_scale))
    ok_cycle, detected, _ = match_labels_in_cycle(
        tokens=tokens,
        vertices=vertices,
        target_labels=["A", "B", "C", "D"],
        max_dist=max_dist,
        allow_reversed=True,
        min_conf=0.08,
        single_char_only=False,
    )
    if not ok_cycle:
        hits = sum(1 for x in detected if x is not None)
        return False, f"Failed to align labels A/B/C/D with parallelogram vertices (hits={hits}). "
    return True, ""


def judge_plane_52(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 4:
        return [False, "Insufficient line structure for parallelogram detection. "]

    para_cands = _extract_parallelogram_candidates(img=img, lines=lines, min_hw=min_hw)
    if not para_cands:
        return [False, "Failed to reconstruct a valid parallelogram boundary. "]

    circle_cands = _detect_circle_candidates(img, min_hw=min_hw)
    if not circle_cands:
        return [False, "Failed to detect a reliable circle boundary. "]

    best = None
    best_score = None
    for para in para_cands[:10]:
        metrics = para["metrics"]
        min_circle_r = max(scale_px(min_hw, 0.030, floor_px=0.0), 0.07 * metrics["min_edge"])
        for cx, cy, r, c_score, cov, vis in circle_cands:
            if float(r) < min_circle_r:
                continue
            inside_ok, stats = _circle_inside_parallelogram_stats((cx, cy, r), para["vertices"], min_hw=min_hw)
            if not inside_ok:
                continue
            clearance = float(stats["clearance"])
            rel_clear = clearance / max(1.0, float(r))
            score = (
                0.0014 * float(para["area"])
                + 0.010 * float(para["score"])
                + 1.6 * (min(para["edge_ratios"]) if para["edge_ratios"] else 0.0)
                + 12.0 * float(cov)
                + 8.0 * float(vis)
                + 0.020 * float(c_score)
                + 2.5 * float(rel_clear)
                + 0.012 * float(r)
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "para": para,
                    "circle": (float(cx), float(cy), float(r), float(cov), float(vis)),
                    "clearance": float(clearance),
                    "strict_gap": float(stats["strict_gap"]),
                }

    if best is None:
        return [False, "No circle is fully inside any detected parallelogram. "]

    para = best["para"]
    cx, cy, r, cov, vis = best["circle"]
    clearance = float(best["clearance"])
    strict_gap = float(best["strict_gap"])

    if vis < 0.50 or cov < 0.30:
        return [False, f"Circle trace is too weak (cov={cov:.2f}, vis={vis:.2f}). "]
    if clearance <= strict_gap:
        return [False, f"Circle is not completely inside parallelogram (clearance={clearance:.1f}). "]

    extra_lines = _count_extra_dominant_lines(lines, expected_refs=para["edge_items"], min_hw=min_hw)
    if extra_lines > 2:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    selected = (cx, cy, r)
    extra_circles = 0
    for cand in circle_cands:
        c = (float(cand[0]), float(cand[1]), float(cand[2]))
        if _circle_equivalent(c, selected, min_hw=min_hw):
            continue
        if float(cand[5]) < 0.50 or float(cand[4]) < 0.32:
            continue
        if c[2] < max(scale_px(min_hw, 0.030, floor_px=0.0), 0.62 * r):
            continue
        if _dist((c[0], c[1]), (cx, cy)) <= scale_px(min_hw, 0.08, floor_px=0.0):
            continue
        extra_circles += 1
    if extra_circles > 0:
        return [False, f"Detected extra prominent circle(s): {extra_circles}. "]

    ok_label, msg = _validate_abcd_labels(
        img=img,
        vertices=para["vertices"],
        min_hw=min_hw,
        edge_scale=para["metrics"]["edge_scale"],
    )
    if not ok_label:
        return [False, msg]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_52,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
