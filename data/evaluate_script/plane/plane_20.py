import argparse
import math

PID = 43
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


def _extract_square_candidates(img, lines, min_hw, min_edge_ratio):
    if not lines:
        return []

    configs = [
        (0.20, 12, 0.010, 12.0),
        (0.16, 14, 0.007, 11.0),
        (0.13, 16, 0.004, 10.0),
        (0.10, 20, 0.0025, 9.0),
        (0.08, 24, 0.0016, 8.0),
        (0.06, 28, 0.0010, 8.0),
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
            margin_ratio=0.18,
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
        if max(metrics["cos_vals"]) > 0.28:
            continue
        if metrics["edge_rel"] > 0.23:
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

        duplicate = False
        for ex in cands:
            if _dist(metrics["center"], ex["metrics"]["center"]) > scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            if abs(metrics["diag"] - ex["metrics"]["diag"]) > scale_px(min_hw, 0.05, floor_px=0.0):
                continue
            duplicate = True
            break
        if duplicate:
            continue

        cands.append(
            {
                "vertices": vertices,
                "lines": quad["lines"],
                "edge_items": edge_items,
                "metrics": metrics,
                "area": float(quad.get("area", 0.0)),
                "score": float(quad.get("score", 0.0)),
                "edge_ratios": edge_ratios,
            }
        )

    cands.sort(
        key=lambda c: (
            c["area"],
            min(c["edge_ratios"]) if c["edge_ratios"] else 0.0,
            c["score"],
        ),
        reverse=True,
    )
    return cands


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


def _square_pair_stats(outer_sq, inner_sq, min_hw):
    outer_v = outer_sq["vertices"]
    inner_v = inner_sq["vertices"]
    if not outer_v or not inner_v:
        return False, {}

    if inner_sq["metrics"]["mean_edge"] >= 0.96 * outer_sq["metrics"]["mean_edge"]:
        return False, {"reason": "Inner square is not smaller than outer square."}
    if inner_sq["area"] >= 0.92 * outer_sq["area"]:
        return False, {"reason": "Inner square area is too close to outer square area."}

    inside_tol = scale_px(min_hw, 0.010, floor_px=0.0)
    for p in inner_v:
        if not _point_inside_convex_polygon_soft(p, outer_v, tol_dist=inside_tol):
            return False, {"reason": "At least one inner-square vertex is outside outer square."}

    if not _point_inside_convex_polygon_soft(inner_sq["metrics"]["center"], outer_v, tol_dist=inside_tol):
        return False, {"reason": "Inner-square center is outside outer square."}

    clearances = []
    for p in inner_v:
        d = min(point_line_distance(p, e["abc"]) for e in outer_sq["edge_items"])
        clearances.append(float(d))
    min_clearance = min(clearances)
    avg_clearance = sum(clearances) / 4.0

    touch_gap = scale_px(min_hw, 0.003, floor_px=0.0)
    if min_clearance <= touch_gap and avg_clearance <= scale_px(min_hw, 0.010, floor_px=0.0):
        return False, {"reason": "Inner square is touching/crossing outer boundary."}

    return True, {
        "min_clearance": float(min_clearance),
        "avg_clearance": float(avg_clearance),
    }


def _cycle_is_rotation(seq, target):
    if len(seq) != len(target):
        return False
    n = len(seq)
    for s in range(n):
        ok = True
        for i in range(n):
            if seq[(s + i) % n] != target[i]:
                ok = False
                break
        if ok:
            return True
    return False


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


def _vertex_label_ink(img, point, edge_lines, min_hw):
    if img is None or point is None:
        return 0
    h, w = img.shape[:2]
    px, py = float(point[0]), float(point[1])
    _, bw = _gray_and_ink_mask(img)
    win = scale_px(min_hw, 0.13, floor_px=0.0)
    x1 = int(max(0, px - win))
    x2 = int(min(w, px + win))
    y1 = int(max(0, py - win))
    y2 = int(min(h, py + win))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    vertex_disk = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.025 * min_hw) ** 2
    edge_band = np.zeros_like(roi, dtype=bool)
    for ln in edge_lines if isinstance(edge_lines, list) else []:
        a, b, c = [float(v) for v in ln["abc"]]
        den = max(1e-6, math.hypot(a, b))
        edge_band |= np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den <= max(3.0, 0.012 * min_hw)
    label_mask = roi & (~vertex_disk) & (~edge_band)
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


def judge_plane_43(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.05)
    if len(lines) < 6:
        return [False, "Insufficient line structure for two-square task. "]

    outer_candidates = _extract_square_candidates(
        img=img,
        lines=lines,
        min_hw=min_hw,
        min_edge_ratio=0.12,
    )
    if not outer_candidates:
        return [False, "Failed to reconstruct outer square candidate. "]

    best_pair = None
    best_pair_score = None
    best_reason = "Failed to find a valid inner square completely inside outer square. "

    for outer_sq in outer_candidates[:3]:
        outer_refs = list(outer_sq["lines"]) + list(outer_sq["edge_items"])
        remaining_lines = _remove_equivalent_lines(lines, refs=outer_refs, min_hw=min_hw)

        inner_candidates = _extract_square_candidates(
            img=img,
            lines=remaining_lines,
            min_hw=min_hw,
            min_edge_ratio=0.05,
        )
        if not inner_candidates:
            loose_inner = _extract_square_candidates(
                img=img,
                lines=lines,
                min_hw=min_hw,
                min_edge_ratio=0.05,
            )
            inner_candidates = []
            for c in loose_inner:
                if _dist(c["metrics"]["center"], outer_sq["metrics"]["center"]) <= scale_px(min_hw, 0.04, floor_px=0.0):
                    if abs(c["metrics"]["diag"] - outer_sq["metrics"]["diag"]) <= scale_px(min_hw, 0.05, floor_px=0.0):
                        continue
                if c["area"] >= 0.92 * outer_sq["area"]:
                    continue
                inner_candidates.append(c)

        for inner_sq in inner_candidates[:10]:
            ok_pair, stats = _square_pair_stats(outer_sq, inner_sq, min_hw=min_hw)
            if not ok_pair:
                if "reason" in stats:
                    best_reason = str(stats["reason"]) + " "
                continue

            score = (
                0.0035 * float(outer_sq["area"])
                + 0.0050 * float(inner_sq["area"])
                + 1.6 * min(inner_sq["edge_ratios"])
                + 1.2 * min(outer_sq["edge_ratios"])
                + 0.07 * (float(stats["avg_clearance"]) / max(1.0, min_hw))
            )
            if best_pair_score is None or score > best_pair_score:
                best_pair_score = score
                best_pair = {"outer": outer_sq, "inner": inner_sq}

    if best_pair is None:
        return [False, best_reason]

    outer_sq = best_pair["outer"]
    inner_sq = best_pair["inner"]
    outer_vertices = outer_sq["vertices"]

    tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.08)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D"], min_conf=0.08)
    missing = [ch for ch in ["A", "B", "C", "D"] if ch not in best_by_char]
    if missing:
        ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
        ink_hits = [_vertex_label_ink(img, v, outer_sq["edge_items"], min_hw) for v in outer_vertices]
        if not (len(ink_hits) == 4 and all(val >= ink_th for val in ink_hits)):
            return [False, f"Missing labels from OCR: {','.join(missing)} (label_ink={ink_hits}). "]
        expected_refs = list(outer_sq["edge_items"]) + list(inner_sq["edge_items"])
        extra_lines = _count_extra_dominant_lines(lines, expected_refs=expected_refs, min_hw=min_hw)
        if extra_lines > 2:
            return [False, f"Detected too many extra dominant lines ({extra_lines}). "]
        return [True, ""]

    assign, dists = assign_labels_to_vertices_min_cost(best_by_char, outer_vertices, ["A", "B", "C", "D"])
    if assign is None or dists is None:
        return [False, "Failed to assign A/B/C/D labels to outer-square vertices. "]

    label_far_th = 0.34 * outer_sq["metrics"]["mean_edge"]
    far = [ch for ch in ["A", "B", "C", "D"] if float(dists[ch]) > label_far_th]
    if far:
        detail = ",".join(f"{ch}:{dists[ch]:.1f}" for ch in far)
        return [False, f"Some labels are too far from outer-square corners ({detail}). "]

    labels_on_cycle = [None] * 4
    for ch in ["A", "B", "C", "D"]:
        labels_on_cycle[assign[ch]] = ch

    if not all(v is not None for v in labels_on_cycle):
        return [False, "Failed to map A/B/C/D labels onto four outer-square vertices. "]

    if not (
        _cycle_is_rotation(labels_on_cycle, ["A", "B", "C", "D"])
        or _cycle_is_rotation(labels_on_cycle, ["A", "D", "C", "B"])
    ):
        return [False, f"Outer-square labels do not follow cyclic ABCD order (detected={labels_on_cycle}). "]

    expected_refs = list(outer_sq["edge_items"]) + list(inner_sq["edge_items"])
    extra_lines = _count_extra_dominant_lines(lines, expected_refs=expected_refs, min_hw=min_hw)
    if extra_lines > 2:
        return [False, f"Detected too many extra dominant lines ({extra_lines}). "]

    residual_lines = _remove_equivalent_lines(lines, refs=expected_refs, min_hw=min_hw)
    extra_square_candidates = _extract_square_candidates(
        img=img,
        lines=residual_lines,
        min_hw=min_hw,
        min_edge_ratio=0.05,
    )
    if extra_square_candidates and float(extra_square_candidates[0]["area"]) > 0.20 * float(inner_sq["area"]):
        return [False, "Detected extra square-like boundary beyond required two squares. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_43,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
