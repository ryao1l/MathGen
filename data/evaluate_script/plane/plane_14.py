import argparse

PID = 22
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
        raise RuntimeError("Failed to build side line from vertices.")
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _rectangle_metrics(vertices):
    if len(vertices) != 4:
        return None

    edges = [_dist(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
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
        cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_vals.append(abs(cosv))
    return edges, cos_vals


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


def _cycle_starting_from(seq, idx):
    n = len(seq)
    return [seq[(idx + i) % n] for i in range(n)]


def judge_plane_22(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 6:
        return [False, "Insufficient line structure for hard case 22. "]

    quad = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=4,
        min_len_ratio=0.18,
        top_k=12,
        min_angle_sep_deg=12.0,
        margin_ratio=0.15,
        point_tol_ratio=0.04,
        min_area_ratio=0.010,
        support_t=(-0.45, 1.35),
    )
    if quad is None:
        return [False, "Failed to reconstruct square boundary. "]

    vertices = quad["vertices"]
    metrics = _rectangle_metrics(vertices)
    if metrics is None:
        return [False, "Invalid quadrilateral geometry. "]
    edges, cos_vals = metrics
    if max(cos_vals) > 0.24:
        return [False, "Detected boundary is not rectangle-like (right-angle check failed). "]

    mean_edge = sum(float(e) for e in edges) / 4.0
    if mean_edge <= 1e-6:
        return [False, "Degenerate square boundary. "]
    edge_rel = (max(edges) - min(edges)) / max(1e-6, mean_edge)
    if edge_rel > 0.12:
        return [False, f"Boundary is not square-like enough (edge_rel={edge_rel:.3f}). "]

    h_geo = (
        sum(float(v[0]) for v in vertices) / 4.0,
        sum(float(v[1]) for v in vertices) / 4.0,
    )

    edge_items = [_segment_item(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    r_in = sum(point_line_distance(h_geo, e["abc"]) for e in edge_items) / 4.0
    r_out = sum(_dist(h_geo, v) for v in vertices) / 4.0
    if r_out <= r_in + scale_px(min_hw, 0.03, floor_px=0.0):
        return [False, "Could not derive distinct inner/outer circle radii from square geometry. "]

    band = int(max(1, round(scale_px(min_hw, 0.012, floor_px=0.0))))
    in_cov, in_vis = _circle_ink_coverage(img, h_geo, r_in, band)
    out_cov, out_vis = _circle_ink_coverage(img, h_geo, r_out, band)
    if in_vis < 0.45 or out_vis < 0.25:
        return [False, "Circle visibility is too low to verify both circles. "]
    if in_cov < 0.50 or out_cov < 0.35:
        return [
            False,
            f"Missing inscribed/circumscribed circle traces (inner_cov={in_cov:.2f}, outer_cov={out_cov:.2f}). ",
        ]

    hv_ratios = [segment_ink_ratio(img, h_geo, vertices[i], thickness=2, trim_ratio=0.03) for i in range(4)]
    if any(r < 0.22 for r in hv_ratios):
        detail = ",".join(f"{r:.2f}" for r in hv_ratios)
        return [False, f"Missing some H-to-vertex segments (ratios={detail}). "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDH", min_conf=0.10)
    best_by_char = pick_best_tokens_by_char(tokens, ["A", "B", "C", "D", "H"], min_conf=0.10)
    missing = [ch for ch in ["A", "B", "C", "D", "H"] if ch not in best_by_char]
    if missing:
        return [False, f"Missing labels from OCR: {','.join(missing)}. "]

    assign, dists = assign_labels_to_vertices_min_cost(best_by_char, vertices, ["A", "B", "C", "D"])
    if assign is None or dists is None:
        return [False, "Failed to assign A/B/C/D labels to square vertices. "]

    max_label_dist = 0.28 * mean_edge
    far = [ch for ch in ["A", "B", "C", "D"] if float(dists[ch]) > max_label_dist]
    if far:
        detail = ",".join(f"{ch}:{dists[ch]:.1f}" for ch in far)
        return [False, f"Some vertex labels are too far from square corners ({detail}). "]

    labels_on_cw = [None] * 4
    for ch in ["A", "B", "C", "D"]:
        labels_on_cw[assign[ch]] = ch

    idx_tl = min(range(4), key=lambda i: float(vertices[i][0]) + float(vertices[i][1]))
    seq_tl_cw = _cycle_starting_from(labels_on_cw, idx_tl)
    if seq_tl_cw not in (["A", "B", "C", "D"], ["A", "D", "C", "B"]):
        return [False, f"Vertex labels are inconsistent from top-left A (detected={seq_tl_cw}). "]

    h_tok = select_token_near_point(
        tokens,
        expected_char="H",
        point=h_geo,
        max_dist=0.12 * mean_edge,
    )
    if h_tok is None:
        return [False, "Failed to detect center label H near square center. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_22,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
