import argparse

PID = 11
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _rectangle_metrics(vertices):
    if not isinstance(vertices, list) or len(vertices) != 4:
        return None

    def _dist(p, q):
        return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))

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
        cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_vals.append(abs(cosv))

    width = 0.5 * (edges[0] + edges[2])
    height = 0.5 * (edges[1] + edges[3])
    opp_rel_1 = abs(edges[0] - edges[2]) / max(1e-6, 0.5 * (edges[0] + edges[2]))
    opp_rel_2 = abs(edges[1] - edges[3]) / max(1e-6, 0.5 * (edges[1] + edges[3]))
    return {
        "edges": edges,
        "cos_vals": cos_vals,
        "width": float(width),
        "height": float(height),
        "opp_rel_1": float(opp_rel_1),
        "opp_rel_2": float(opp_rel_2),
    }


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


def judge_plane_11(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.10)
    if len(lines) < 4:
        return [False, "Insufficient line structure for a quadrilateral. "]

    quad = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=4,
        min_len_ratio=0.18,
        top_k=10,
        min_angle_sep_deg=12.0,
        margin_ratio=0.15,
        point_tol_ratio=0.04,
        min_area_ratio=0.010,
        support_t=(-0.45, 1.35),
    )
    if quad is None:
        return [False, "Failed to reconstruct a closed quadrilateral. "]

    edge_lines = quad["lines"]
    vertices = quad["vertices"]
    metrics = _rectangle_metrics(vertices)
    if metrics is None:
        return [False, "Detected quadrilateral is not rectangle-like. "]

    if max(metrics["cos_vals"]) > 0.22:
        return [False, "Detected quadrilateral is not rectangle-like. "]
    if metrics["opp_rel_1"] > 0.20 or metrics["opp_rel_2"] > 0.20:
        return [False, "Detected quadrilateral is not rectangle-like. "]
    width_margin = scale_px(min_hw, 0.03, floor_px=0.0)
    if metrics["width"] <= metrics["height"] + width_margin:
        return [False, f"Rectangle width is not greater than height (w={metrics['width']:.1f}, h={metrics['height']:.1f}). "]

    label_radius = 0.22 * min(metrics["width"], metrics["height"])
    tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.10)
    ok_cycle, _, _ = match_labels_in_cycle(
        tokens=tokens,
        vertices=vertices,
        target_labels=["A", "B", "C", "D"],
        max_dist=label_radius,
        allow_reversed=True,
        min_conf=0.10,
        single_char_only=False,
    )
    if not ok_cycle:
        ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
        ink_hits = [_vertex_label_ink(img, v, edge_lines, min_hw) for v in vertices]
        if not (len(ink_hits) == 4 and all(val >= ink_th for val in ink_hits)):
            return [False, f"Failed to detect rectangle vertex labels A/B/C/D around the shape (label_ink={ink_hits}). "]

    extra_th = scale_px(min_hw, 0.24, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in edge_lines):
            continue
        extras.append(ln)
    if extras:
        return [False, f"Detected extra dominant line(s) outside rectangle: {len(extras)}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_11,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
