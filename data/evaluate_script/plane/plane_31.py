import argparse

PID = 4
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _upper_left_vertex_index(vertices):
    if len(vertices) != 4:
        return None
    cx = sum(float(v[0]) for v in vertices) / 4.0
    cy = sum(float(v[1]) for v in vertices) / 4.0
    scored = []
    for i, v in enumerate(vertices):
        x = float(v[0])
        y = float(v[1])
        in_ul = (x <= cx) and (y <= cy)
        scored.append((0 if in_ul else 1, x + y, y, x, i))
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return int(scored[0][4])


def _labels_match_clockwise_from_upper_left(tokens, vertices, radius):
    target = ["A", "B", "C", "D"]
    if len(vertices) != 4:
        return False, [], []
    start = _upper_left_vertex_index(vertices)
    if start is None:
        return False, [], []
    ordered = [vertices[(start + i) % 4] for i in range(4)]
    got = []
    for i, expect in enumerate(target):
        tok = select_token_near_point(tokens, expected_char=expect, point=ordered[i], max_dist=radius)
        got.append(expect if tok is not None else None)
    return got == target, got, ordered


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


def judge_plane_4(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
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
        return [False, "Failed to reconstruct a closed quadrilateral from detected lines. "]

    edge_lines = quad["lines"]
    vertices = quad["vertices"]

    extra_th = 0.28 * min_hw
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in edge_lines):
            continue
        extras.append(ln)
    if extras:
        return [False, f"Detected extra dominant line(s) outside quadrilateral: {len(extras)}. "]

    violated, info = has_excess_outside_ink(
        img=img,
        allowed_lines=edge_lines,
        anchor_points=vertices,
        max_outside_ratio=0.06,
        max_outside_px_ratio=0.00004,
        max_outside_px_floor=0,
    )
    if violated:
        return [
            False,
            (
                "Detected extra drawing content outside target quadrilateral "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]

    tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.10)
    label_radius = scale_px(min_hw, 0.16)
    ok, got, ordered = _labels_match_clockwise_from_upper_left(tokens, vertices, label_radius)
    if not ok:
        ink_th = max(10, int(0.000006 * h * w))
        ink_hits = [_vertex_label_ink(img, v, edge_lines, min_hw) for v in ordered]
        if len(ink_hits) == 4 and all(val >= ink_th for val in ink_hits):
            return [True, ""]
        ordered_fmt = [(round(float(v[0]), 1), round(float(v[1]), 1)) for v in ordered] if ordered else []
        return [
            False,
            f"Vertex labels must be A->B->C->D clockwise from upper-left vertex "
            f"(detected={got}, label_ink={ink_hits}, ordered_vertices={ordered_fmt}). ",
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_4,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
