import argparse

PID = 19
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _rectangle_like(vertices):
    if len(vertices) != 4:
        return False
    for i in range(4):
        a = vertices[i]
        b = vertices[(i + 1) % 4]
        c = vertices[(i + 2) % 4]
        v1 = (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
        v2 = (float(c[0]) - float(b[0]), float(c[1]) - float(b[1]))
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 <= 1e-6 or n2 <= 1e-6:
            return False
        cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        if abs(cosv) > 0.22:
            return False
    return True


def judge_plane_19(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.10)
    if len(lines) < 5:
        return [False, "Insufficient line structure for rectangle + diagonal. "]

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

    vertices = quad["vertices"]
    edge_lines = quad["lines"]
    if not _rectangle_like(vertices):
        return [False, "Detected quadrilateral is not rectangle-like. "]

    diag_pairs = [(vertices[0], vertices[2]), (vertices[1], vertices[3])]
    diag_ok = []
    for p, q in diag_pairs:
        ok, ratio = has_segment_between_points(img, p, q, ratio_th=0.16, thickness=2, trim_ratio=0.06)
        support = find_support_line(lines, p, q, min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.05)
        if support is not None and any(line_equivalent(support, ref, min_hw) for ref in edge_lines):
            support = None
        if ok or support is not None:
            diag_ok.append((p, q, ratio))
    if not diag_ok:
        return [False, "Missing diagonal segment between opposite rectangle vertices. "]

    extra_th = scale_px(min_hw, 0.24, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in edge_lines):
            continue
        extras.append(ln)
    if len(extras) > 1:
        return [False, f"Detected extra dominant line(s) beyond single diagonal: {len(extras)}. "]

    label_radius = scale_px(min_hw, 0.16, floor_px=0.0)
    tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.10)
    ok_cycle, _, cyc_vertices = match_labels_in_cycle(
        tokens=tokens,
        vertices=vertices,
        target_labels=["A", "B", "C", "D"],
        max_dist=label_radius,
        allow_reversed=True,
        min_conf=0.10,
        single_char_only=False,
    )
    if not ok_cycle:
        return [False, "Failed to detect rectangle labels A/B/C/D around the shape. "]

    mapping = {"A": cyc_vertices[0], "B": cyc_vertices[1], "C": cyc_vertices[2], "D": cyc_vertices[3]}
    ok_ac, ratio_ac = has_segment_between_points(img, mapping["A"], mapping["C"], ratio_th=0.16, thickness=2, trim_ratio=0.06)
    ac_support = find_support_line(lines, mapping["A"], mapping["C"], min_hw=min_hw, ang_tol_deg=12.0, dist_ratio=0.05)
    if ac_support is not None and any(line_equivalent(ac_support, ref, min_hw) for ref in edge_lines):
        ac_support = None
    if not ok_ac and ac_support is None:
        return [False, f"Missing diagonal AC between labeled vertices A and C (ratio={ratio_ac:.3f}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_19,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
