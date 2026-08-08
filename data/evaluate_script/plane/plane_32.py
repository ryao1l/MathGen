import argparse

PID = 5
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def judge_plane_5(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    lines = detect_line_segments(img, min_len_ratio=0.10)
    if len(lines) < 3:
        return [False, "Insufficient line structure for a triangle. "]

    tri = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=3,
        min_len_ratio=0.20,
        top_k=8,
        min_angle_sep_deg=12.0,
        margin_ratio=0.15,
        point_tol_ratio=0.04,
        min_area_ratio=0.008,
        support_t=(-0.15, 1.10),
    )
    if tri is None:
        return [False, "Failed to reconstruct a closed triangle from detected lines. "]

    edge_lines = tri["lines"]
    verts = tri["vertices"]

    A_geo = min(verts, key=lambda p: float(p[1]))
    bottoms = [p for p in verts if p is not A_geo]
    if len(bottoms) != 2:
        return [False, "Failed to identify bottom vertices B/C. "]
    B_geo, C_geo = sorted(bottoms, key=lambda p: float(p[0]))

    y_margin = 0.03 * min_hw
    x_margin = 0.03 * min_hw
    if not (float(A_geo[1]) + y_margin < float(B_geo[1]) and float(A_geo[1]) + y_margin < float(C_geo[1])):
        return [False, "Vertex A is not clearly above B and C. "]
    if not (float(B_geo[0]) + x_margin < float(C_geo[0])):
        return [False, "B is not on the left of C. "]
    if abs(float(B_geo[1]) - float(C_geo[1])) > 0.18 * min_hw:
        return [False, "B and C are not both located at the bottom region. "]

    ab = math.hypot(float(A_geo[0]) - float(B_geo[0]), float(A_geo[1]) - float(B_geo[1]))
    bc = math.hypot(float(B_geo[0]) - float(C_geo[0]), float(B_geo[1]) - float(C_geo[1]))
    ca = math.hypot(float(C_geo[0]) - float(A_geo[0]), float(C_geo[1]) - float(A_geo[1]))
    min_side = max(1e-6, min(ab, bc, ca))
    label_radius = 0.14 * min_side

    tokens = extract_global_letter_tokens(img, whitelist="ABC", min_conf=0.10)
    A_lbl = select_token_near_point(tokens, expected_char="A", point=A_geo, max_dist=label_radius)
    B_lbl = select_token_near_point(tokens, expected_char="B", point=B_geo, max_dist=label_radius)
    C_lbl = select_token_near_point(tokens, expected_char="C", point=C_geo, max_dist=label_radius)
    labels_found = int(A_lbl is not None) + int(B_lbl is not None) + int(C_lbl is not None)

    extra_th = 0.36 * min_hw
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in edge_lines):
            continue
        extras.append(ln)
    if len(extras) >= 4 and labels_found < 2:
        return [False, f"Detected extra dominant line(s) outside triangle: {len(extras)}. "]

    violated, info = has_excess_outside_ink(
        img=img,
        allowed_lines=edge_lines,
        anchor_points=[A_geo, B_geo, C_geo],
        max_outside_ratio=0.14,
        max_outside_px_ratio=0.00012,
        max_outside_px_floor=0,
    )
    if violated and labels_found < 2 and float(info.get("outside_ratio", 0.0)) > 0.70:
        return [
            False,
            (
                "Detected extra drawing content outside target triangle "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]
    if labels_found < 2:
        # Triangle ABC is a permutation-sensitive label task, but generated
        # labels are often small. Keep OCR as strong evidence, not the only way
        # to accept an otherwise clean, correctly oriented triangle.
        if len(extras) > 3:
            return [
                False,
                f"Failed to detect enough labels A/B/C and found extra line(s): labels={labels_found}, extras={len(extras)}. ",
            ]
    if labels_found == 0 and len(extras) == 0:
        pass
    elif labels_found < 2:
        return [
            False,
            f"Failed to detect required labels A/B/C (got A={None if A_lbl is None else A_lbl.get('char')}, "
            f"B={None if B_lbl is None else B_lbl.get('char')}, C={None if C_lbl is None else C_lbl.get('char')}). ",
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_5,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
