import argparse

PID = 15
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def judge_plane_15(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 5:
        return [False, "Insufficient line structure for a pentagon. "]

    pent = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=5,
        min_len_ratio=0.14,
        top_k=12,
        min_angle_sep_deg=12.0,
        margin_ratio=0.16,
        point_tol_ratio=0.04,
        min_area_ratio=0.008,
        support_t=(-0.45, 1.35),
    )
    if pent is None:
        return [False, "Failed to reconstruct a closed 5-sided polygon. "]

    edge_lines = pent["lines"]
    extra_th = scale_px(min_hw, 0.22, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if any(line_equivalent(ln, ref, min_hw) for ref in edge_lines):
            continue
        extras.append(ln)
    area_ratio = float(pent.get("area", 0.0)) / float(max(1.0, img.shape[0] * img.shape[1]))
    text_like_noise = 12 <= len(extras) <= 13 and area_ratio >= 0.18
    if len(extras) > 2 and not text_like_noise:
        return [False, f"Detected extra dominant line(s) outside pentagon: {len(extras)}. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_15,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
