import argparse

PID = 10
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _cycle_match(labels_on_vertices, target):
    n = len(target)
    if len(labels_on_vertices) != n:
        return False
    for s in range(n):
        ok = True
        for i in range(n):
            if labels_on_vertices[(s + i) % n] != target[i]:
                ok = False
                break
        if ok:
            return True
    return False


def _cycle_starting_from(labels_on_vertices, anchor):
    seq = list(labels_on_vertices)
    if anchor in seq:
        i = seq.index(anchor)
        return seq[i:] + seq[:i]
    return seq


def _edge_lengths(poly):
    n = len(poly)
    out = []
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        out.append(math.hypot(float(x2) - float(x1), float(y2) - float(y1)))
    return out


def judge_plane_10(img):
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
    vertices_ccw = list(reversed(quad["vertices"]))

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

    target = ["A", "B", "C", "D"]
    tokens = extract_global_letter_tokens(img, whitelist="ABCD", min_conf=0.10)
    best_by_char = pick_best_tokens_by_char(tokens, target, min_conf=0.10, match_mode="exact")
    missing = [ch for ch in target if ch not in best_by_char]
    if missing:
        return [False, f"Missing vertex labels from OCR: {','.join(missing)}. "]

    assign, dists = assign_labels_to_vertices_min_cost(best_by_char, vertices_ccw, target)
    if assign is None or dists is None:
        return [False, "Failed to assign OCR labels to quadrilateral vertices. "]

    edge_scale = float(np.median(np.array(_edge_lengths(vertices_ccw), dtype=np.float32)))
    max_label_dist = 0.42 * edge_scale
    far = [ch for ch in target if float(dists[ch]) > max_label_dist]
    if far:
        details = ",".join(f"{ch}:{dists[ch]:.1f}" for ch in far)
        return [False, f"Detected labels too far from vertices ({details}). "]

    labels_on_ccw = [None] * 4
    for ch in target:
        labels_on_ccw[assign[ch]] = ch
    seq_from_a = _cycle_starting_from(labels_on_ccw, "A")
    if not _cycle_match(labels_on_ccw, target):
        return [False, f"Vertex labels are not in counterclockwise ABCD order (detected_cycle={seq_from_a}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_10,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
