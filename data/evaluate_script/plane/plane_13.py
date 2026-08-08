import argparse

PID = 21
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
        raise RuntimeError("Failed to build line from segment points.")
    return {
        "seg": seg,
        "abc": abc,
        "ang": segment_angle_deg(seg),
        "len": float(_dist(p, q)),
    }


def _pick_bc_mapping(tokens, p1, p2, tol):
    def _tok_dist(tok, pt):
        if tok is None:
            return 1e9
        d = token_edge_distance_to_point(tok, pt)
        if not np.isfinite(d):
            return 1e9
        return float(d)

    best = None
    best_key = None
    for b_geo, c_geo in ((p1, p2), (p2, p1)):
        b_tok = select_token_near_point(tokens, expected_char="B", point=b_geo, max_dist=tol)
        c_tok = select_token_near_point(tokens, expected_char="C", point=c_geo, max_dist=tol)
        hit = int(b_tok is not None) + int(c_tok is not None)
        key = (-hit, _tok_dist(b_tok, b_geo) + _tok_dist(c_tok, c_geo))
        if best_key is None or key < best_key:
            best_key = key
            best = (b_geo, c_geo, b_tok, c_tok)
    return best


def _altitude_foot(vertex, opposite_side, min_hw):
    foot = project_point_to_line(vertex, opposite_side["abc"])
    if foot is None:
        return None
    foot = (float(foot[0]), float(foot[1]))
    if not point_on_segment_support(
        opposite_side,
        foot,
        min_hw,
        dist_ratio=0.02,
        dist_floor_px=0.0,
        t_min=-0.60,
        t_max=1.35,
    ):
        return None
    return foot


def judge_plane_21(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 6:
        return [False, "Insufficient line structure for hard case 21. "]

    tri = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=3,
        min_len_ratio=0.20,
        top_k=12,
        min_angle_sep_deg=10.0,
        margin_ratio=0.18,
        point_tol_ratio=0.04,
        min_area_ratio=0.010,
        support_t=(-0.60, 1.35),
    )
    if tri is None:
        return [False, "Failed to reconstruct the outer triangle ABC. "]

    verts = tri["vertices"]
    idx_a = min(range(3), key=lambda i: float(verts[i][1]))
    a_geo = verts[idx_a]
    rest = [verts[i] for i in range(3) if i != idx_a]
    p_left, p_right = sorted(rest, key=lambda p: float(p[0]))

    l1 = _dist(verts[0], verts[1])
    l2 = _dist(verts[1], verts[2])
    l3 = _dist(verts[2], verts[0])
    mean_len = (l1 + l2 + l3) / 3.0
    rel_dev = max(abs(l1 - mean_len), abs(l2 - mean_len), abs(l3 - mean_len)) / max(1e-6, mean_len)
    if rel_dev > 0.12:
        return [False, f"Outer triangle is not equilateral enough (rel_dev={rel_dev:.3f}). "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDEF", min_conf=0.10)
    vertex_tol = scale_px(min_hw, 0.18, floor_px=0.0)
    a_tok = select_token_near_point(tokens, expected_char="A", point=a_geo, max_dist=vertex_tol)
    bc_pick = _pick_bc_mapping(tokens, p_left, p_right, vertex_tol)
    b_geo, c_geo, b_tok, c_tok = bc_pick
    if a_tok is None or b_tok is None or c_tok is None:
        return [
            False,
            f"Failed to detect outer labels A/B/C (A={a_tok is not None}, B={b_tok is not None}, C={c_tok is not None}). ",
        ]

    ab = _segment_item(a_geo, b_geo)
    ac = _segment_item(a_geo, c_geo)
    bc = _segment_item(b_geo, c_geo)

    d_geo = _altitude_foot(a_geo, bc, min_hw)
    e_geo = _altitude_foot(b_geo, ac, min_hw)
    f_geo = _altitude_foot(c_geo, ab, min_hw)
    if d_geo is None or e_geo is None or f_geo is None:
        return [False, "Failed to derive all three altitude feet on opposite sides. "]

    ok_ad, ratio_ad = has_segment_between_points(img, a_geo, d_geo, ratio_th=0.16, thickness=2, trim_ratio=0.02)
    ok_be, ratio_be = has_segment_between_points(img, b_geo, e_geo, ratio_th=0.16, thickness=2, trim_ratio=0.02)
    ok_cf, ratio_cf = has_segment_between_points(img, c_geo, f_geo, ratio_th=0.16, thickness=2, trim_ratio=0.02)
    if not (ok_ad and ok_be and ok_cf):
        return [
            False,
            f"Missing altitude segment(s): AD={ratio_ad:.2f}, BE={ratio_be:.2f}, CF={ratio_cf:.2f}. ",
        ]

    foot_tol = scale_px(min_hw, 0.14, floor_px=0.0)
    d_tok = select_token_near_point(tokens, expected_char="D", point=d_geo, max_dist=foot_tol)
    e_tok = select_token_near_point(tokens, expected_char="E", point=e_geo, max_dist=foot_tol)
    f_tok = select_token_near_point(tokens, expected_char="F", point=f_geo, max_dist=foot_tol)
    if d_tok is None or e_tok is None or f_tok is None:
        return [
            False,
            f"Failed to detect foot labels D/E/F (D={d_tok is not None}, E={e_tok is not None}, F={f_tok is not None}). ",
        ]

    ok_de, ratio_de = has_segment_between_points(img, d_geo, e_geo, ratio_th=0.18, thickness=2, trim_ratio=0.03)
    ok_ef, ratio_ef = has_segment_between_points(img, e_geo, f_geo, ratio_th=0.18, thickness=2, trim_ratio=0.03)
    ok_fd, ratio_fd = has_segment_between_points(img, f_geo, d_geo, ratio_th=0.18, thickness=2, trim_ratio=0.03)
    if not (ok_de and ok_ef and ok_fd):
        return [
            False,
            f"Inner triangle DEF is incomplete: DE={ratio_de:.2f}, EF={ratio_ef:.2f}, FD={ratio_fd:.2f}. ",
        ]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_21,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
