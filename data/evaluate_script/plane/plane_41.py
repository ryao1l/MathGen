import argparse
import itertools
import math

PID = 14
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _token_bbox(token):
    if not isinstance(token, dict):
        return None
    bbox = token.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _token_exact_single_char(token, expected):
    if not isinstance(token, dict):
        return False
    exp = str(expected).upper()[:1]
    if not exp:
        return False
    letters = str(token.get("letters", "")).upper()
    char = str(token.get("char", "")).upper()[:1]
    if len(letters) == 1:
        return letters == exp
    return char == exp and len(letters) <= 1


def _collect_label_pool(tokens, min_hw):
    side_lim = scale_px(min_hw, 0.16, floor_px=0.0)
    area_lim = side_lim * side_lim
    pool = {"A": [], "B": [], "C": []}
    for idx, tok in enumerate(tokens if isinstance(tokens, list) else []):
        if not isinstance(tok, dict):
            continue
        bbox = _token_bbox(tok)
        if bbox is None:
            continue
        bw = float(bbox[2] - bbox[0])
        bh = float(bbox[3] - bbox[1])
        if bw <= 0.0 or bh <= 0.0:
            continue
        if bw > side_lim or bh > side_lim:
            continue
        if bw * bh > area_lim:
            continue
        aspect = bw / max(1e-6, bh)
        if not (0.18 <= aspect <= 5.5):
            continue
        for ch in ("A", "B", "C"):
            if _token_exact_single_char(tok, ch):
                pool[ch].append((idx, tok))
    return pool


def _pick_token_near_vertex(candidates, vertex, max_dist, used_ids):
    best = None
    best_key = None
    for idx, tok in candidates:
        if idx in used_ids:
            continue
        d = token_edge_distance_to_point(tok, vertex)
        if not math.isfinite(float(d)) or float(d) > float(max_dist):
            continue
        conf = float(tok.get("conf", 0.0))
        key = (float(d), -conf)
        if best_key is None or key < best_key:
            best_key = key
            best = (idx, tok, float(d))
    return best


def _build_token_allow_mask(img_shape, tokens, min_hw):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pad = int(round(scale_px(min_hw, 0.012, floor_px=1.0)))
    for tok in tokens if isinstance(tokens, list) else []:
        bbox = _token_bbox(tok)
        if bbox is None:
            continue
        x1 = int(round(max(0.0, bbox[0] - pad)))
        y1 = int(round(max(0.0, bbox[1] - pad)))
        x2 = int(round(min(float(w - 1), bbox[2] + pad)))
        y2 = int(round(min(float(h - 1), bbox[3] + pad)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def _build_expected_allow_mask(img_shape, A, B, C, min_hw, label_tokens):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    band = int(round(scale_px(min_hw, 0.02, floor_px=1.0)))
    for p, q in ((A, B), (A, C), (B, C)):
        x1 = int(round(float(p[0])))
        y1 = int(round(float(p[1])))
        x2 = int(round(float(q[0])))
        y2 = int(round(float(q[1])))
        cv2.line(mask, (x1, y1), (x2, y2), 255, int(2 * band + 1))

    v_rad = int(round(scale_px(min_hw, 0.02, floor_px=1.0)))
    for p in (A, B, C):
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        cv2.circle(mask, (px, py), v_rad, 255, -1)

    token_mask = _build_token_allow_mask(img_shape, label_tokens, min_hw)
    mask = cv2.bitwise_or(mask, token_mask)
    return mask


def _outside_component_stats(img, allow_mask, min_area_ratio=0.00003):
    if img is None or allow_mask is None:
        return 0, 0
    if allow_mask.shape[:2] != img.shape[:2]:
        return 0, 0

    h, w = img.shape[:2]
    _, bw = _gray_and_ink_mask(img)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    ink = (bw > 0).astype(np.uint8)
    outside = (ink & ((allow_mask == 0).astype(np.uint8))).astype(np.uint8)
    outside = cv2.morphologyEx(
        outside,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    num, _, stats, _ = cv2.connectedComponentsWithStats(outside, 8)
    area_th = max(1, scale_area(h, w, min_area_ratio, floor_px=0))
    areas = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= area_th:
            areas.append(area)
    if not areas:
        return 0, 0
    return int(len(areas)), int(max(areas))


def _count_extra_dominant_lines(lines, tri_lines, min_hw):
    long_th = scale_px(min_hw, 0.23, floor_px=0.0)
    count = 0
    for ln in lines if isinstance(lines, list) else []:
        length = float(ln.get("len", 0.0))
        if length < long_th:
            continue
        same = any(line_equivalent(ln, ref, min_hw) for ref in (tri_lines or []))
        if not same:
            count += 1
    return int(count)


def _evaluate_assignment(img, lines, tri_item, perm, label_pool, min_hw):
    verts = tri_item["vertices"]
    v0, v1, v2 = verts[0], verts[1], verts[2]
    A = verts[perm[0]]
    B = verts[perm[1]]
    C = verts[perm[2]]

    ab = _dist(A, B)
    ac = _dist(A, C)
    bc = _dist(B, C)
    mean_eq = 0.5 * (ab + ac)
    if mean_eq <= 1e-6:
        return None
    rel_err = abs(ab - ac) / mean_eq
    if rel_err > 0.08:
        return None



    def _vertex_iso_err(p, q, r):
        pq = _dist(p, q)
        pr = _dist(p, r)
        m = 0.5 * (pq + pr)
        if m <= 1e-6:
            return float("inf")
        return abs(pq - pr) / m

    vertex_iso_errs = [
        _vertex_iso_err(v0, v1, v2),
        _vertex_iso_err(v1, v0, v2),
        _vertex_iso_err(v2, v0, v1),
    ]
    a_vertex_err = float(vertex_iso_errs[perm[0]])
    if a_vertex_err > (min(vertex_iso_errs) + 0.02):
        return None

    side_max = max(ab, ac, bc)
    side_min = min(ab, ac, bc)
    if side_max <= 1e-6 or side_min / side_max < 0.22:
        return None

    ok_ab, r_ab = has_segment_between_points(img, A, B, ratio_th=0.14, trim_ratio=0.03)
    ok_ac, r_ac = has_segment_between_points(img, A, C, ratio_th=0.14, trim_ratio=0.03)
    ok_bc, r_bc = has_segment_between_points(img, B, C, ratio_th=0.14, trim_ratio=0.03)
    if not (ok_ab and ok_ac and ok_bc):
        return None
    vis_mean = (float(r_ab) + float(r_ac) + float(r_bc)) / 3.0
    if vis_mean < 0.62:
        return None

    area_ratio = float(tri_item.get("area", 0.0)) / float(max(1.0, float(img.shape[0]) * float(img.shape[1])))
    if area_ratio < 0.010:
        return None

    label_tol = max(scale_px(min_hw, 0.02, floor_px=0.0), 0.12 * mean_eq)
    used = set()
    tok_a = _pick_token_near_vertex(label_pool["A"], A, label_tol, used)
    if tok_a is None:
        return None
    if float(tok_a[1].get("conf", 0.0)) < 0.35:
        return None
    used.add(tok_a[0])
    tok_b = _pick_token_near_vertex(label_pool["B"], B, label_tol, used)
    if tok_b is None:
        return None
    if float(tok_b[1].get("conf", 0.0)) < 0.35:
        return None
    used.add(tok_b[0])
    tok_c = _pick_token_near_vertex(label_pool["C"], C, label_tol, used)
    if tok_c is None:
        return None
    if float(tok_c[1].get("conf", 0.0)) < 0.35:
        return None
    label_norm = float(tok_a[2] + tok_b[2] + tok_c[2]) / max(1e-6, 3.0 * label_tol)
    if label_norm > 0.45:
        return None

    extras = _count_extra_dominant_lines(lines, tri_item.get("lines", []), min_hw)
    if extras > 0:
        return None

    label_tokens = [tok_a[1], tok_b[1], tok_c[1]]
    allow_mask = _build_expected_allow_mask(
        img_shape=img.shape,
        A=A,
        B=B,
        C=C,
        min_hw=min_hw,
        label_tokens=label_tokens,
    )

    geom_lines = [
        {"seg": (float(A[0]), float(A[1]), float(B[0]), float(B[1]))},
        {"seg": (float(A[0]), float(A[1]), float(C[0]), float(C[1]))},
        {"seg": (float(B[0]), float(B[1]), float(C[0]), float(C[1]))},
    ]
    outside_ratio, outside_px, _ = compute_outside_ink_stats(
        img=img,
        allowed_lines=geom_lines,
        anchor_points=[A, B, C],
        band_ratio=0.02,
        band_floor_px=1,
        anchor_radius_ratio=0.02,
        anchor_radius_floor_px=1,
        extra_allow_mask=allow_mask,
    )
    outside_px_th = scale_area(img.shape[0], img.shape[1], 0.00002, floor_px=0)
    if outside_px > outside_px_th and outside_ratio > 0.06:
        return None

    comp_cnt, comp_largest = _outside_component_stats(
        img=img,
        allow_mask=allow_mask,
        min_area_ratio=0.00003,
    )
    if comp_cnt > 4:
        return None
    if comp_largest > scale_area(img.shape[0], img.shape[1], 0.00035, floor_px=0):
        return None

    score = (
        1.6 * vis_mean
        - 4.8 * rel_err
        + 1.0 * area_ratio
        - 0.25 * float(extras)
        - 0.35 * label_norm
        - 0.30 * float(outside_ratio)
        - 0.08 * float(comp_cnt)
    )
    return {
        "A": A,
        "B": B,
        "C": C,
        "vertices": [A, B, C],
        "score": float(score),
    }

def _triangles_close(v1, v2, tol):
    s1 = sorted([(float(p[0]), float(p[1])) for p in v1], key=lambda p: (p[0], p[1]))
    s2 = sorted([(float(p[0]), float(p[1])) for p in v2], key=lambda p: (p[0], p[1]))
    return all(_dist(s1[i], s2[i]) <= float(tol) for i in range(3))


def _dedup_valid_triangles(items, min_hw):
    tol = scale_px(min_hw, 0.025, floor_px=0.0)
    out = []
    for it in sorted(items, key=lambda z: float(z["score"]), reverse=True):
        merged = False
        for ex in out:
            if _triangles_close(it["vertices"], ex["vertices"], tol):
                merged = True
                break
        if not merged:
            out.append(it)
    return out

def judge_plane_14(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))

    lines = detect_line_segments(img, min_len_ratio=0.08)
    if len(lines) < 3:
        return [False, "Insufficient line structure for triangle detection. "]

    triangles = extract_triangle_candidates_from_lines(
        lines=lines,
        img_shape=img.shape,
        min_len_ratio=0.12,
        top_k=14,
        min_angle_sep_deg=9.0,
        margin_ratio=0.18,
        point_tol_ratio=0.04,
        min_area_ratio=0.006,
        support_t=(-0.60, 1.35),
    )
    if not triangles:
        return [False, "Failed to reconstruct triangle candidate(s). "]

    tokens = extract_global_letter_tokens(img, whitelist="ABC", min_conf=0.10)
    label_pool = _collect_label_pool(tokens, min_hw=min_hw)
    if not label_pool["A"] or not label_pool["B"] or not label_pool["C"]:
        return [False, "Failed to detect clean vertex labels A, B, C near triangle vertices. "]

    valid = []
    for tri in triangles:
        if not isinstance(tri, dict):
            continue
        verts = tri.get("vertices")
        if not (isinstance(verts, list) and len(verts) == 3):
            continue
        for perm in itertools.permutations((0, 1, 2), 3):
            cand = _evaluate_assignment(
                img=img,
                lines=lines,
                tri_item=tri,
                perm=perm,
                label_pool=label_pool,
                min_hw=min_hw,
            )
            if cand is not None:
                valid.append(cand)

    valid = _dedup_valid_triangles(valid, min_hw=min_hw)
    if not valid:
        return [False, "No clean labeled isosceles triangle ABC satisfies AB = AC. "]
    if len(valid) > 1:
        return [False, "Detected multiple clean triangle candidates; expected exactly one triangle ABC. "]
    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_14,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
