import argparse

PID = 28
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dist(p, q):
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


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


def _annotation_ink_near_point(img, point, min_hw):
    px, py = float(point[0]), float(point[1])
    _, bw = _gray_and_ink_mask(img)
    h, w = bw.shape[:2]
    win = scale_px(min_hw, 0.13, floor_px=0.0)
    x1 = int(max(0, px - win))
    x2 = int(min(w, px + win))
    y1 = int(max(0, py - win))
    y2 = int(min(h, py + win))
    if x2 <= x1 or y2 <= y1:
        return 0

    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    point_disk = (
        (xx.astype(np.float32) - px) ** 2
        + (yy.astype(np.float32) - py) ** 2
    ) <= scale_px(min_hw, 0.045, floor_px=0.0) ** 2
    label_mask = roi & (~point_disk)
    if int(label_mask.sum()) <= 0:
        return 0

    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    best_area = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < max(8, int(0.000004 * h * w)):
            continue
        if area > max(900, int(0.0040 * h * w)):
            continue
        if bw_box > 0.18 * min_hw or bh_box > 0.18 * min_hw:
            continue
        best_area = max(best_area, area)
    return best_area


def judge_plane_28(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.07)
    if len(lines) < 6:
        return [False, "Insufficient line structure for hard case 28. "]

    quad = extract_polygon_from_lines(
        lines=lines,
        img_shape=img.shape,
        sides=4,
        min_len_ratio=0.18,
        top_k=14,
        min_angle_sep_deg=10.0,
        margin_ratio=0.15,
        point_tol_ratio=0.04,
        min_area_ratio=0.010,
        support_t=(-0.50, 1.35),
    )
    if quad is None:
        return [False, "Failed to reconstruct rectangle boundary. "]

    vertices = quad["vertices"]
    metrics = _rectangle_metrics(vertices)
    if metrics is None:
        return [False, "Failed rectangle metrics. "]
    edges, cos_vals = metrics

    if max(cos_vals) > 0.24:
        return [False, "Boundary is not rectangle-like (right-angle check failed). "]

    if min(edges) <= 1e-6:
        return [False, "Degenerate rectangle boundary. "]
    long_e = max(edges)
    short_e = min(edges)
    ratio_wh = float(long_e) / float(short_e)
    if ratio_wh < 1.45 or ratio_wh > 2.85:
        return [False, f"Rectangle width/height ratio is off (ratio={ratio_wh:.3f}). "]

    h_geo = (
        sum(float(v[0]) for v in vertices) / 4.0,
        sum(float(v[1]) for v in vertices) / 4.0,
    )

    ok_d1, r_d1 = has_segment_between_points(img, vertices[0], vertices[2], ratio_th=0.14, thickness=2, trim_ratio=0.02)
    ok_d2, r_d2 = has_segment_between_points(img, vertices[1], vertices[3], ratio_th=0.14, thickness=2, trim_ratio=0.02)
    if not (ok_d1 and ok_d2):
        return [False, f"Missing one or both diagonals (d1={r_d1:.2f}, d2={r_d2:.2f}). "]

    rs = [_dist(v, h_geo) for v in vertices]
    r_exp = sum(rs) / 4.0
    spread = max(abs(r - r_exp) for r in rs) / max(1e-6, r_exp)
    if spread > 0.18:
        return [False, f"Corners are not concyclic enough around center (spread={spread:.3f}). "]

    band = int(max(1, round(scale_px(min_hw, 0.012, floor_px=0.0))))
    cov, vis = _circle_ink_coverage(img, h_geo, r_exp, band)
    if vis < 0.80:
        return [False, "Circle visibility too low for verification. "]
    if cov < 0.55:
        return [False, f"Missing circumcircle through rectangle vertices (coverage={cov:.2f}). "]

    tokens = extract_global_letter_tokens(img, whitelist="ABCDH", min_conf=0.10)
    ok_cycle, detected, _ = match_labels_in_cycle(
        tokens=tokens,
        vertices=vertices,
        target_labels=["A", "B", "C", "D"],
        max_dist=scale_px(min_hw, 0.11, floor_px=0.0),
        allow_reversed=True,
        min_conf=0.10,
        single_char_only=False,
    )
    hits = sum(1 for x in detected if x is not None)
    if not ok_cycle and hits < 3:
        ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
        ink_hits = sum(1 for v in vertices if _annotation_ink_near_point(img, v, min_hw) >= ink_th)
        if ink_hits < 3:
            return [False, f"Failed to align enough vertex labels A/B/C/D (hits={hits}). "]

    h_tok = select_token_near_point(tokens, expected_char="H", point=h_geo, max_dist=scale_px(min_hw, 0.12, floor_px=0.0))
    if h_tok is None:
        ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
        if _annotation_ink_near_point(img, h_geo, min_hw) < ink_th:
            return [False, "Failed to detect center label H near diagonal intersection. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_28,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
