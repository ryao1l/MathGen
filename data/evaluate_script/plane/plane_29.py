import argparse

PID = 1
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def _dedup_dot_candidates(cands, tol):
    if not cands:
        return []
    ordered = sorted(cands, key=lambda t: (-float(t[3]), float(t[2])))
    keep = []
    for x, y, rr, fill in ordered:
        merged = False
        for qx, qy, qr, _ in keep:
            if math.hypot(float(x) - float(qx), float(y) - float(qy)) <= max(float(tol), 0.6 * max(float(rr), float(qr))):
                merged = True
                break
        if not merged:
            keep.append((float(x), float(y), float(rr), float(fill)))
    return keep


def _detect_filled_dot_candidates(img, circle):
    if img is None:
        return []

    h, w = img.shape[:2]
    m = float(min(h, w))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    blur = cv2.medianBlur(gray, 5)

    min_r = max(1, int(round(scale_px(m, 0.0025))))
    max_r = max(min_r + 1, int(round(scale_px(m, 0.014))))
    min_dist = max(1, int(round(scale_px(m, 0.03))))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=100,
        param2=12,
        minRadius=min_r,
        maxRadius=max_r,
    )

    _, bw = _gray_and_ink_mask(img)
    cands = []
    if circles is not None:
        for cc in circles[0]:
            x, y, rr = float(cc[0]), float(cc[1]), float(cc[2])
            sample_r = max(1.0, 0.80 * rr)
            x1 = max(0, int(x - sample_r - 1))
            x2 = min(w, int(x + sample_r + 2))
            y1 = max(0, int(y - sample_r - 1))
            y2 = min(h, int(y + sample_r + 2))
            if x2 <= x1 or y2 <= y1:
                continue
            yy, xx = np.ogrid[y1:y2, x1:x2]
            disk = ((xx - x) ** 2 + (yy - y) ** 2) <= (sample_r * sample_r)
            den = int(disk.sum())
            if den <= 0:
                continue
            ink = int(((bw[y1:y2, x1:x2] > 0) & disk).sum())
            fill_ratio = float(ink) / float(den)
            if fill_ratio < 0.68:
                continue
            cands.append((x, y, rr, fill_ratio))

    marker_rr = scale_px(m, 0.007)
    for px, py in detect_marker_points(img):
        if not has_point_at_point(img, (px, py)):
            continue
        cands.append((float(px), float(py), float(marker_rr), 1.0))

    if circle is not None:
        _, _, cr = [float(v) for v in circle]
        cands = [t for t in cands if float(t[2]) <= 0.16 * cr]

    return _dedup_dot_candidates(cands, tol=scale_px(m, 0.012))


def _select_point_p(dot_candidates, circle):
    if circle is None:
        return None, "Missing circle for point-P selection. "
    if not dot_candidates:
        return None, "No filled dot candidate detected. "

    cx, cy, r = [float(v) for v in circle]
    right_req = 0.20 * r
    outside_req = 0.10 * r
    y_tol = 0.12 * r

    valid = []
    for x, y, rr, fill_ratio in dot_candidates:
        dx = float(x) - cx
        dy = float(y) - cy
        d = math.hypot(dx, dy)
        margin = d - r
        if dx < right_req:
            continue
        if margin < outside_req:
            continue
        if abs(dy) > y_tol:
            continue
        valid.append((float(x), float(y), float(rr), float(fill_ratio), margin, abs(dy)))

    if not valid:
        return None, "No dot satisfies right-side + outside + horizontal alignment constraints. "

    valid.sort(key=lambda t: (t[5], -t[4], t[0]))
    best = valid[0]
    return (best[0], best[1]), ""


def _select_main_circle(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    min_inside = 0.01 * min_hw
    max_r = 0.48 * min_hw
    cands = [detect_largest_circle(img), detect_second_largest_circle(img), detect_third_largest_circle(img)]
    valid = []
    for c in cands:
        if c is None:
            continue
        cx, cy, r = [float(v) for v in c]
        if r < 0.08 * min_hw:
            continue
        if r > max_r:
            continue
        inside = min(cx - r, cy - r, float(w) - (cx + r), float(h) - (cy + r))
        if inside < min_inside:
            continue
        valid.append((cx, cy, r))
    if not valid:
        return None
    valid.sort(key=lambda t: t[2], reverse=True)
    return valid[0]


def _point_to_bbox_distance(point, bbox):
    if point is None or bbox is None:
        return float("inf")
    if len(bbox) != 4:
        return float("inf")
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _select_p_token_near_point(tokens, point, r):
    if not tokens or point is None:
        return None
    edge_pref_th = 0.12 * float(r)
    edge_hard_th = 0.24 * float(r)
    best = None
    best_key = None
    px, py = float(point[0]), float(point[1])
    for t in tokens:
        letters = str(t.get("letters", "")).upper()
        if not letters:
            letters = str(t.get("char", "")).upper()
        if "P" not in letters:
            continue
        edge_d = token_edge_distance_to_point(t, (px, py))
        if edge_d > edge_hard_th:
            continue
        near_band = 0 if edge_d <= edge_pref_th else 1
        key = (near_band, edge_d, -float(t.get("conf", 0.0)))
        if best_key is None or key < best_key:
            best_key = key
            best = t
    return best


def _p_label_ink_near_point(img, point, r):
    if img is None or point is None:
        return 0
    h, w = img.shape[:2]
    px, py = float(point[0]), float(point[1])
    rr = float(r)
    _, bw = _gray_and_ink_mask(img)
    x1 = int(max(0, px - 0.08 * rr))
    x2 = int(min(w, px + 0.45 * rr))
    y1 = int(max(0, py - 0.22 * rr))
    y2 = int(min(h, py + 0.22 * rr))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bw[y1:y2, x1:x2] > 0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    point_disk = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.08 * rr) ** 2
    right_of_point = xx.astype(np.float32) >= px + 0.04 * rr
    label_mask = roi & (~point_disk) & right_of_point
    if int(label_mask.sum()) <= 0:
        return 0
    num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
    min_area = max(8, int(0.000006 * h * w))
    max_area = max(300, int(0.0040 * h * w))
    areas = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw_box > 0.22 * rr or bh_box > 0.22 * rr:
            continue
        areas.append(area)
    return max(areas) if areas else 0


def judge_plane_1(img):
    if img is None:
        return [False, "Input image is None. "]

    h, w = img.shape[:2]
    min_hw = float(min(h, w))
    circle = _select_main_circle(img)
    if circle is None:
        return [False, "Failed to detect a valid main circle. "]
    cx, cy, r = [float(v) for v in circle]
    if r < 0.08 * min_hw:
        return [False, f"Detected circle is too small (r={r:.1f}). "]

    lines = detect_line_segments(img, min_len_ratio=0.18)
    dominant_th = max(0.45 * r, 0.14 * min_hw)
    disallowed_lines = []
    for ln in lines:
        if float(ln["len"]) < dominant_th:
            continue
        x1, y1, x2, y2 = [float(v) for v in ln["seg"]]
        y_mid = 0.5 * (y1 + y2)
        x_lo = min(x1, x2)
        x_hi = max(x1, x2)
        center_aligned = abs(y_mid - cy) <= 0.16 * r
        reaches_circle_or_center = x_lo <= cx + 0.30 * r and x_hi >= cx + 0.75 * r
        horizontal_aux = horizontal_error_deg(ln["ang"]) <= 10.0 and center_aligned and reaches_circle_or_center
        if not horizontal_aux:
            disallowed_lines.append(ln)
    # Auxiliary center/radius lines and page text are common in accepted human labels for this prompt.
    # The decisive structure is the circle plus a right-side, horizontally aligned point P.
    crossing_center_lines = []
    for ln in disallowed_lines:
        if point_line_distance((cx, cy), ln["abc"]) <= 0.10 * r:
            crossing_center_lines.append(ln)
    if len(crossing_center_lines) >= 2:
        return [False, f"Detected multiple non-auxiliary construction lines through the circle ({len(crossing_center_lines)}). "]

    dot_candidates = _detect_filled_dot_candidates(img, circle)
    if not dot_candidates:
        return [False, "Failed to detect point markers. "]

    P, reason = _select_point_p(dot_candidates, circle)
    if P is None:
        return [False, reason]

    tokens = extract_global_letter_tokens(img, whitelist="P", min_conf=0.10)
    p_tok = _select_p_token_near_point(tokens, P, r)
    p_ink = _p_label_ink_near_point(img, P, r)
    if p_tok is None and p_ink < max(10, int(0.000006 * h * w)):
        if len(disallowed_lines) > 1:
            return [False, f"Failed to detect label 'P' near the outside point (label_ink={p_ink}). "]
        margin = math.hypot(float(P[0]) - cx, float(P[1]) - cy) - r
        if margin < 0.18 * r:
            return [False, f"Failed to detect label 'P' near the outside point (label_ink={p_ink}). "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_1,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
