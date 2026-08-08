import argparse

PID = 13
TYPE = "plane"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plane_common as _common
from plane_common import *


def judge_plane_13(img):
    if img is None:
        return [False, "Input image is None. "]

    min_hw = float(min(img.shape[:2]))
    lines = detect_line_segments(img, min_len_ratio=0.10)
    if not lines:
        return [False, "Failed to detect line segment structure. "]

    long_th = scale_px(min_hw, 0.40, floor_px=0.0)
    candidates = [it for it in lines if float(it["len"]) >= long_th]
    if not candidates:
        return [False, "Failed to detect a sufficiently long segment PQ. "]

    main = max(candidates, key=lambda t: float(t["len"]))
    P_geo, Q_geo = segment_endpoints_lr(main["seg"])
    M_geo = ((P_geo[0] + Q_geo[0]) * 0.5, (P_geo[1] + Q_geo[1]) * 0.5)

    def _label_ink(point):
        h, w = img.shape[:2]
        px, py = float(point[0]), float(point[1])
        _, bw = _gray_and_ink_mask(img)
        win = scale_px(min_hw, 0.16, floor_px=0.0)
        x1 = int(max(0, px - win))
        x2 = int(min(w, px + win))
        y1 = int(max(0, py - win))
        y2 = int(min(h, py + win))
        if x2 <= x1 or y2 <= y1:
            return 0
        roi = bw[y1:y2, x1:x2] > 0
        yy, xx = np.ogrid[y1:y2, x1:x2]
        a, b, c = [float(v) for v in main["abc"]]
        den = max(1e-6, math.hypot(a, b))
        line_band = np.abs(a * xx.astype(np.float32) + b * yy.astype(np.float32) + c) / den <= max(3.0, 0.012 * min_hw)
        dot_disk = ((xx.astype(np.float32) - px) ** 2 + (yy.astype(np.float32) - py) ** 2) <= (0.025 * min_hw) ** 2
        label_mask = roi & (~line_band) & (~dot_disk)
        if int(label_mask.sum()) <= 0:
            return 0
        num, _, stats, _ = cv2.connectedComponentsWithStats(label_mask.astype(np.uint8), connectivity=8)
        min_area = max(8, int(0.000005 * h * w))
        max_area = max(400, int(0.0040 * h * w))
        areas = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw_box = int(stats[i, cv2.CC_STAT_WIDTH])
            bh_box = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area:
                continue
            if bw_box > 0.16 * min_hw or bh_box > 0.16 * min_hw:
                continue
            areas.append(area)
        return max(areas) if areas else 0

    extra_th = scale_px(min_hw, 0.25, floor_px=0.0)
    extras = []
    for ln in lines:
        if float(ln["len"]) < extra_th:
            continue
        if line_equivalent(ln, main, min_hw):
            continue
        extras.append(ln)
    if extras:
        return [False, f"Detected extra dominant line(s) outside segment PQ: {len(extras)}. "]

    radius = scale_px(min_hw, 0.16, floor_px=0.0)
    tokens = extract_global_letter_tokens(img, whitelist="PQM", min_conf=0.10)

    def _has_char_near(point, expect):
        tok = select_token_near_point(tokens, expected_char=expect, point=point, max_dist=radius)
        return tok is not None

    p_end_has_p = _has_char_near(P_geo, "P")
    p_end_has_q = _has_char_near(P_geo, "Q")
    q_end_has_p = _has_char_near(Q_geo, "P")
    q_end_has_q = _has_char_near(Q_geo, "Q")
    p_ink = _label_ink(P_geo)
    q_ink = _label_ink(Q_geo)
    m_ink = _label_ink(M_geo)
    ink_th = max(10, int(0.000006 * img.shape[0] * img.shape[1]))
    if not (p_end_has_p or p_end_has_q) and p_ink >= ink_th:
        p_end_has_p = True
    if not (q_end_has_p or q_end_has_q) and q_ink >= ink_th:
        q_end_has_q = True
    assign_pq = p_end_has_p and q_end_has_q
    assign_qp = p_end_has_q and q_end_has_p
    if not (assign_pq or assign_qp):
        return [
            False,
            (
                "Failed to detect endpoint labels P/Q "
                f"(P_end:P={p_end_has_p},Q={p_end_has_q}; "
                f"Q_end:P={q_end_has_p},Q={q_end_has_q}). "
            ),
        ]

    m_tok = select_token_near_point(tokens, expected_char="M", point=M_geo, max_dist=radius)
    if m_tok is None and m_ink < ink_th:
        return [False, f"Failed to detect midpoint label M near segment center (label_ink={m_ink}). "]

    violated, info = has_excess_outside_ink(
        img=img,
        allowed_lines=[main],
        anchor_points=[P_geo, Q_geo, M_geo],
        max_outside_ratio=0.10,
        max_outside_px_ratio=0.00004,
        max_outside_px_floor=0,
    )
    if violated and float(info.get("outside_ratio", 0.0)) > 0.65:
        return [
            False,
            (
                "Detected extra drawing content outside target segment PQ "
                f"(outside={info['outside_px']}, ratio={info['outside_ratio']:.3f}). "
            ),
        ]

    pts = detect_marker_points(img)
    if pts:
        near_mid = nearest_point(pts, M_geo)
        if near_mid is not None:
            d_mid = math.hypot(float(near_mid[0]) - M_geo[0], float(near_mid[1]) - M_geo[1])
            if d_mid > scale_px(min_hw, 0.16, floor_px=0.0):
                return [False, "Failed to find midpoint marker near the center of PQ. "]

    return [True, ""]


def evaluate(image_path):
    return evaluate_plane_task(
        image_path=image_path,
        pid=PID,
        judge_fn=judge_plane_13,
        require_ocr=True,
        task_type=TYPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate plane problem {PID}.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    print(evaluate(args.img_path))
