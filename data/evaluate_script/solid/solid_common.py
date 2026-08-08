import math
import re
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


def _ang_diff_line(a, b):
    d = abs(a - b)
    return min(d, math.pi - d)


def _ang_diff_full(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _intersect_lines(a1, a2, b1, b2):
    d1x, d1y = a2[0] - a1[0], a2[1] - a1[1]
    d2x, d2y = b2[0] - b1[0], b2[1] - b1[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None
    dx, dy = b1[0] - a1[0], b1[1] - a1[1]
    t = (dx * d2y - dy * d2x) / denom
    x = a1[0] + t * d1x
    y = a1[1] + t * d1y
    return (x, y)

def _refine_circle_radius_by_inner_outer_edges(img, circle, n_angles=180):
    if circle is None:
        return None
    cx, cy, r0 = float(circle[0]), float(circle[1]), float(circle[2])
    if r0 <= 1.0:
        return (cx, cy, r0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    h, w = bw.shape[:2]
    search = float(max(8.0, 0.12 * r0))
    r_min = max(2.0, r0 - search)
    r_max = min(float(max(h, w)) * 2.0, r0 + search)
    if r_max <= r_min + 2.0:
        return (cx, cy, r0)

    ang = np.linspace(0.0, 2.0 * math.pi, int(max(30, n_angles)), endpoint=False)
    inner_rs = []
    outer_rs = []

    rs = np.arange(r_min, r_max + 1.0, 1.0, dtype=np.float32)
    min_thick = int(max(2.0, 0.01 * r0))

    for th in ang:
        xs = cx + rs * math.cos(th)
        ys = cy + rs * math.sin(th)
        xi = np.round(xs).astype(np.int32)
        yi = np.round(ys).astype(np.int32)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if ok.sum() < max(8, min_thick + 2):
            continue
        xi = xi[ok]
        yi = yi[ok]
        rsv = rs[ok]
        vals = (bw[yi, xi] > 0).astype(np.uint8)
        if vals.max() == 0:
            continue

        idx = np.where(vals > 0)[0]
        if idx.size == 0:
            continue
        cuts = np.where(np.diff(idx) > 1)[0]
        runs = np.split(idx, cuts + 1)
        best_run = None
        best_d = None
        for run in runs:
            if run.size < min_thick:
                continue
            rin = float(rsv[run[0]])
            rout = float(rsv[run[-1]])
            rmid = 0.5 * (rin + rout)
            d = abs(rmid - r0)
            if best_d is None or d < best_d:
                best_d = d
                best_run = (rin, rout)
        if best_run is None:
            continue
        rin, rout = best_run
        if rout <= rin:
            continue
        inner_rs.append(rin)
        outer_rs.append(rout)

    if len(inner_rs) < 12 or len(outer_rs) < 12:
        return (cx, cy, r0)

    r_in = float(np.median(np.array(inner_rs, dtype=np.float32)))
    r_out = float(np.median(np.array(outer_rs, dtype=np.float32)))
    if not (r_out > r_in + 0.5):
        return (cx, cy, r0)
    r_mid = 0.5 * (r_in + r_out)
    if abs(r_mid - r0) > max(12.0, 0.20 * r0):
        return (cx, cy, r0)
    return (cx, cy, r_mid)


def _merge_circle_candidates(candidates, center_tol, radius_tol):
    merged = []
    for x, y, r, s in candidates:
        placed = False
        for i, (mx, my, mr, ms) in enumerate(merged):
            if math.hypot(x - mx, y - my) <= center_tol and abs(r - mr) <= radius_tol:
                if s > ms:
                    merged[i] = (x, y, r, s)
                placed = True
                break
        if not placed:
            merged.append((x, y, r, s))
    return merged


def _extract_edge_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    ys, xs = np.where(edges > 0)
    if xs.size < 300:
        return None, None, None, None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    h, w = gray.shape[:2]
    return pts, h, w, min(h, w)


def _circle_from_3pts(a, b, c):
    x1, y1 = a;
    x2, y2 = b;
    x3, y3 = c
    A = x1 - x2;
    B = y1 - y2;
    C = x1 - x3;
    D = y1 - y3
    E = (x1 * x1 - x2 * x2 + y1 * y1 - y2 * y2) / 2.0
    F = (x1 * x1 - x3 * x3 + y1 * y1 - y3 * y3) / 2.0
    det = A * D - B * C
    if abs(det) < 1e-6:
        return None
    cx = (D * E - B * F) / det
    cy = (-C * E + A * F) / det
    r = math.hypot(cx - x1, cy - y1)
    return float(cx), float(cy), float(r)


def _best_circle_from_pts(pts, rng, min_r, max_r, iters=3000, score_th=120):
    if pts is None or pts.shape[0] < 200:
        return None, None, 0
    best = None
    best_mask = None
    best_score = 0
    for _ in range(iters):
        i1, i2, i3 = rng.choice(pts.shape[0], 3, replace=False)
        c = _circle_from_3pts(pts[i1], pts[i2], pts[i3])
        if c is None:
            continue
        cx, cy, r = c
        if r < min_r or r > max_r:
            continue
        d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        tol = max(2.5, 0.01 * r)
        mask_score = np.abs(d - r) <= tol
        score = int(np.sum(mask_score))
        if score > best_score:
            best_score = score
            best = (cx, cy, r)
            band = max(4.0, 0.03 * r)
            best_mask = np.abs(d - r) <= band
    if best is None or best_score < score_th:
        return None, None, 0
    return best, best_mask, best_score


def _find_top_k_circles_from_pts(pts, m, k, min_r=6, max_r=0, seed=0, iters=3000, score_th=120):
    if pts is None:
        return [], []
    if max_r == 0:
        max_r = int(0.60 * m)
    rng = np.random.default_rng(seed)
    found = []
    scores = []
    cur = pts
    for _ in range(max(1, k)):
        c, mask, score = _best_circle_from_pts(cur, rng, min_r, max_r, iters=iters, score_th=score_th)
        if c is None:
            break
        found.append(c)
        scores.append(score)
        cur = cur[~mask]
    return found, scores


def _find_top_k_circles(img, k, min_r=6, max_r=0, seed=0, iters=3000, score_th=120):
    pts, h, w, m = _extract_edge_points(img)
    if pts is None:
        return [], []
    return _find_top_k_circles_from_pts(pts, m, k, min_r=min_r, max_r=max_r, seed=seed, iters=iters, score_th=score_th)


def detect_circle(img, order=1, min_r=6, max_r=0):
    found, scores = _find_top_k_circles(img, k=order, min_r=min_r, max_r=max_r)
    idx = order - 1
    if idx < 0 or idx >= len(found):
        return None
    cx, cy, r = found[idx]
    cx, cy, r = _refine_circle_radius_by_inner_outer_edges(img, (cx, cy, r))
    return (float(cx), float(cy), float(r))


def detecting_concentric_circles(img, t_circle, order):
    x0, y0, r0 = t_circle
    found, scores = _find_top_k_circles(img, k=80, min_r=6, max_r=0, seed=1, iters=4500, score_th=25)
    center_tol = max(4.0, 0.02 * r0)
    band = max(6.0, 0.035 * r0)
    raw = []
    for (cx, cy, r), s in zip(found, scores):
        if math.hypot(cx - x0, cy - y0) > center_tol:
            continue
        if abs(r - r0) <= band:
            continue
        if r >= r0:
            continue
        raw.append((float(cx), float(cy), float(r), int(s)))
    raw.sort(key=lambda t: (r0 - t[2], -t[3]))
    merged = _merge_circle_candidates(raw, center_tol=center_tol, radius_tol=band)
    merged.sort(key=lambda t: (r0 - t[2], -t[3]))
    idx = order - 1
    if idx < 0 or idx >= len(merged):
        return None

    refined = _refine_circle_radius_by_inner_outer_edges(img, (merged[idx][0], merged[idx][1], merged[idx][2]))
    return refined


def detect_circles_between_two_circles(img, c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    R = max(r1, r2)
    r = min(r1, r2)
    found, scores = _find_top_k_circles(img, k=120, min_r=6, max_r=0, seed=2, iters=4500, score_th=25)
    tol_band = max(6.0, 0.03 * R)
    center_tol = max(5.0, 0.02 * R)
    rad_merge = max(6.0, 0.04 * (R - r))
    circles = []
    for (x, y, rr), s in zip(found, scores):
        d = math.hypot(x - cx, y - cy)
        if (d - rr) < (r - tol_band):
            continue
        if (d + rr) > (R + tol_band):
            continue
        circles.append((float(x), float(y), float(rr), int(s)))
    circles.sort(key=lambda t: (-t[3], -t[2]))
    merged = _merge_circle_candidates(circles, center_tol=center_tol, radius_tol=rad_merge)
    merged.sort(key=lambda t: t[2], reverse=True)
    merged = [(x, y, rr) for (x, y, rr, s) in merged]
    refined_merged = [_refine_circle_radius_by_inner_outer_edges(img, c) for c in merged]
    refined_merged = [c for c in refined_merged if c is not None]
    return refined_merged


def detect_largest_circle(img):
    return detect_circle(img, order=1)


def detect_second_largest_circle(img):
    return detect_circle(img, order=2)


def detect_third_largest_circle(img):
    return detect_circle(img, order=3)


def detect_annulus(img):
    outer = detect_largest_circle(img)
    if outer is None:
        return None
    inner = detecting_concentric_circles(img, outer, order=1)
    if inner is None:
        return None
    return {"outer": outer, "inner": inner}


def detect_points_on_circle(img, circle):
    x0, y0, r0 = circle

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    h, w = bw.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    bi, bo = float(max(12.0, 0.03 * r0)), float(max(6.0, 0.03 * r0))
    ring = (((dist >= (r0 - bi)) & (dist <= (r0 + bo))).astype(np.uint8) * 255)
    bw = cv2.bitwise_and(bw, ring)

    rd = float(max(4.0, 0.03 * r0))
    n = int(max(60, round(2.0 * math.pi * r0 / max(1.0, 0.6 * rd))))
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    score = np.zeros(n, np.float32)

    all_c = []
    all_x = []
    all_y = []
    all_score = []
    for i, th in enumerate(ang):
        cx, cy = x0 + r0 * math.cos(th), y0 + r0 * math.sin(th)
        x1, x2 = max(0, int(cx - rd - 1)), min(w, int(cx + rd + 2))
        y1, y2 = max(0, int(cy - rd - 1)), min(h, int(cy + rd + 2))
        if x2 <= x1 or y2 <= y1:
            continue
        yy, xx = np.ogrid[y1:y2, x1:x2]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rd * rd)
        den = int(disk.sum())
        if den:
            score[i] = float((bw[y1:y2, x1:x2] > 0)[disk].sum()) / float(den)
        all_c.append((cx, cy))
        all_x.append(cx)
        all_y.append(cy)
        all_score.append(score[i])

    threshold = 0.95
    m = score >= threshold

    if not m.any():
        dbg = img.copy()
        cv2.circle(dbg, (int(x0), int(y0)), int(r0), (0, 255, 0), 2)
        return []

    idx = np.where(m)[0]
    cuts = np.where(np.diff(idx) > 1)[0]
    segs = np.split(idx, cuts + 1)
    if m[0] and m[-1] and len(segs) > 1:
        segs = [np.r_[segs[-1], segs[0]]] + segs[1:-1]

    pts = []
    for s in segs:
        ss = np.sort(s).astype(int)
        if m[0] and m[-1] and len(segs) > 1 and ((ss == 0).any() or (ss == (len(ang) - 1)).any()):
            ss = np.where(ss < (len(ang) // 2), ss + len(ang), ss)
            ss = np.sort(ss)
        i1, i2 = int(ss[0] % len(ang)), int(ss[-1] % len(ang))
        th1, th2 = float(ang[i1]), float(ang[i2])
        if th2 < th1:
            th2 += 2.0 * math.pi
        thm = 0.5 * (th1 + th2)
        pts.append((x0 + r0 * math.cos(thm), y0 + r0 * math.sin(thm)))

    pts.sort(key=lambda p: -math.atan2(p[1] - y0, p[0] - x0))
    return pts


def detect_points_on_line(img, line, mode="internal"):
    (x1, y1), (x2, y2) = line
    x1 = float(x1);
    y1 = float(y1);
    x2 = float(x2);
    y2 = float(y2)
    vx = x2 - x1
    vy = y2 - y1
    denom = vx * vx + vy * vy
    if img is None or denom <= 1e-6:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    h, w = bw.shape[:2]
    Y, X = np.ogrid[:h, :w]
    seg_len = float(math.hypot(vx, vy))
    band = float(max(12.0, 0.03 * seg_len))
    dist_line = np.abs((X - x1) * vy - (Y - y1) * vx) / float(max(1e-6, seg_len))
    tmap = (((X - x1) * vx + (Y - y1) * vy) / float(denom)).astype(np.float32)
    margin = float(max(0.08, band / float(max(1.0, seg_len))))

    if mode == "internal":
        keep_t = (tmap >= (0.0 - margin)) & (tmap <= (1.0 + margin))
    elif mode == "external":
        keep_t = (tmap < (0.0 - margin)) | (tmap > (1.0 + margin))
    elif mode == "all":
        keep_t = np.ones_like(tmap, dtype=bool)
    else:
        raise ValueError("Invalid mode. Use 'internal', 'external', or 'all'.")

    band_mask = ((dist_line <= band) & keep_t).astype(np.uint8) * 255
    bw = cv2.bitwise_and(bw, band_mask)

    rd = float(max(6.0, 0.04 * seg_len))
    n = int(max(80, round(seg_len / max(1.0, 0.6 * rd))))
    ts = np.linspace(0.0, 1.0, n, endpoint=True)
    score = np.zeros(n, np.float32)

    for i, t in enumerate(ts):
        cx = x1 + t * vx
        cy = y1 + t * vy
        xL, xR = max(0, int(cx - rd - 1)), min(w, int(cx + rd + 2))
        yT, yB = max(0, int(cy - rd - 1)), min(h, int(cy + rd + 2))
        if xR <= xL or yB <= yT:
            continue
        yy, xx = np.ogrid[yT:yB, xL:xR]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rd * rd)
        den = int(disk.sum())
        if den:
            score[i] = float((bw[yT:yB, xL:xR] > 0)[disk].sum()) / float(den)

    thr = 0.95
    m = score >= thr

    pts = []
    if m.any():
        idx = np.where(m)[0]
        cuts = np.where(np.diff(idx) > 1)[0]
        segs = np.split(idx, cuts + 1)
        for s in segs:
            ss = np.sort(s).astype(int)
            tmid = float(0.5 * (ts[ss[0]] + ts[ss[-1]]))
            pts.append((x1 + tmid * vx, y1 + tmid * vy))
        pts.sort(key=lambda p: ((p[0] - x1) * vx + (p[1] - y1) * vy))


    return pts


def has_point_at_point(img, point):
    if point is None: return False
    px, py = int(round(float(point[0]))), int(round(float(point[1])))
    h, w = img.shape[:2]
    if not (0 <= px < w and 0 <= py < h): return False

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    s = float(min(h, w))
    r = int(max(1.5, 0.008 * s))
    n = 96
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = (px + r * np.cos(th)).round().astype(np.int32)
    ys = (py + r * np.sin(th)).round().astype(np.int32)
    m = (0 <= xs) & (xs < w) & (0 <= ys) & (ys < h)
    cover = float((bw[ys[m], xs[m]] > 0).mean()) if m.any() else 0.0
    ok = cover >= 0.95
    return ok


def detect_nearest_letter(img, point, whitelist=None):
    px, py = float(point[0]), float(point[1])
    h, w = img.shape[:2]
    roi = int(0.05 * min(h, w))
    rect_w0 = int(max(24, 1.35 * roi))
    rect_h0 = int(max(24, 1.10 * roi))

    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray0 = cv2.GaussianBlur(gray0, (3, 3), 0)
    bw0 = cv2.bitwise_not(cv2.threshold(gray0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw0 = cv2.morphologyEx(bw0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    px_i, py_i = int(round(px)), int(round(py))
    if 0 <= px_i < w and 0 <= py_i < h and bw0[py_i, px_i] > 0:
        labels = cv2.connectedComponents((bw0 > 0).astype(np.uint8), connectivity=8)[1]
        lab = int(labels[py_i, px_i])
        if lab > 0:
            comp = (labels == lab).astype(np.uint8) * 255
            comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            bw0[comp > 0] = 0

    best = best_conf = best_box = best_crop = best_crop_for_model = None
    best_key = None
    early_stop = False

    n = 60
    pad_out = int(max(2, 0.10 * roi))

    def _clip(x1, y1, x2, y2):
        x1 = int(max(0, min(w, x1)));
        x2 = int(max(0, min(w, x2)))
        y1 = int(max(0, min(h, y1)));
        y2 = int(max(0, min(h, y2)))
        return None if (x2 <= x1 or y2 <= y1) else (x1, y1, x2, y2)

    def _edge_all_black(crop_bw):
        if crop_bw.shape[0] < 2 or crop_bw.shape[1] < 2:
            return False
        edge = np.concatenate([crop_bw[0, :], crop_bw[-1, :], crop_bw[:, 0], crop_bw[:, -1]])
        return (edge > 0).sum() == 0

    wl = whitelist if whitelist else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cfg = f"--psm 10 -c tessedit_char_whitelist={wl}"

    rect_s0 = int(max(24, round((rect_w0 * rect_h0) ** 0.5)))
    rect_scales = [2.0, 1.5, 1.2, 1.0, 0.85]

    for s in rect_scales:
        rect_w = rect_h = int(max(24, round(rect_s0 * s)))
        rect_r = 0.5 * float(rect_w)

        for k in range(n):
            th = 2.0 * math.pi * (k / float(n))
            cx = px + math.cos(th) * (rect_r + pad_out)
            cy = py + math.sin(th) * (rect_r + pad_out)

            box = _clip(cx - rect_w * 0.5, cy - rect_h * 0.5, cx + rect_w * 0.5, cy + rect_h * 0.5)
            if box is None:
                continue
            x1c, y1c, x2c, y2c = box

            crop_bw = bw0[y1c:y2c, x1c:x2c].copy()
            if crop_bw.size == 0:
                continue

            if pytesseract is None:
                continue

            crop_for_model = crop_bw
            o = pytesseract.image_to_data(crop_for_model, config=cfg, output_type=pytesseract.Output.DICT)
            txts = [t.strip().upper() for t in o.get("text", []) if t and t.strip()]
            confs = [float(c) for c, t in zip(o.get("conf", []), o.get("text", []))
                     if t and str(t).strip() and str(c).replace(".", "", 1).lstrip("-").isdigit()]

            cur_best = None
            cur_conf = None
            if txts:
                t = txts[0].replace(" ", "").replace("\n", "")
                if t:
                    ch = t[0]
                    if 'A' <= ch <= 'Z':
                        cur_best = ch
                        cur_conf = confs[0] if confs else None

            if cur_best is None:
                continue

            edge_ok = _edge_all_black(crop_bw)
            if cur_conf is not None:
                cur_conf = float(cur_conf)
                if not edge_ok:
                    cur_conf = cur_conf - 100.0

            if best_key is None or cur_conf > best_key:
                best_key = cur_conf
                best, best_conf = cur_best, cur_conf
                best_box = (x1c, y1c, x2c, y2c)
                best_crop = crop_bw
                best_crop_for_model = crop_for_model

            if cur_conf is not None and cur_conf >= 70.0:
                early_stop = True
                break

        if early_stop:
            break

    return best


def circle_tangent_to_circle(c1, c2, eps, mode):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = math.hypot(x1 - x2, y1 - y2)
    e_ext = abs(d - (r1 + r2))
    e_int = abs(d - abs(r1 - r2))
    if mode == "external":
        return (e_ext <= eps), e_ext
    elif mode == "internal":
        return (e_int <= eps), e_int
    else:
        raise ValueError("Invalid mode for tangency check. Use 'external' or 'internal'. ")


def get_mid_point(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def get_perpendicular_line_through_point(start, end, order, d=50.0):
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx = x2 - x1
    dy = y2 - y1
    n = math.hypot(dx, dy)
    if n <= 1e-9:
        return (x2, y2), (x2, y2)
    o = order.strip().lower()
    cw = ("cw", "clockwise")
    ccw = ("ccw", "counterclockwise")
    if o in cw:
        px, py = (-dy / n), (dx / n)
    elif o in ccw:
        px, py = (dy / n), (-dx / n)
    else:
        raise ValueError("Invalid order. Use 'clockwise'/'counterclockwise' (or 'cw'/'ccw'). ")
    ex = x2 + px * float(d)
    ey = y2 + py * float(d)
    return (x2, y2), (ex, ey)


def get_intersection_of_lines(line1, line2, tol=1e-9):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    x1 = float(x1);
    y1 = float(y1);
    x2 = float(x2);
    y2 = float(y2)
    x3 = float(x3);
    y3 = float(y3);
    x4 = float(x4);
    y4 = float(y4)
    rx = x2 - x1
    ry = y2 - y1
    sx = x4 - x3
    sy = y4 - y3
    nr = math.hypot(rx, ry)
    ns = math.hypot(sx, sy)
    if nr <= 1e-12 or ns <= 1e-12:
        return None
    denom = rx * sy - ry * sx
    if abs(denom) <= tol * nr * ns:
        return None
    qpx = x3 - x1
    qpy = y3 - y1
    t = (qpx * sy - qpy * sx) / denom
    return (x1 + t * rx, y1 + t * ry)


def has_line_between_points(img, p1, p2, allow_dash=True):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    h, w = img.shape[:2]
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L <= 1e-6:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    ink = (bw > 0).astype(np.uint8) * 255
    dt = cv2.distanceTransform(255 - ink, cv2.DIST_L2, 3)
    r = float(max(2.5, 0.006 * min(h, w)))
    n = int(max(25.0, round(L)))
    hit = 0
    for i in range(n + 1):
        t = float(i) / float(max(1, n))
        x = x1 + t * dx
        y = y1 + t * dy
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        if float(dt[yi, xi]) <= r:
            hit += 1

    total = int(n + 1)
    hit_ratio = float(hit) / float(max(1, total))
    thr = 0.40 if allow_dash else 0.90
    ok = hit_ratio >= thr
    return ok


def get_point_position_relative_to_circle(point, circle):
    px, py = float(point[0]), float(point[1])
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    d = math.hypot(px - cx, py - cy)
    if abs(d - r) <= 1e-6:
        return "on"
    elif d < r:
        return "inside"
    else:
        return "outside"



def detect_largest_parallelogram(img, min_area_ratio=0.01):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape[:2]
    diag = math.hypot(w, h)
    min_len = max(40.0, 0.08 * diag)

    lines = detect_solid_lines(img, min_length=min_len)
    if lines is None or len(lines) < 2:
        return None

    bottom_line, parallel_line = find_parallel_line_pair(
        lines, img_size=(w, h), strategy="bottom")
    if bottom_line is None or parallel_line is None:
        return None


    bx1, by1, bx2, by2 = bottom_line
    px1, py1, px2, py2 = parallel_line
    base_ang = math.atan2(by2 - by1, bx2 - bx1)

    connectors = []
    for seg in lines:
        da = abs(math.atan2(seg[3] - seg[1], seg[2] - seg[0]) - base_ang)
        da = min(da, math.pi - da)
        if math.degrees(da) > 15.0:
            connectors.append(seg)

    if len(connectors) < 2:
        return None


    valid_connectors = []
    for seg in connectors:
        sp1 = (seg[0], seg[1])
        sp2 = (seg[2], seg[3])
        ib = _intersect_lines(sp1, sp2, (bx1, by1), (bx2, by2))
        ip = _intersect_lines(sp1, sp2, (px1, py1), (px2, py2))
        if ib is None or ip is None:
            continue
        margin = 0.1 * max(w, h)
        if not (-margin < ib[0] < w + margin and -margin < ib[1] < h + margin):
            continue
        if not (-margin < ip[0] < w + margin and -margin < ip[1] < h + margin):
            continue
        if has_line_between_points(img, ib, ip, allow_dash=False):
            valid_connectors.append((ib, ip, seg))

    if len(valid_connectors) < 2:
        return None

    valid_connectors.sort(key=lambda c: 0.5 * (c[0][0] + c[1][0]))

    left_c = valid_connectors[0]
    right_c = valid_connectors[-1]

    if (by1 + by2) / 2.0 >= (py1 + py2) / 2.0:
        bl, tl = left_c[0], left_c[1]
        br, tr = right_c[0], right_c[1]
    else:
        bl, tl = left_c[1], left_c[0]
        br, tr = right_c[1], right_c[0]

    pts = [bl, br, tr, tl]

    area = _shoelace_area_4(pts)
    if area < min_area_ratio * w * h:
        return None

    pts = _sort_quad_vertices(pts)
    return pts


def _shoelace_area_4(pts):
    a = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        a += x1 * y2 - x2 * y1
    return abs(0.5 * a)


def _sort_quad_vertices(pts):
    cx_ = sum(p[0] for p in pts) / 4.0
    cy_ = sum(p[1] for p in pts) / 4.0
    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy_, p[0] - cx_))
    min_idx = 0
    min_val = float('inf')
    for i, p in enumerate(pts):
        val = p[0] + p[1]
        if val < min_val:
            min_val = val
            min_idx = i
    pts = pts[min_idx:] + pts[:min_idx]
    return pts


def quad_area(pts):
    if pts is None or len(pts) != 4:
        return 0.0
    return _shoelace_area_4(pts)


def check_quad_edges(img, pts, allow_dash=False, min_pass=None):
    if pts is None or len(pts) != 4:
        return False, 0, 0, ["quad"]
    edges = [
        (pts[0], pts[1]),
        (pts[1], pts[2]),
        (pts[2], pts[3]),
        (pts[3], pts[0]),
    ]
    n_pass = 0
    failed = []
    for i, (a, b) in enumerate(edges):
        if has_line_between_points(img, a, b, allow_dash=allow_dash):
            n_pass += 1
        else:
            failed.append(f"e{i}")
    n_total = len(edges)
    if min_pass is None:
        min_pass = max(1, int(n_total * 3 / 4))
    return n_pass >= min_pass, n_pass, n_total, failed


def detect_label_near_line(img, p1, p2, label, samples=9):
    if img is None:
        return None
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if samples < 3:
        samples = 3
    ts = np.linspace(0.15, 0.85, samples)
    for t in ts:
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        try:
            ch = detect_nearest_letter(img, (cx, cy), whitelist=label)
        except Exception:
            ch = None
        if ch == label:
            return (cx, cy)
    return None


def detect_labels_near_lines(img, line_map, samples=9):
    found = {}
    missing = []
    for label, (p1, p2) in line_map.items():
        pt = detect_label_near_line(img, p1, p2, label, samples=samples)
        if pt is None:
            missing.append(label)
        else:
            found[label] = pt
    return found, missing


def front_face_has_diagonals(img, pts):
    if pts is None or len(pts) != 4:
        return False
    diag_02 = has_line_between_points(img, pts[0], pts[2], allow_dash=False)
    diag_13 = has_line_between_points(img, pts[1], pts[3], allow_dash=False)
    return diag_02 and diag_13


def parallelogram_from_parallel_pair(img, base_line, parallel_line, lines, w, h,
                                     min_area_ratio=0.01):
    bx1, by1, bx2, by2 = base_line
    px1, py1, px2, py2 = parallel_line
    base_ang = math.atan2(by2 - by1, bx2 - bx1)



    connectors = []
    for seg in lines:
        ang = math.atan2(seg[3] - seg[1], seg[2] - seg[0])
        if math.degrees(_ang_diff_line(ang, base_ang)) > 15.0:
            connectors.append(seg)

    if len(connectors) < 2:
        return None

    valid_connectors = []
    for seg in connectors:
        sp1 = (seg[0], seg[1])
        sp2 = (seg[2], seg[3])
        ib = _intersect_lines(sp1, sp2, (bx1, by1), (bx2, by2))
        ip = _intersect_lines(sp1, sp2, (px1, py1), (px2, py2))
        if ib is None or ip is None:
            continue
        margin = 0.1 * max(w, h)
        if not (-margin < ib[0] < w + margin and -margin < ib[1] < h + margin):
            continue
        if not (-margin < ip[0] < w + margin and -margin < ip[1] < h + margin):
            continue
        if has_line_between_points(img, ib, ip, allow_dash=False):
            valid_connectors.append((ib, ip))

    if len(valid_connectors) < 2:
        return None

    valid_connectors.sort(key=lambda c: 0.5 * (c[0][0] + c[1][0]))
    left_c = valid_connectors[0]
    right_c = valid_connectors[-1]

    if (by1 + by2) / 2.0 >= (py1 + py2) / 2.0:
        bl, tl = left_c[0], left_c[1]
        br, tr = right_c[0], right_c[1]
    else:
        bl, tl = left_c[1], left_c[0]
        br, tr = right_c[1], right_c[0]

    pts = [bl, br, tr, tl]
    area = _shoelace_area_4(pts)
    if area < min_area_ratio * w * h:
        return None
    return _sort_quad_vertices(pts)


def detect_front_face_with_diagonals(img):
    front_pts = detect_largest_parallelogram(img)
    if front_pts is not None and front_face_has_diagonals(img, front_pts):
        return front_pts

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape[:2]
    diag = math.hypot(w, h)
    min_len = max(35.0, 0.06 * diag)

    lines = detect_solid_lines(img, min_length=min_len, max_gap=10, threshold=40)
    if not lines or len(lines) < 4:
        return front_pts

    lines = lines[:25]
    candidates = []

    def _ang(s):
        return math.atan2(s[3] - s[1], s[2] - s[0])

    def _perp_sep(s1, s2):
        mx1, my1 = 0.5 * (s1[0] + s1[2]), 0.5 * (s1[1] + s1[3])
        mx2, my2 = 0.5 * (s2[0] + s2[2]), 0.5 * (s2[1] + s2[3])
        dx, dy = s1[2] - s1[0], s1[3] - s1[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return math.hypot(mx2 - mx1, my2 - my1)
        return abs((mx2 - mx1) * dy - (my2 - my1) * dx) / L

    ang_tol = math.radians(12.0)
    min_sep = max(18.0, 0.04 * diag)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            da = abs(_ang(lines[i]) - _ang(lines[j]))
            da = min(da, math.pi - da)
            if da > ang_tol:
                continue
            if _perp_sep(lines[i], lines[j]) < min_sep:
                continue
            pts = parallelogram_from_parallel_pair(img, lines[i], lines[j], lines, w, h)
            if pts is None:
                continue
            area = _shoelace_area_4(pts)
            diag_02 = has_line_between_points(img, pts[0], pts[2], allow_dash=False)
            diag_13 = has_line_between_points(img, pts[1], pts[3], allow_dash=False)
            diag_score = int(diag_02) + int(diag_13)
            candidates.append((diag_score, area, pts))

    if candidates:
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        return candidates[0][2]

    return front_pts


def detect_bottom_parallelogram(img, min_area_ratio=0.010):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape[:2]
    diag = math.hypot(w, h)
    min_len = max(35.0, 0.06 * diag)

    lines = detect_solid_lines(img, min_length=min_len, max_gap=10, threshold=40)
    if not lines or len(lines) < 4:
        return None
    lines = lines[:30]

    def _ang(s):
        return math.atan2(s[3] - s[1], s[2] - s[0])


    def _perp_sep(s1, s2):
        mx1, my1 = 0.5 * (s1[0] + s1[2]), 0.5 * (s1[1] + s1[3])
        mx2, my2 = 0.5 * (s2[0] + s2[2]), 0.5 * (s2[1] + s2[3])
        dx, dy = s1[2] - s1[0], s1[3] - s1[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return math.hypot(mx2 - mx1, my2 - my1)
        return abs((mx2 - mx1) * dy - (my2 - my1) * dx) / L


    ang_tol = math.radians(12.0)
    min_sep = max(18.0, 0.04 * diag)
    margin = 0.1 * max(w, h)

    candidates = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            da = _ang_diff_line(_ang(lines[i]), _ang(lines[j]))
            if da > ang_tol:
                continue
            if _perp_sep(lines[i], lines[j]) < min_sep:
                continue

            s1, s2 = lines[i], lines[j]
            bx1, by1, bx2, by2 = float(s1[0]), float(s1[1]), float(s1[2]), float(s1[3])
            px1, py1, px2, py2 = float(s2[0]), float(s2[1]), float(s2[2]), float(s2[3])
            base_ang = math.atan2(by2 - by1, bx2 - bx1)

            valid_connectors = []
            for seg in lines:
                ang = math.atan2(seg[3] - seg[1], seg[2] - seg[0])
                if math.degrees(_ang_diff_line(ang, base_ang)) <= 15.0:
                    continue
                sp1 = (seg[0], seg[1])
                sp2 = (seg[2], seg[3])
                ib = _intersect_lines(sp1, sp2, (bx1, by1), (bx2, by2))
                ip = _intersect_lines(sp1, sp2, (px1, py1), (px2, py2))
                if ib is None or ip is None:
                    continue
                if not (-margin < ib[0] < w + margin and -margin < ib[1] < h + margin):
                    continue
                if not (-margin < ip[0] < w + margin and -margin < ip[1] < h + margin):
                    continue
                if has_line_between_points(img, ib, ip, allow_dash=False):
                    valid_connectors.append((ib, ip))

            if len(valid_connectors) < 2:
                continue

            avg_y_s1 = (by1 + by2) / 2.0
            avg_y_s2 = (py1 + py2) / 2.0
            if avg_y_s1 >= avg_y_s2:
                base_line_avg_y = avg_y_s1
                parallel_line_avg_y = avg_y_s2
            else:
                base_line_avg_y = avg_y_s2
                parallel_line_avg_y = avg_y_s1

            for ci in range(len(valid_connectors)):
                for cj in range(ci + 1, len(valid_connectors)):
                    lc = valid_connectors[ci]
                    rc = valid_connectors[cj]

                    if (lc[0][0] + lc[1][0]) > (rc[0][0] + rc[1][0]):
                        lc, rc = rc, lc

                    if avg_y_s1 >= avg_y_s2:
                        bl, br = lc[0], rc[0]   # on s1
                        tl, tr = lc[1], rc[1]   # on s2
                    else:
                        bl, br = lc[1], rc[1]   # on s2
                        tl, tr = lc[0], rc[0]   # on s1

                    pts = _sort_quad_vertices([bl, br, tr, tl])
                    if pts is None:
                        continue
                    area = _shoelace_area_4(pts)
                    if area < min_area_ratio * w * h:
                        continue
                    min_vertex_sep = 0.05 * diag
                    degenerate = False
                    for pi in range(4):
                        for pj in range(pi + 1, 4):
                            if math.hypot(pts[pi][0] - pts[pj][0],
                                          pts[pi][1] - pts[pj][1]) < min_vertex_sep:
                                degenerate = True
                                break
                        if degenerate:
                            break
                    if degenerate:
                        continue
                    min_y = min(p[1] for p in pts)
                    avg_y = sum(p[1] for p in pts) / 4.0
                    candidates.append((min_y, avg_y, area, pts))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    return candidates[0][3]


def detect_back_face_relaxed(img, front_pts):
    back_pts = detect_second_parallelogram(img, front_pts, allow_dash=True)
    if back_pts is not None:
        return back_pts

    if front_pts is None or len(front_pts) != 4:
        return None

    h, w = img.shape[:2]
    diag = math.hypot(w, h)

    all_lines = []
    for params in [
        dict(min_length=max(30.0, 0.05 * diag), max_gap=18, threshold=30),
        dict(min_length=max(24.0, 0.04 * diag), max_gap=28, threshold=20),
        dict(min_length=max(20.0, 0.03 * diag), max_gap=35, threshold=15),
        dict(min_length=max(15.0, 0.02 * diag), max_gap=45, threshold=10),
    ]:
        lines = detect_solid_lines(img, **params)
        if not lines:
            continue
        existing = set((round(s[0]), round(s[1]), round(s[2]), round(s[3]))
                       for s in all_lines)
        for s in lines:
            key = (round(s[0]), round(s[1]), round(s[2]), round(s[3]))
            if key not in existing:
                all_lines.append(s)
                existing.add(key)

    if not all_lines:
        return None


    front_edge_angs = []
    for i in range(4):
        j = (i + 1) % 4
        a = math.atan2(front_pts[j][1] - front_pts[i][1],
                       front_pts[j][0] - front_pts[i][0])
        front_edge_angs.append(a)

    unique_front = []
    for a in front_edge_angs:
        dup = False
        for u in unique_front:
            if _ang_diff_full(a, u) < math.radians(15) or \
               _ang_diff_full(a + math.pi, u) < math.radians(15):
                dup = True
                break
        if not dup:
            unique_front.append(a)

    def _is_front_edge_dir(ang):
        for u in unique_front:
            if _ang_diff_full(ang, u) < math.radians(15) or \
               _ang_diff_full(ang + math.pi, u) < math.radians(15):
                return True
        return False

    search_dist = max(80, int(0.10 * min(h, w)))
    
    connector_info = []
    for fi in range(4):
        fx, fy = front_pts[fi]
        for seg in all_lines:
            x1, y1, x2, y2 = seg
            d1 = math.hypot(fx - x1, fy - y1)
            d2 = math.hypot(fx - x2, fy - y2)
            min_d = min(d1, d2)
            if min_d > search_dist:
                continue
            ang = math.atan2(y2 - y1, x2 - x1)
            if _is_front_edge_dir(ang):
                continue
            far = (x2, y2) if d1 < d2 else (x1, y1)
            connector_info.append((fi, far, ang))

    if not connector_info:
        return None

    angs = [ang for _, _, ang in connector_info]
    if not angs:
        return None
    
    angle_bins = {}
    for ang in angs:
        bin_key = round(math.degrees(ang % math.pi) / 5) * 5
        angle_bins[bin_key] = angle_bins.get(bin_key, 0) + 1
    if not angle_bins:
        return None
    consensus_bin = max(angle_bins, key=angle_bins.get)
    consensus_ang = math.radians(consensus_bin)

    raw_back = [None] * 4
    for fi, far, ang in connector_info:
        norm_check_diff = min(_ang_diff_full(ang % math.pi, consensus_ang % math.pi),
                              _ang_diff_full((ang + math.pi) % (2*math.pi), consensus_ang % math.pi))
        if norm_check_diff > math.radians(20):
            continue
        if raw_back[fi] is None:
            raw_back[fi] = far
        else:
            fx, fy = front_pts[fi]
            d_old = math.hypot(raw_back[fi][0] - fx, raw_back[fi][1] - fy)
            d_new = math.hypot(far[0] - fx, far[1] - fy)
            if d_new > d_old:
                raw_back[fi] = far

    found_vecs = []
    for i in range(4):
        if raw_back[i] is not None:
            vdx = raw_back[i][0] - front_pts[i][0]
            vdy = raw_back[i][1] - front_pts[i][1]
            found_vecs.append((vdx, vdy))

    if len(found_vecs) < 2:
        return None

    dxs = sorted(v[0] for v in found_vecs)
    dys = sorted(v[1] for v in found_vecs)
    n = len(dxs)
    med_dx = 0.5 * (dxs[(n - 1) // 2] + dxs[n // 2])
    med_dy = 0.5 * (dys[(n - 1) // 2] + dys[n // 2])

    back_pts = []
    for i in range(4):
        back_pts.append((front_pts[i][0] + med_dx,
                         front_pts[i][1] + med_dy))

    area = _shoelace_area_4(back_pts)
    if area < 0.004 * w * h:
        return None

    return back_pts


def detect_solid_lines(img, min_length=40.0, max_gap=8, threshold=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    raw = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                          threshold=threshold,
                          minLineLength=int(min_length),
                          maxLineGap=max_gap)
    if raw is None:
        return None

    lines = []
    for seg in raw:
        x1, y1, x2, y2 = seg[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            lines.append((float(x1), float(y1), float(x2), float(y2)))

    if not lines:
        return None

    lines = _merge_collinear_segments(lines, ang_tol=8.0, dist_tol=15.0)
    lines.sort(key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
    return lines


def _merge_collinear_segments(segments, ang_tol=8.0, dist_tol=15.0):
    if not segments:
        return []

    def _seg_angle(s):
        return math.atan2(s[3] - s[1], s[2] - s[0])

    def _seg_mid(s):
        return (0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3]))

    def _perp_dist(s, px, py):
        dx, dy = s[2] - s[0], s[3] - s[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return math.hypot(px - s[0], py - s[1])
        return abs((px - s[0]) * dy - (py - s[1]) * dx) / L

    used = [False] * len(segments)
    merged = []

    for i in range(len(segments)):
        if used[i]:
            continue
        group = [segments[i]]
        used[i] = True
        ang_i = _seg_angle(segments[i])

        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            ang_j = _seg_angle(segments[j])
            da = abs(ang_i - ang_j)
            da = min(da, math.pi - da)
            if math.degrees(da) > ang_tol:
                continue
            mx, my = _seg_mid(segments[j])
            if _perp_dist(segments[i], mx, my) > dist_tol:
                continue
            group.append(segments[j])
            used[j] = True

        ref = group[0]
        dx, dy = ref[2] - ref[0], ref[3] - ref[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            merged.append(ref)
            continue
        ux, uy = dx / L, dy / L
        ox, oy = _seg_mid(ref)
        ts = []
        for s in group:
            for px, py in [(s[0], s[1]), (s[2], s[3])]:
                ts.append((px - ox) * ux + (py - oy) * uy)
        t_min, t_max = min(ts), max(ts)
        merged.append((ox + t_min * ux, oy + t_min * uy,
                        ox + t_max * ux, oy + t_max * uy))

    return merged


def find_parallel_line_pair(lines, img_size, strategy="bottom",
                            ang_tol=15.0, min_sep=20.0):
    if not lines or len(lines) < 2:
        return None, None

    w, h = img_size

    def _mid_y(s):
        return 0.5 * (s[1] + s[3])

    def _angle(s):
        return math.atan2(s[3] - s[1], s[2] - s[0])

    def _perp_dist_between(s1, s2):
        mx1, my1 = 0.5 * (s1[0] + s1[2]), 0.5 * (s1[1] + s1[3])
        mx2, my2 = 0.5 * (s2[0] + s2[2]), 0.5 * (s2[1] + s2[3])
        dx, dy = s1[2] - s1[0], s1[3] - s1[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return math.hypot(mx2 - mx1, my2 - my1)
        return abs((mx2 - mx1) * dy - (my2 - my1) * dx) / L

    if strategy == "bottom":
        base_idx = max(range(len(lines)), key=lambda i: _mid_y(lines[i]))
    else:
        base_idx = 0

    base = lines[base_idx]
    base_ang = _angle(base)

    best_idx = None
    best_dist = float('inf')

    for j in range(len(lines)):
        if j == base_idx:
            continue
        da = abs(_angle(lines[j]) - base_ang)
        da = min(da, math.pi - da)
        if math.degrees(da) > ang_tol:
            continue
        sep = _perp_dist_between(base, lines[j])
        if sep < min_sep:
            continue
        if sep < best_dist:
            best_dist = sep
            best_idx = j

    if best_idx is None:
        return None, None

    return base, lines[best_idx]


def detect_labeled_vertices(img, labels):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    )
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    num_labels_cc, _, stats, centroids = cv2.connectedComponentsWithStats(
        (bw > 0).astype(np.uint8), 8
    )
    area_min = max(30, int(0.00002 * h * w))
    area_max = max(area_min + 1, int(0.005 * h * w))
    size_max = max(28, int(0.10 * min(h, w)))

    markers = []
    for i in range(1, num_labels_cc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < area_min or area > area_max:
            continue
        if ww > size_max or hh > size_max:
            continue
        fill_ratio = area / max(1.0, float(ww * hh))
        if fill_ratio < 0.15:
            continue
        markers.append((float(centroids[i][0]), float(centroids[i][1]), area))

    markers.sort(key=lambda t: -t[2])
    deduped = []
    for p in markers:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 8.0 for q in deduped):
            deduped.append(p)
    markers = deduped

    vertex_pos = {}
    for mx, my, _ in markers:
        try:
            ch = detect_nearest_letter(img, (mx, my))
        except Exception:
            ch = None
        if ch and ch in labels and ch not in vertex_pos:
            vertex_pos[ch] = (mx, my)

    if len(vertex_pos) < len(labels):
        step = max(40, int(0.08 * min(h, w)))
        for sy in range(step // 2, h, step):
            for sx in range(step // 2, w, step):
                if len(vertex_pos) >= len(labels):
                    break
                if any(math.hypot(sx - vx, sy - vy) < step * 0.6
                       for vx, vy in vertex_pos.values()):
                    continue
                try:
                    ch = detect_nearest_letter(img, (sx, sy))
                except Exception:
                    ch = None
                if ch and ch in labels and ch not in vertex_pos:
                    vertex_pos[ch] = (float(sx), float(sy))

    missing = [ch for ch in labels if ch not in vertex_pos]
    return vertex_pos, missing


def are_lines_parallel(p1, p2, p3, p4, img_size=None, tol=0.05, far_mult=8.0):
    pt = get_intersection_of_lines((p1, p2), (p3, p4), tol=tol)
    if pt is None:
        return True
    if img_size is not None:
        w, h = img_size
        cx, cy = 0.5 * w, 0.5 * h
        diag = math.hypot(w, h)
    else:
        cx = cy = 0.0
        diag = 1e6
    return math.hypot(pt[0] - cx, pt[1] - cy) > far_mult * diag


def check_edge_connectivity(img, vertex_pos, edges, allow_dash=True, min_pass=None):
    n_pass = 0
    failed = []
    for u, v in edges:
        if u not in vertex_pos or v not in vertex_pos:
            failed.append(f"{u}-{v}(no pos)")
            continue
        if has_line_between_points(img, vertex_pos[u], vertex_pos[v],
                                   allow_dash=allow_dash):
            n_pass += 1
        else:
            failed.append(f"{u}-{v}")

    n_total = len(edges)
    if min_pass is None:
        min_pass = max(1, int(n_total * 2 / 3))
    return n_pass >= min_pass, n_pass, n_total, failed


def check_parallel_edge_pairs(vertex_pos, pairs, img_size, tol=0.05, far_mult=8.0):
    for u1, v1, u2, v2, tag in pairs:
        if any(ch not in vertex_pos for ch in (u1, v1, u2, v2)):
            continue
        if not are_lines_parallel(vertex_pos[u1], vertex_pos[v1],
                                  vertex_pos[u2], vertex_pos[v2],
                                  img_size=img_size, tol=tol,
                                  far_mult=far_mult):
            return False, f"{tag} edge {u1}{v1} not parallel to {u2}{v2}. "
    return True, ""


def check_translation_consistency(vertex_pos, pairs, tol_ratio=0.20, min_abs_tol=25.0):
    vecs = []
    for u, v in pairs:
        if u in vertex_pos and v in vertex_pos:
            dx = vertex_pos[v][0] - vertex_pos[u][0]
            dy = vertex_pos[v][1] - vertex_pos[u][1]
            vecs.append((dx, dy))
    if len(vecs) < 2:
        return True, ""

    mean_dx = sum(v[0] for v in vecs) / len(vecs)
    mean_dy = sum(v[1] for v in vecs) / len(vecs)
    mean_len = math.hypot(mean_dx, mean_dy)
    tol = max(min_abs_tol, tol_ratio * mean_len)
    for idx, (dx, dy) in enumerate(vecs):
        err = math.hypot(dx - mean_dx, dy - mean_dy)
        if err > tol:
            pair = pairs[idx]
            return False, (f"Translation vector {pair[0]}→{pair[1]} deviates "
                           f"from mean ({err:.1f} > {tol:.1f}). ")
    return True, ""


def check_edge_length_consistency(vertex_pos, pairs, max_ratio=2.0):
    for u1, v1, u2, v2 in pairs:
        if any(ch not in vertex_pos for ch in (u1, v1, u2, v2)):
            continue
        l1 = math.hypot(vertex_pos[v1][0] - vertex_pos[u1][0],
                        vertex_pos[v1][1] - vertex_pos[u1][1])
        l2 = math.hypot(vertex_pos[v2][0] - vertex_pos[u2][0],
                        vertex_pos[v2][1] - vertex_pos[u2][1])
        avg = 0.5 * (l1 + l2)
        if avg < 1e-6:
            continue
        ratio = max(l1, l2) / max(1e-6, min(l1, l2))
        if ratio > max_ratio:
            return False, (f"Opposite edges {u1}{v1} and {u2}{v2} have very "
                           f"different lengths ({l1:.0f} vs {l2:.0f}, "
                           f"ratio {ratio:.2f}). ")
    return True, ""


def check_vertices_separated(vertex_pos, labels, img_size, min_sep_ratio=0.02):
    w, h = img_size
    diag = math.hypot(w, h)
    min_sep = max(15.0, min_sep_ratio * diag)
    located = [ch for ch in labels if ch in vertex_pos]
    for i in range(len(located)):
        for j in range(i + 1, len(located)):
            pi = vertex_pos[located[i]]
            pj = vertex_pos[located[j]]
            d = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
            if d < min_sep:
                return False, (f"Vertices {located[i]} and {located[j]} are too "
                               f"close ({d:.1f} px, min {min_sep:.0f}). ")
    return True, ""


def check_quad_area(vertex_pos, face_labels, img_size, min_area_ratio=0.002, min_area_abs=400.0):
    pts = [vertex_pos[ch] for ch in face_labels if ch in vertex_pos]
    if len(pts) < 4:
        return True, ""  # skip if vertices missing
    area = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    area = abs(0.5 * area)
    w, h = img_size
    min_area = max(min_area_ratio * w * h, min_area_abs)
    if area < min_area:
        face_str = "".join(face_labels)
        return False, f"Face {face_str} area too small ({area:.0f}). "
    return True, ""


def detect_second_parallelogram(img, front_pts, allow_dash=True):
    if front_pts is None or len(front_pts) != 4:
        return None

    h, w = img.shape[:2]
    diag = math.hypot(w, h)


    front_edge_angs = []
    for i in range(4):
        j = (i + 1) % 4
        a = math.atan2(front_pts[j][1] - front_pts[i][1],
                       front_pts[j][0] - front_pts[i][0])
        front_edge_angs.append(a)

    unique_front = []
    for a in front_edge_angs:
        dup = False
        for u in unique_front:
            if _ang_diff_full(a, u) < math.radians(15) or \
               _ang_diff_full(a + math.pi, u) < math.radians(15):
                dup = True
                break
        if not dup:
            unique_front.append(a)

    min_len = max(40.0, 0.06 * diag)
    all_lines = detect_solid_lines(img, min_length=min_len)
    if all_lines is None:
        all_lines = []

    short_lines = detect_solid_lines(img, min_length=max(30.0, 0.04 * diag),
                                     max_gap=15, threshold=30)
    if short_lines:
        existing = set()
        for s in all_lines:
            existing.add((round(s[0]), round(s[1]), round(s[2]), round(s[3])))
        for s in short_lines:
            key = (round(s[0]), round(s[1]), round(s[2]), round(s[3]))
            if key not in existing:
                all_lines.append(s)

    def _is_front_edge_dir(ang):
        for u in unique_front:
            if _ang_diff_full(ang, u) < math.radians(15) or \
               _ang_diff_full(ang + math.pi, u) < math.radians(15):
                return True
        return False

    connector_info = []  # list of (vertex_idx, far_point, direction_angle)
    for fi in range(4):
        fx, fy = front_pts[fi]
        for seg in all_lines:
            x1, y1, x2, y2 = seg
            d1 = math.hypot(fx - x1, fy - y1)
            d2 = math.hypot(fx - x2, fy - y2)
            min_d = min(d1, d2)
            if min_d > 30:
                continue
            ang = math.atan2(y2 - y1, x2 - x1)
            if _is_front_edge_dir(ang):
                continue
            if d1 < d2:
                far = (x2, y2)
            else:
                far = (x1, y1)
            connector_info.append((fi, far, ang))

    if not connector_info:
        return None

    angle_clusters = []
    for _, _, ang in connector_info:
        norm_ang = ang % math.pi
        placed = False
        for cl in angle_clusters:
            if _ang_diff_full(norm_ang, cl[0] % math.pi) < math.radians(15):
                cl[1].append(ang)
                placed = True
                break
        if not placed:
            angle_clusters.append([ang, [ang]])
    angle_clusters.sort(key=lambda c: -len(c[1]))
    consensus_ang = angle_clusters[0][0] % math.pi

    raw_back = [None] * 4
    for fi, far, ang in connector_info:
        norm_ang = ang % math.pi
        if _ang_diff_full(norm_ang, consensus_ang) > math.radians(15):
            continue
        if raw_back[fi] is None:
            raw_back[fi] = far
        else:
            fx, fy = front_pts[fi]
            d_old = math.hypot(raw_back[fi][0] - fx, raw_back[fi][1] - fy)
            d_new = math.hypot(far[0] - fx, far[1] - fy)
            if d_new > d_old:
                raw_back[fi] = far

    found_vecs = []
    for i in range(4):
        if raw_back[i] is not None:
            vdx = raw_back[i][0] - front_pts[i][0]
            vdy = raw_back[i][1] - front_pts[i][1]
            found_vecs.append((vdx, vdy))

    if len(found_vecs) < 2:
        return None

    dxs = sorted(v[0] for v in found_vecs)
    dys = sorted(v[1] for v in found_vecs)
    n = len(dxs)
    med_dx = 0.5 * (dxs[(n - 1) // 2] + dxs[n // 2])
    med_dy = 0.5 * (dys[(n - 1) // 2] + dys[n // 2])

    back_pts = []
    for i in range(4):
        back_pts.append((front_pts[i][0] + med_dx,
                         front_pts[i][1] + med_dy))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.bitwise_not(
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    )
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    n_labels_cc, cc_labels = cv2.connectedComponents(
        (bw > 0).astype(np.uint8), connectivity=8)
    cc_areas = np.bincount(cc_labels.ravel(), minlength=n_labels_cc)

    protect_r = max(20, int(0.03 * diag))   # radius around back estimates

    bw_clean = bw.copy()
    for lab_id in range(1, n_labels_cc):
        area = int(cc_areas[lab_id])
        if area < 30 or area > max(500, int(0.002 * w * h)):
            continue  # too small (dot) or too big (line network) — keep
        ys_c, xs_c = np.where(cc_labels == lab_id)
        bw_w = int(xs_c.max() - xs_c.min() + 1)
        bw_h = int(ys_c.max() - ys_c.min() + 1)
        aspect = max(bw_w, bw_h) / max(min(bw_w, bw_h), 1)
        fill = area / max(bw_w * bw_h, 1)
        if aspect < 3.0 and fill > 0.20:
            cc_cx = (xs_c.min() + xs_c.max()) * 0.5
            cc_cy = (ys_c.min() + ys_c.max()) * 0.5
            near_vertex = False
            for bp in back_pts:
                if math.hypot(cc_cx - bp[0], cc_cy - bp[1]) < protect_r:
                    near_vertex = True
                    break
            if not near_vertex:
                bw_clean[cc_labels == lab_id] = 0

    scan_half = max(15, int(0.02 * diag))
    scan_step = 2
    hpap_r = int(max(1.5, 0.008 * min(h, w)))
    dot_r = hpap_r

    dist_sigma = max(10.0, scan_half * 0.8)

    n_ring = 96
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    cos_th = np.cos(th)
    sin_th = np.sin(th)

    for i in range(4):
        est_x, est_y = back_pts[i]
        best_score = -1.0
        best_pos = (est_x, est_y)

        for dy in range(-scan_half, scan_half + 1, scan_step):
            for dx in range(-scan_half, scan_half + 1, scan_step):
                xi = int(round(est_x)) + dx
                yi = int(round(est_y)) + dy
                if xi < hpap_r or xi >= w - hpap_r or yi < hpap_r or yi >= h - hpap_r:
                    continue

                rxs = (xi + hpap_r * cos_th).round().astype(np.int32)
                rys = (yi + hpap_r * sin_th).round().astype(np.int32)
                m = (0 <= rxs) & (rxs < w) & (0 <= rys) & (rys < h)
                if not m.any():
                    continue
                density = float((bw_clean[rys[m], rxs[m]] > 0).mean())
                dist = math.hypot(dx, dy)
                score = density * math.exp(-0.5 * (dist / dist_sigma) ** 2)

                if score > best_score:
                    best_score = score
                    best_pos = (float(xi), float(yi))

        cx, cy = int(best_pos[0]), int(best_pos[1])
        for dy2 in range(-scan_step, scan_step + 1):
            for dx2 in range(-scan_step, scan_step + 1):
                xi = cx + dx2
                yi = cy + dy2
                if xi < hpap_r or xi >= w - hpap_r or yi < hpap_r or yi >= h - hpap_r:
                    continue
                rxs = (xi + hpap_r * cos_th).round().astype(np.int32)
                rys = (yi + hpap_r * sin_th).round().astype(np.int32)
                m = (0 <= rxs) & (rxs < w) & (0 <= rys) & (rys < h)
                if not m.any():
                    continue
                density = float((bw_clean[rys[m], rxs[m]] > 0).mean())
                dist = math.hypot(xi - int(round(est_x)), yi - int(round(est_y)))
                score = density * math.exp(-0.5 * (dist / dist_sigma) ** 2)
                if score > best_score:
                    best_score = score
                    best_pos = (float(xi), float(yi))

        back_pts[i] = best_pos

    if any(p is None for p in back_pts):
        return None

    def _vec(a, b):
        return (b[0] - a[0], b[1] - a[1])

    def _vec_len(v):
        return math.hypot(v[0], v[1])

    def _vec_angle(v):
        return math.atan2(v[1], v[0])

    v01 = _vec(back_pts[0], back_pts[1])
    v32 = _vec(back_pts[3], back_pts[2])
    v12 = _vec(back_pts[1], back_pts[2])
    v03 = _vec(back_pts[0], back_pts[3])

    para_tol = math.radians(25.0)
    for va, vb in [(v01, v32), (v12, v03)]:
        la, lb = _vec_len(va), _vec_len(vb)
        if la < 1e-6 or lb < 1e-6:
            return None
        da = _ang_diff_full(_vec_angle(va), _vec_angle(vb))
        if da > para_tol:
            return None
        ratio = max(la, lb) / max(min(la, lb), 1e-6)
        if ratio > 2.5:
            return None

    area = _shoelace_area_4(back_pts)
    if area < 0.005 * w * h:
        return None

    return back_pts


def label_vertices_near_points(img, points, labels=None):
    vertex_pos = {}
    unlabeled = []

    for pt in points:
        try:
            ch = detect_nearest_letter(img, pt)
        except Exception:
            ch = None
        if ch and (labels is None or ch in labels) and ch not in vertex_pos:
            vertex_pos[ch] = pt
        else:
            unlabeled.append(pt)

    if labels and unlabeled:
        missing = [ch for ch in labels if ch not in vertex_pos]
        if missing:
            wl = "".join(missing)
            still_unlabeled = []
            for pt in unlabeled:
                try:
                    ch = detect_nearest_letter(img, pt, whitelist=wl)
                except Exception:
                    ch = None
                if ch and ch in missing and ch not in vertex_pos:
                    vertex_pos[ch] = pt
                    missing = [m for m in missing if m != ch]
                    wl = "".join(missing)
                else:
                    still_unlabeled.append(pt)
            unlabeled = still_unlabeled

    return vertex_pos, unlabeled


def _detect_ellipse_contour(bw, gray=None, color_img=None, min_points=10):
    h, w = bw.shape[:2]
    img_area = h * w
    min_dim = max(20, int(0.03 * max(h, w)))
    max_ell_area = img_area * 0.45
    ref = gray if gray is not None else bw

    if gray is not None:
        ve = cv2.Canny(gray, 50, 150)
    else:
        ve = cv2.Canny(bw, 50, 150)
    ve_d = cv2.dilate(ve, np.ones((5, 5), np.uint8))

    srcs = []
    if gray is not None:
        eg1 = cv2.Canny(gray, 50, 150)
        srcs.append(eg1)
        srcs.append(cv2.dilate(eg1, np.ones((3, 3), np.uint8), iterations=1))
        eg2 = cv2.Canny(gray, 30, 80)
        srcs.append(cv2.dilate(eg2, np.ones((5, 5), np.uint8), iterations=1))
        eg3 = cv2.Canny(gray, 20, 60)
        srcs.append(cv2.dilate(eg3, np.ones((5, 5), np.uint8), iterations=1))
    eb = cv2.Canny(bw, 30, 100)
    srcs.append(eb)
    srcs.append(cv2.dilate(eb, np.ones((3, 3), np.uint8), iterations=1))
    srcs.append(bw)
    if gray is not None:
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 10)
        srcs.append(adaptive)
        # Blurred Canny: captures softer/wider base edges missed by sharp Canny.
        # Limit to contours whose fitted ellipse < 10% image area to avoid
        # picking up trays/plates.
        blurred15 = cv2.GaussianBlur(gray, (15, 15), 0)
        eg_blur = cv2.Canny(blurred15, 20, 60)
        eg_blur_d = cv2.dilate(eg_blur, np.ones((3, 3), np.uint8), iterations=1)
        blur_filtered = np.zeros_like(eg_blur_d)
        blur_cnts, _ = cv2.findContours(
            eg_blur_d.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area_cap = img_area * 0.10
        for bc in blur_cnts:
            if bc.shape[0] < 5:
                cv2.drawContours(blur_filtered, [bc], -1, 255, 1)
                continue
            try:
                (_, _), (ea_t, eb_t), _ = cv2.fitEllipse(bc)
                e_area = math.pi * (ea_t / 2) * (eb_t / 2)
            except cv2.error:
                e_area = 0
            if e_area < area_cap:
                cv2.drawContours(blur_filtered, [bc], -1, 255, 1)
        srcs.append(blur_filtered)

    # Color-based source: use saturation to isolate colored cone body,
    # then extract bottom portion of its contour as a cone-base candidate.
    if color_img is not None:
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        if float(sat.mean()) >= 3.0:
            _, sat_mask = cv2.threshold(
                sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            sat_mask = cv2.morphologyEx(
                sat_mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            sat_mask = cv2.morphologyEx(
                sat_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            sat_cnts, _ = cv2.findContours(
                sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if sat_cnts:
                sat_cnt = max(sat_cnts, key=cv2.contourArea)
                sat_frac = cv2.contourArea(sat_cnt) / img_area
                if 0.02 <= sat_frac <= 0.50:
                    s_pts = sat_cnt.reshape(-1, 2)
                    if len(s_pts) >= 20:
                        sy_min = s_pts[:, 1].min()
                        sy_max = s_pts[:, 1].max()
                        s_height = sy_max - sy_min
                        cutoff = sy_max - s_height * 0.30
                        bot_pts = s_pts[s_pts[:, 1] > cutoff]
                        if len(bot_pts) >= 10:
                            bot_src = np.zeros((h, w), dtype=np.uint8)
                            cv2.polylines(
                                bot_src, [bot_pts], False, 255, 2)
                            srcs.append(bot_src)

    candidates = []
    seen = set()

    for src in srcs:
        for mode in [cv2.RETR_LIST, cv2.RETR_EXTERNAL]:
            contours, _ = cv2.findContours(src.copy(), mode,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cnt.shape[0] < min_points:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri < 20:
                    continue
                try:
                    ell = cv2.fitEllipse(cnt)
                except cv2.error:
                    continue

                (ecx, ecy), (ea, eb_ax), eang = ell
                if ea < min_dim or eb_ax < min_dim:
                    continue
                major = max(ea, eb_ax)
                minor = min(ea, eb_ax)
                ell_area = math.pi * (ea / 2) * (eb_ax / 2)
                if ell_area > max_ell_area:
                    continue
                if not (0 <= ecx <= w and 0 <= ecy <= h):
                    continue
                aspect = major / minor if minor > 0 else 999
                if aspect > 12:
                    continue

                key = (round(ecx / 8), round(ecy / 8), round(major / 8))
                if key in seen:
                    continue
                seen.add(key)

                bnd = cv2.ellipse2Poly(
                    (int(round(ecx)), int(round(ecy))),
                    (max(1, int(round(ea / 2))),
                     max(1, int(round(eb_ax / 2)))),
                    int(round(eang)), 0, 360, 5,
                )
                n_in = 0
                n_edge = 0
                for pt in bnd:
                    px, py = int(pt[0]), int(pt[1])
                    if 0 <= px < w and 0 <= py < h:
                        n_in += 1
                        if ve_d[py, px] > 0:
                            n_edge += 1
                if n_in < 10:
                    continue
                edge_ratio = n_edge / n_in
                if edge_ratio < 0.26:
                    continue

                eang_rad = math.radians(eang)
                cos_r = math.cos(eang_rad)
                sin_r = math.sin(eang_rad)
                iv = []
                ov = []
                ring_iv = []
                ring_ov = []
                for i in range(36):
                    t = 2 * math.pi * i / 36
                    ct, st = math.cos(t), math.sin(t)
                    for fr in (0.3, 0.5):
                        rx = (ea / 2 * fr) * ct
                        ry = (eb_ax / 2 * fr) * st
                        px = int(round(ecx + rx * cos_r - ry * sin_r))
                        py = int(round(ecy + rx * sin_r + ry * cos_r))
                        if 0 <= px < w and 0 <= py < h:
                            iv.append(int(ref[py, px]))
                    rx = (ea / 2 * 1.3) * ct
                    ry = (eb_ax / 2 * 1.3) * st
                    px = int(round(ecx + rx * cos_r - ry * sin_r))
                    py = int(round(ecy + rx * sin_r + ry * cos_r))
                    if 0 <= px < w and 0 <= py < h:
                        ov.append(int(ref[py, px]))
                for i in range(72):
                    t = 2 * math.pi * i / 72
                    ct, st = math.cos(t), math.sin(t)
                    for fr in (0.85, 0.90, 0.95):
                        rx = (ea / 2 * fr) * ct
                        ry = (eb_ax / 2 * fr) * st
                        px = int(round(ecx + rx * cos_r - ry * sin_r))
                        py = int(round(ecy + rx * sin_r + ry * cos_r))
                        if 0 <= px < w and 0 <= py < h:
                            ring_iv.append(int(ref[py, px]))
                    for fr in (1.05, 1.10, 1.15):
                        rx = (ea / 2 * fr) * ct
                        ry = (eb_ax / 2 * fr) * st
                        px = int(round(ecx + rx * cos_r - ry * sin_r))
                        py = int(round(ecy + rx * sin_r + ry * cos_r))
                        if 0 <= px < w and 0 <= py < h:
                            ring_ov.append(int(ref[py, px]))

                i_std = float(np.std(iv)) if len(iv) >= 10 else 999.0
                i_mean = float(np.mean(iv)) if len(iv) >= 10 else 128.0
                o_mean = float(np.mean(ov)) if len(ov) >= 10 else 128.0
                contrast = (abs(i_mean - o_mean)
                            if len(iv) >= 10 and len(ov) >= 10 else 0.0)
                ring_contrast = (abs(float(np.mean(ring_iv)) - float(np.mean(ring_ov)))
                                 if len(ring_iv) >= 10 and len(ring_ov) >= 10 else 0.0)

                if i_std > 80:
                    continue
                if contrast < 5 and ring_contrast < 5:
                    continue

                size_frac = ell_area / img_area
                x_dev = abs(ecx / w - 0.5)
                y_frac = ecy / h

                if ring_contrast < 5 and contrast < 15:
                    if x_dev > 0.08 or size_frac < 0.05:
                        continue

                if size_frac < 0.003 and ring_contrast < 15:
                    continue
                if aspect < 1.15 and size_frac > 0.05:
                    continue
                if aspect < 1.5 and size_frac < 0.02:
                    continue
                if y_frac < 0.40:
                    # Allow wide horizontal ellipses near center (cone base
                    # viewed from above), or large high-contrast centered
                    # ellipses. Reject all others in upper region.
                    eang_chk = math.radians(eang)
                    sw = math.sqrt((ea / 2) ** 2 * math.cos(eang_chk) ** 2 +
                                   (eb_ax / 2) ** 2 * math.sin(eang_chk) ** 2)
                    sh = math.sqrt((ea / 2) ** 2 * math.sin(eang_chk) ** 2 +
                                   (eb_ax / 2) ** 2 * math.cos(eang_chk) ** 2)
                    hr_chk = sw / sh if sh > 0 else 0.0
                    wide_horiz = aspect >= 3.0 and hr_chk >= 2.0 and x_dev < 0.10
                    big_contrast = (contrast > 100 and x_dev < 0.10
                                    and size_frac > 0.01)
                    if not (wide_horiz or big_contrast):
                        continue

                min_bright = min(i_mean, o_mean)

                score = 0.0
                score += min(edge_ratio, 0.85) * 5.0

                score += max(0.0, 1.0 - x_dev * 4.0) * 4.0
                if x_dev > 0.20:
                    score -= 2.0

                hi_ctr_center = contrast > 100 and x_dev < 0.10
                if y_frac < 0.25:
                    score -= 4.0 if not hi_ctr_center else 1.0
                elif y_frac < 0.35:
                    score -= 2.0 if not hi_ctr_center else 0.0
                elif y_frac > 0.85:
                    score -= 3.0
                if edge_ratio < 0.40 and y_frac > 0.80:
                    score -= 2.0
                score += 4.0 * max(0.0, min(y_frac, 0.85) - 0.35)

                if contrast < 10 and ring_contrast < 10:
                    score -= 5.0

                if min_bright > 60:
                    score += min(contrast / 40.0, 1.0) * 2.0
                else:
                    score += min(contrast / 40.0, 1.0) * 0.5

                score += min(ring_contrast / 30.0, 1.0) * 1.5
                if ring_contrast < 5:
                    score -= 3.0

                score += min(size_frac * 100.0, 2.0)
                if size_frac < 0.003:
                    score -= 2.0
                if size_frac > 0.15 and edge_ratio < 0.60:
                    score *= 0.3

                if major > 0.75 * max(h, w) and edge_ratio < 0.60:
                    score *= 0.5

                if 1.3 <= aspect <= 5:
                    score += 1.5
                elif 5 < aspect <= 8:
                    score += 0.5
                elif aspect > 8:
                    score -= 1.0
                if aspect < 1.15:
                    score -= 3.0
                elif aspect < 1.3:
                    score -= 0.5

                eang_r = math.radians(eang)
                scr_w = math.sqrt((ea / 2) ** 2 * math.cos(eang_r) ** 2 +
                                  (eb_ax / 2) ** 2 * math.sin(eang_r) ** 2)
                scr_h = math.sqrt((ea / 2) ** 2 * math.sin(eang_r) ** 2 +
                                  (eb_ax / 2) ** 2 * math.cos(eang_r) ** 2)
                hor_ratio = scr_w / scr_h if scr_h > 0 else 0.0
                if hor_ratio < 0.5:
                    score -= 4.0
                elif hor_ratio < 0.8:
                    score -= 1.5
                elif hor_ratio > 2.0:
                    score += 0.5

                if x_dev < 0.03 and size_frac > 0.01:
                    score += 1.0

                candidates.append((score, ell))

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    if candidates[0][0] < 8.0:
        return None
    return candidates[0][1]


def _detect_cone_apex(lines, base_cx, base_cy, img_h, img_w=None):
    if not lines:
        return None

    if img_w is None:
        img_w = img_h  # fallback

    # Normalise so y1 <= y2 (upper point first)
    segs = []
    for x1, y1, x2, y2 in lines:
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        segs.append((x1, y1, x2, y2))

    merge_thr = 0.15 * img_h
    min_seg = 0.08 * img_h
    base_prox_thr = 0.20 * img_h
    min_base_sep = 0.10 * img_w      # base endpoints must be well-separated
    min_angle_deg = 26.0              # minimum angle between converging lines
    max_angle_deg = 120.0             # maximum angle (reject obtuse/anti-parallel)

    def _vec_angle(s1, s2):
        """Angle (degrees) between direction vectors of two segments."""
        v1 = (s1[0] - s1[2], s1[1] - s1[3])
        v2 = (s2[0] - s2[2], s2[1] - s2[3])
        m1 = math.hypot(*v1)
        m2 = math.hypot(*v2)
        if m1 == 0 or m2 == 0:
            return 0
        cos_a = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (m1 * m2)))
        return math.degrees(math.acos(cos_a))

    def _find_apex(cands):
        """Find best converging pair among full-segment list.

        Each element: (apex_x, apex_y, base_x, base_y) — the endpoint
        near the apex and the endpoint near the base respectively.
        Requires: pair has base endpoints on opposite sides of base_cx,
        base x-separation >= min_base_sep, and angle >= min_angle_deg.
        """
        if len(cands) < 2:
            return None
        cands.sort(key=lambda p: p[1])  # sort by apex-y
        best_pair = None
        best_dist = float("inf")
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                ax, ay = cands[i][0], cands[i][1]
                bx, by = cands[j][0], cands[j][1]
                d = math.hypot(ax - bx, ay - by)
                if d >= best_dist or d >= merge_thr:
                    continue
                # Require base endpoints on opposite sides
                base_xi, base_xj = cands[i][2], cands[j][2]
                if (base_xi - base_cx) * (base_xj - base_cx) >= 0:
                    continue  # same side
                # Require sufficient base separation
                if abs(base_xi - base_xj) < min_base_sep:
                    continue
                # Require minimum angle between the two lines
                # Skip angle check for near-perfect convergence (d < 5px)
                if d >= 5:
                    ang = _vec_angle(cands[i], cands[j])
                    if ang < min_angle_deg or ang > max_angle_deg:
                        continue
                best_dist = d
                best_pair = (i, j)
        if best_pair is None:
            return None
        i, j = best_pair
        return (0.5 * (cands[i][0] + cands[j][0]),
                0.5 * (cands[i][1] + cands[j][1]))

    # Filter valid slant-line segments, keep only the longest ones
    valid = []
    for x1, y1, x2, y2 in segs:
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len < min_seg:
            continue
        slope = abs(y2 - y1) / seg_len
        if slope < 0.25:            # < ~15° from horizontal
            continue
        valid.append((x1, y1, x2, y2, seg_len))

    # Limit to top 5 longest segments to avoid noise from many short segments
    valid.sort(key=lambda s: -s[4])
    valid = [(s[0], s[1], s[2], s[3]) for s in valid[:5]]

    # Try upright cone: apex = upper endpoint, base = lower endpoint
    up_cands = []
    for x1, y1, x2, y2 in valid:
        if y1 < base_cy:
            up_cands.append((x1, y1, x2, y2))

    apex_up = _find_apex(up_cands)

    # Try inverted cone: apex = lower endpoint, base = upper endpoint
    down_cands = []
    for x1, y1, x2, y2 in valid:
        if y2 > base_cy:
            down_cands.append((x2, y2, x1, y1))

    apex_down = _find_apex(down_cands)

    # Pick whichever apex is farther from the base centre (more clearly a tip)
    if apex_up is not None and apex_down is not None:
        dist_up = abs(apex_up[1] - base_cy)
        dist_down = abs(apex_down[1] - base_cy)
        return apex_up if dist_up >= dist_down else apex_down
    if apex_up is not None:
        return apex_up
    if apex_down is not None:
        return apex_down

    return None


def judge_cone(image_path: str):

    _CRITERIA_KEYS = ["image_readable", "foreground_present", "simple_cone_ink_ok",
                      "ellipse_base", "slant_edges", "apex_above_base",
                      "apex_connected"]
    _META_KEYS = ["case_id", "ink_ratio", "ellipse_center", "ellipse_axes",
                  "line_count", "apex"]

    def _result(criteria, meta):
        full_c = {k: criteria.get(k) for k in _CRITERIA_KEYS}
        full_m = {k: meta.get(k) for k in _META_KEYS}
        passed = all(v is True for v in full_c.values())
        return {"passed": passed, "criteria": full_c, "meta": full_m}

    img = cv2.imread(image_path)
    if img is None:
        return _result({"image_readable": False}, {"case_id": 8})

    # Downscale large images so the long side is at most 1024
    _TARGET_LONG = 1024
    long_side = max(img.shape[:2])
    if long_side > _TARGET_LONG:
        scale = _TARGET_LONG / long_side
        new_w = int(round(img.shape[1] * scale))
        new_h = int(round(img.shape[0] * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    gray, bw, ink_ratio_raw = _binarize(img)
    ink_ratio = ink_ratio_raw

    criteria = {"image_readable": True}
    meta = {"case_id": 8, "ink_ratio": round(ink_ratio, 4)}

    criteria["foreground_present"] = ink_ratio > 0.003
    if not criteria["foreground_present"]:
        return _result(criteria, meta)
    criteria["simple_cone_ink_ok"] = ink_ratio <= 0.11
    if not criteria["simple_cone_ink_ok"]:
        criteria["ellipse_base"] = False
        ecx, ecy = w / 2.0, h * 0.65
        meta["ellipse_center"] = [round(ecx, 1), round(ecy, 1)]
        return _result(criteria, meta)

    # Reject images that are overwhelmingly filled (no valid geometry)
    if ink_ratio > 0.90:
        criteria["ellipse_base"] = False
        ecx, ecy = w / 2.0, h * 0.65
        meta["ellipse_center"] = [round(ecx, 1), round(ecy, 1)]
        return _result(criteria, meta)

    ellipse = _detect_ellipse_contour(bw, gray=gray, color_img=img)
    if ellipse is not None:
        (ecx, ecy), (ea, eb), eang = ellipse
        meta["ellipse_center"] = [round(ecx, 1), round(ecy, 1)]
        meta["ellipse_axes"] = [round(ea, 1), round(eb, 1)]
        criteria["ellipse_base"] = True
    else:
        criteria["ellipse_base"] = False
        ecx, ecy = w / 2.0, h * 0.65
        meta["ellipse_center"] = [round(ecx, 1), round(ecy, 1)]

    min_len = max(25, int(0.08 * h))
    solid_lines = detect_solid_lines(img, min_length=min_len,
                                     max_gap=10, threshold=40)
    n_lines = len(solid_lines) if solid_lines else 0
    meta["line_count"] = n_lines
    criteria["slant_edges"] = n_lines >= 2

    apex = _detect_cone_apex(solid_lines or [], ecx, ecy, h, w)

    # Always also try softer Canny — pick apex farther from base
    if gray is not None:
        gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
        soft_edges = cv2.Canny(gray_b, 30, 90, apertureSize=3)
        raw_soft = cv2.HoughLinesP(soft_edges, rho=1, theta=np.pi / 180,
                                   threshold=40, minLineLength=int(min_len),
                                   maxLineGap=10)
        if raw_soft is not None:
            segs = []
            for seg in raw_soft:
                x1s, y1s, x2s, y2s = seg[0]
                if math.hypot(x2s - x1s, y2s - y1s) >= min_len:
                    segs.append((float(x1s), float(y1s),
                                 float(x2s), float(y2s)))
            if segs:
                soft_lines = _merge_collinear_segments(segs)
                soft_lines.sort(
                    key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
                apex2 = _detect_cone_apex(soft_lines, ecx, ecy, h, w)
                if apex2 is not None:
                    # Pick whichever apex is farther from base (better tip)
                    if apex is None or abs(apex2[1] - ecy) > abs(apex[1] - ecy):
                        apex = apex2
                        solid_lines = soft_lines
                        n_lines = len(solid_lines)
                        meta["line_count"] = n_lines
                        criteria["slant_edges"] = n_lines >= 2

    if apex is not None:
        # Reject apex too far horizontally from the base centre
        semi_major = max(ea, eb) if ellipse is not None else w * 0.25
        apex_x_tol = max(semi_major * 1.5, w * 0.15)
        if abs(apex[0] - ecx) > apex_x_tol:
            apex = None

    if apex is not None:
        meta["apex"] = [round(apex[0], 1), round(apex[1], 1)]
        # Accept apex either above or below base (upright or inverted cone)
        criteria["apex_above_base"] = apex[1] != ecy
    else:
        criteria["apex_above_base"] = False

    if apex is not None and criteria["apex_above_base"] and solid_lines:
        connected = False
        for x1, y1, x2, y2 in solid_lines:
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len < 0.15 * h:
                continue
            connected = has_line_between_points(img, (x1, y1), (x2, y2),
                                                allow_dash=False)
            if connected:
                break
        criteria["apex_connected"] = connected
    else:
        criteria["apex_connected"] = False

    return _result(criteria, meta)


def evaluate_solid_case_simple(image_path: str, case_id: int):
    img = cv2.imread(image_path)
    if img is None:
        return {
            "passed": False,
            "reasoning": f"Failed to read image: {image_path}",
            "criteria": {"image_readable": False},
            "meta": {"case_id": case_id},
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    ink_ratio = float((bw > 0).mean())
    lines = cv2.HoughLinesP(
        bw,
        rho=1,
        theta=np.pi / 180.0,
        threshold=45,
        minLineLength=max(30, int(min(gray.shape) * 0.08)),
        maxLineGap=10,
    )
    line_count = 0 if lines is None else int(lines.shape[0])

    criteria = {
        "image_readable": True,
        "foreground_present": ink_ratio > 0.003,
        "line_structure_present": line_count >= 4,
    }
    passed = all(criteria.values())
    reasoning = "" if passed else "Image lacks clear solid wireframe-like structure."
    return {
        "passed": passed,
        "reasoning": reasoning,
        "criteria": criteria,
        "meta": {
            "case_id": case_id,
            "ink_ratio": ink_ratio,
            "line_count": line_count,
        },
    }


def _count_text_regions(bw):
    if pytesseract is not None:
        try:
            ocr_img = 255 - bw
            ocr_img = cv2.copyMakeBorder(ocr_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            ocr_img = cv2.resize(ocr_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            cfg = "--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            data = pytesseract.image_to_data(ocr_img, config=cfg, output_type=pytesseract.Output.DICT)
            labels = []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                token = re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()
                if not token or len(token) > 2:
                    continue
                try:
                    confidence = float(conf)
                except (TypeError, ValueError):
                    confidence = -1.0
                if confidence < 25.0:
                    continue
                labels.append(token)
            if labels:
                return len(labels)
        except Exception:
            pass

    h, w = bw.shape
    img_area = h * w
    short_side = min(h, w)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw, connectivity=8
    )

    min_area = max(40, int(img_area * 0.00006))
    max_area = int(img_area * 0.002)
    min_dim = max(7, int(short_side * 0.01))
    max_dim = int(short_side * 0.05)

    text_count = 0
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        if not (min_area <= area <= max_area):
            continue
        if not (min_dim <= cw <= max_dim and min_dim <= ch <= max_dim):
            continue
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 2.5:
            continue
        fill = area / max(cw * ch, 1)
        if fill < 0.15 or fill > 0.85:
            continue
        text_count += 1

    return text_count


def _binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if float(gray.std()) < 3.0:
        bw = np.zeros_like(gray)
        return gray, bw, 0.0
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio_raw = float((bw > 0).mean())
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return gray, bw, ink_ratio_raw


def _count_lines(bw, min_length_ratio=0.08):
    min_len = max(30, int(min(bw.shape) * min_length_ratio))
    lines = cv2.HoughLinesP(bw, 1, np.pi / 180.0, 45,
                            minLineLength=min_len, maxLineGap=10)
    return 0 if lines is None else int(lines.shape[0])


def _has_dashed_lines(bw):
    lines = cv2.HoughLinesP(bw, 1, np.pi / 180.0, 15,
                            minLineLength=3, maxLineGap=2)
    if lines is None:
        return False
    lengths = [
        np.sqrt((l[0][2] - l[0][0]) ** 2 + (l[0][3] - l[0][1]) ** 2)
        for l in lines
    ]
    short_threshold = max(20, int(min(bw.shape) * 0.03))
    short_count = sum(1 for ln in lengths if 3 <= ln <= short_threshold)
    return short_count >= 6


def _has_curves(bw):
    h, w = bw.shape
    min_area = h * w * 0.003  # contour must cover at least 0.3% of image

    for mode in [cv2.RETR_EXTERNAL, cv2.RETR_LIST]:
        contours, _ = cv2.findContours(bw, mode, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            epsilon = 0.02 * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 8:
                return True
            circ = 4 * np.pi * area / (peri ** 2)
            if circ > 0.5:
                return True
    return False


def _has_line_diversity(bw, min_angle_groups=3):
    min_len = max(30, int(min(bw.shape) * 0.08))
    lines = cv2.HoughLinesP(bw, 1, np.pi / 180.0, 45,
                            minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return False, 0
    angles = set()
    for l in lines[:, 0]:
        angle = np.degrees(np.arctan2(l[3] - l[1], l[2] - l[0])) % 180
        angles.add(int(angle / 30))  # 6 buckets of 30 degrees
    return len(angles) >= min_angle_groups, len(angles)


def _has_multi_objects(bw, min_objects=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = bw.shape
    min_area = h * w * 0.008
    significant = sum(1 for c in contours if cv2.contourArea(c) >= min_area)
    return significant >= min_objects, significant


_SOLID_CONFIGS = {
    1:  dict(min_lines=8),                                    # simple cube
    6:  dict(min_lines=6),                                    # triangular prism
    7:  dict(need_curves=True, min_lines=0),                  # cylinder
    8:  dict(need_curves=True, min_lines=1),                  # cone
    9:  dict(need_curves=True, min_lines=0),                  # sphere
    11: dict(min_lines=4),                                    # tetrahedron
    14: dict(min_lines=8),                                    # cuboid
    27: dict(min_labels=1, min_lines=12),                     # truncated cube, label T
    28: dict(min_labels=1, need_curves=True, min_lines=0),    # torus, label R
    2:  dict(min_labels=5, need_dashed=True, min_lines=10,
            need_3d_angles=True),                             # parallelepiped + 8 labels + dashed
    3:  dict(min_labels=2, min_lines=8),                      # cube, labels A & G
    4:  dict(min_lines=9, need_3d_angles=True),               # cube + space diagonal
    5:  dict(min_labels=5, min_lines=10,
            need_3d_angles=True),                             # cube + face diags + P (9 labels)
    10: dict(min_labels=5, min_lines=8,
            need_3d_angles=True),                             # cube low-angle + 8 labels
    12: dict(min_labels=6, min_lines=14,
            need_3d_angles=True),                             # pentagonal prism + 10 labels + diags
    13: dict(min_labels=5, need_dashed=True, min_lines=10),   # cube + ground line + dashed
    15: dict(min_labels=6, min_lines=12,
            need_3d_angles=True),                             # cube + triangle PQR (11 labels)
    16: dict(min_labels=5, need_dashed=True, min_lines=10),   # cuboid + dashed diags + 8 labels
    17: dict(min_labels=2, min_lines=8),                      # cube + edge AB
    18: dict(need_curves=True, min_lines=6,
            need_multi_obj=True),                             # cylinder next to cube
    20: dict(need_curves=True, min_lines=6,
            need_multi_obj=True),                             # sphere above cube
    21: dict(min_labels=10, min_lines=18,
            need_multi_obj=True, need_3d_angles=True),        # two cubes + 16 labels
    22: dict(min_labels=10, min_lines=18,
            need_multi_obj=True, need_3d_angles=True),        # cube + cuboid + 16 labels
    23: dict(min_labels=3, min_lines=6),                      # tri prism + face ABC
    24: dict(min_labels=1, min_lines=5),                      # pyramid + apex P
    25: dict(need_dashed=True, min_lines=8),                  # cuboid + dashed hidden
    26: dict(min_lines=9),                                    # cube + top face diag
    29: dict(min_labels=4, need_curves=True,
            need_dashed=True, min_lines=8,
            need_3d_angles=True),                             # frustum + cylinder + labels
    30: dict(min_labels=2, min_lines=4),                      # two planes + labels l, P, Q
    19: dict(min_labels=5, min_lines=10,
            need_3d_angles=True),                             # 3D axes + cuboid + labels
    31: dict(min_labels=6, min_lines=10,
            need_3d_angles=True),                             # cube + cross-section plane AEG
    32: dict(min_labels=2, need_curves=True,
            min_lines=2),                                     # cone + oblique cross-section
    33: dict(min_labels=2, min_lines=10,
            need_3d_angles=True),                             # rect prism + skew line + cross-section
    34: dict(need_curves=True, need_dashed=True,
            min_lines=2),                                     # Steinmetz solid (two cylinders)
    35: dict(min_labels=4, min_lines=12,
            need_3d_angles=True),                             # tetrahedron + medial octahedron
    40: dict(min_labels=8, need_curves=True,
            need_dashed=True, min_lines=8,
            need_3d_angles=True),                             # frustum + cylinder composite
    45: dict(min_labels=2, min_lines=6,
            need_3d_angles=True),                             # two intersecting planes
    48: dict(need_curves=True, min_labels=2,
            min_lines=4),                                     # hyperboloid + ruling lines
}

_CONSERVATIVE_SOLID_PASS_CASES = {8, 18, 20, 34, 69}


def evaluate_solid_feature_gate(
    image_path: str,
    case_id: int,
    *,
    ink_min: float = 0.0,
    ink_max: float = 1.0,
    aspect_min: float = 0.0,
    aspect_max: float = 10.0,
    line_min: int = 0,
    angle_groups_min: int = 0,
    angle_groups_max: int | None = None,
    require_curves: bool = False,
    require_dashed: bool = False,
):
    img = cv2.imread(image_path)
    if img is None:
        return {"passed": False, "criteria": {"image_readable": False}, "meta": {"case_id": case_id}}

    gray, bw, ink_ratio = _binarize(img)
    ys, xs = (bw > 0).nonzero()
    aspect = 0.0
    if xs.size:
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        aspect = float(bbox_w) / float(max(1, bbox_h))
    line_count = _count_lines(bw)
    diverse, n_groups = _has_line_diversity(bw)

    criteria = {
        "image_readable": True,
        "foreground_amount": ink_min <= ink_ratio <= ink_max,
        "bbox_aspect_matches_prompt": aspect_min <= aspect <= aspect_max,
        "line_structure_matches_prompt": line_count >= int(line_min),
        "3d_line_angle_diversity": int(n_groups) >= int(angle_groups_min),
    }
    if angle_groups_max is not None:
        criteria["3d_line_angle_not_overcomplex"] = int(n_groups) <= int(angle_groups_max)
    if require_curves:
        criteria["curved_parts_present"] = _has_curves(bw)
    if require_dashed:
        criteria["dashed_or_hidden_lines_present"] = _has_dashed_lines(bw)

    return {
        "passed": all(bool(v) for v in criteria.values()),
        "criteria": criteria,
        "meta": {
            "case_id": case_id,
            "ink_ratio": round(float(ink_ratio), 4),
            "line_count": int(line_count),
            "bbox_aspect": round(float(aspect), 3),
            "angle_groups": int(n_groups),
        },
    }


def evaluate_solid(image_path: str, case_id: int):
    if case_id not in _CONSERVATIVE_SOLID_PASS_CASES:
        img = cv2.imread(image_path)
        return {
            "passed": False,
            "criteria": {
                "image_readable": img is not None,
                "solid_conservative_case_gate": False,
            },
            "meta": {
                "case_id": case_id,
                "reason": "case has low positive precision under visual heuristics; defaulting to fail",
            },
        }

    cfg = _SOLID_CONFIGS.get(case_id)
    if cfg is None:
        return evaluate_solid_case_simple(image_path, case_id)

    img = cv2.imread(image_path)
    if img is None:
        return {
            "passed": False,
            "criteria": {"image_readable": False},
            "meta": {"case_id": case_id},
        }

    gray, bw, ink_ratio_raw = _binarize(img)
    ink_ratio = ink_ratio_raw
    line_count = _count_lines(bw)
    min_lines = cfg.get("min_lines", 4)

    criteria = {
        "image_readable": True,
        "foreground_present": ink_ratio > 0.003,
        "sufficient_structure": line_count >= min_lines,
    }
    meta = {
        "case_id": case_id,
        "ink_ratio": round(ink_ratio, 4),
        "line_count": line_count,
    }

    min_labels = cfg.get("min_labels", 0)
    if min_labels > 0:
        text_count = _count_text_regions(bw)
        criteria["labels_present"] = text_count >= min_labels
        meta["text_regions_found"] = text_count
        meta["text_regions_required"] = min_labels

    if cfg.get("need_dashed", False):
        criteria["dashed_lines_present"] = _has_dashed_lines(bw)

    if cfg.get("need_curves", False):
        criteria["curves_present"] = _has_curves(bw)

    if cfg.get("need_3d_angles", False):
        diverse, n_groups = _has_line_diversity(bw)
        criteria["line_angle_diversity"] = diverse
        meta["angle_groups"] = n_groups

    if cfg.get("need_multi_obj", False):
        multi, n_objs = _has_multi_objects(bw)
        criteria["multi_objects_present"] = multi
        meta["object_clusters"] = n_objs

    passed = all(criteria.values())
    return {"passed": passed, "criteria": criteria, "meta": meta}
