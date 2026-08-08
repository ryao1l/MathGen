"""
Shared OCR-based label detection utilities for set evaluation scripts.

Replaces the old template-matching approach (cv2.HERSHEY_SIMPLEX) with
EasyOCR, which correctly handles italic serif fonts from gpt-image-1.5.

Usage:
    from ocr_label_utils import find_labels_ocr, verify_circle_labels
"""
import threading
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None

# ──────────────── lazy singleton EasyOCR reader ────────────────

_reader = None
_reader_lock = threading.Lock()
_label_cache = {}
_label_cache_lock = threading.Lock()


def get_reader():
    """Return a cached EasyOCR Reader (created once, reused across calls)."""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


# ──────────────── core OCR detection ────────────────

LabelHit = Tuple[str, int, int, float]  # (letter, cx, cy, confidence)

_ALLOWLIST = 'ABCDUabcdu'


def _allow_easyocr_fallback() -> bool:
    return os.environ.get("MATHGEN_SET_EASYOCR_FALLBACK", "0").lower() in {"1", "true", "yes"}


def _allow_ocr() -> bool:
    return os.environ.get("MATHGEN_SET_ENABLE_OCR", "0").lower() in {"1", "true", "yes"}


def _image_cache_key(image_path_or_img):
    if not isinstance(image_path_or_img, str):
        return None
    try:
        import os
        stat = os.stat(image_path_or_img)
    except OSError:
        return None
    return (image_path_or_img, int(stat.st_mtime_ns), int(stat.st_size))


def _letters_cache_key(target_letters: Optional[List[str]]):
    if target_letters is None:
        return None
    return tuple(sorted({letter.upper() for letter in target_letters}))


def _circles_cache_key(circles: list):
    return tuple((round(float(cx), 1), round(float(cy), 1), round(float(r), 1)) for cx, cy, r in circles)


def _get_cached_labels(key):
    if key is None:
        return None
    with _label_cache_lock:
        cached = _label_cache.get(key)
        return list(cached) if cached is not None else None


def _set_cached_labels(key, hits):
    if key is None:
        return
    with _label_cache_lock:
        if len(_label_cache) > 512:
            _label_cache.clear()
        _label_cache[key] = tuple(hits)


def _ocr_single_letters(reader, img_bgr, offset_x: int = 0, offset_y: int = 0,
                         min_conf: float = 0.30,
                         text_threshold: float = 0.5,
                         low_text: float = 0.4) -> List[LabelHit]:
    """Run EasyOCR on an image and return single-letter detections."""
    hits = []
    try:
        results = reader.readtext(img_bgr, detail=1, paragraph=False,
                                   allowlist=_ALLOWLIST,
                                   text_threshold=text_threshold,
                                   low_text=low_text)
    except Exception:
        return hits

    for (bbox, text, conf) in results:
        text = text.strip()
        if len(text) != 1 or not text.isalpha():
            continue
        letter = text.upper()
        if conf < min_conf:
            continue
        cx = int(sum(p[0] for p in bbox) / 4) + offset_x
        cy = int(sum(p[1] for p in bbox) / 4) + offset_y
        hits.append((letter, cx, cy, float(conf)))
    return hits


def _ocr_single_letters_tesseract(img_bgr, offset_x: int = 0, offset_y: int = 0,
                                  min_conf: float = 0.30) -> List[LabelHit]:
    """Fast OCR path for the single A/B/C/U labels used in set diagrams."""
    if pytesseract is None:
        return []

    try:
        if img_bgr.ndim == 2:
            gray = img_bgr
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        variants = [gray, cv2.bitwise_not(gray)]
    except Exception:
        return []

    hits: List[LabelHit] = []
    cfg = "--psm 11 -c tessedit_char_whitelist=ABCDUabcdu"
    try:
        timeout = float(os.environ.get("MATHGEN_SET_TESSERACT_TIMEOUT", "0.8"))
    except ValueError:
        timeout = 0.8
    for variant in variants:
        try:
            scaled = cv2.copyMakeBorder(variant, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            scaled = cv2.resize(scaled, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            data = pytesseract.image_to_data(
                scaled,
                config=cfg,
                output_type=pytesseract.Output.DICT,
                timeout=timeout,
            )
        except Exception:
            continue

        for text, conf, left, top, width, height in zip(
            data.get("text", []),
            data.get("conf", []),
            data.get("left", []),
            data.get("top", []),
            data.get("width", []),
            data.get("height", []),
        ):
            token = str(text).strip()
            if len(token) != 1 or not token.isalpha():
                continue
            letter = token.upper()
            if letter not in set(_ALLOWLIST.upper()):
                continue
            try:
                confidence = float(conf) / 100.0
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < min_conf:
                continue
            cx = int(round(((float(left) + float(width) / 2.0) / 2.0) - 12)) + offset_x
            cy = int(round(((float(top) + float(height) / 2.0) / 2.0) - 12)) + offset_y
            hits.append((letter, cx, cy, float(confidence)))
    return hits


def find_labels_ocr(
    image_path_or_img,
    *,
    target_letters: Optional[List[str]] = None,
    min_conf: float = 0.30,
) -> List[LabelHit]:
    """Detect single-letter labels in an image using EasyOCR.

    Runs OCR on:
      1. The full original image
      2. The full inverted image (for white-on-dark text)

    Args:
        image_path_or_img: Path to image file, or BGR ndarray.
        target_letters: If given, only return these letters.
        min_conf: Minimum confidence threshold.

    Returns:
        List of (letter_upper, center_x, center_y, confidence).
    """
    if not _allow_ocr():
        return []

    cache_key = ("full", _image_cache_key(image_path_or_img), _letters_cache_key(target_letters), round(float(min_conf), 3))
    cached = _get_cached_labels(cache_key)
    if cached is not None:
        return cached

    if isinstance(image_path_or_img, str):
        img = cv2.imread(image_path_or_img)
        if img is None:
            return []
    else:
        img = image_path_or_img

    targets_upper = None
    if target_letters is not None:
        targets_upper = {l.upper() for l in target_letters}

    all_hits: List[LabelHit] = []

    all_hits.extend(_ocr_single_letters_tesseract(img, min_conf=min_conf))

    if all_hits:
        if targets_upper is not None:
            all_hits = [h for h in all_hits if h[0] in targets_upper]
        hits = _dedupe_hits(all_hits, merge_dist=60.0)
        _set_cached_labels(cache_key, hits)
        return hits

    if not _allow_easyocr_fallback():
        hits = _dedupe_hits(all_hits, merge_dist=60.0)
        _set_cached_labels(cache_key, hits)
        return hits

    reader = get_reader()

    # Pass 1: original
    all_hits.extend(_ocr_single_letters(reader, img, min_conf=min_conf))

    # Pass 2: inverted
    inv = cv2.bitwise_not(img)
    all_hits.extend(_ocr_single_letters(reader, inv, min_conf=min_conf))

    # Filter by target letters
    if targets_upper is not None:
        all_hits = [h for h in all_hits if h[0] in targets_upper]

    hits = _dedupe_hits(all_hits, merge_dist=60.0)
    _set_cached_labels(cache_key, hits)
    return hits


def find_labels_in_circles(
    image_path_or_img,
    circles: list,
    *,
    target_letters: Optional[List[str]] = None,
    min_conf: float = 0.30,
    crop_ratio: float = 0.55,
) -> List[LabelHit]:
    """Detect labels using both full-image AND per-circle-crop OCR.

    This is MORE RELIABLE than find_labels_ocr alone, because cropping
    around each circle center gives EasyOCR much better context for
    small or low-contrast letters (e.g., white C on red background).

    Args:
        image_path_or_img: Path or BGR ndarray.
        circles: List of (cx, cy, r) tuples.
        target_letters: Only return these letters.
        min_conf: Minimum confidence.
        crop_ratio: Fraction of radius to use for crop half-size.

    Returns:
        List of (letter, cx, cy, confidence).
    """
    if not _allow_ocr():
        return []

    cache_key = (
        "circles",
        _image_cache_key(image_path_or_img),
        _circles_cache_key(circles),
        _letters_cache_key(target_letters),
        round(float(min_conf), 3),
        round(float(crop_ratio), 3),
    )
    cached = _get_cached_labels(cache_key)
    if cached is not None:
        return cached

    if isinstance(image_path_or_img, str):
        img = cv2.imread(image_path_or_img)
        if img is None:
            return []
    else:
        img = image_path_or_img

    h, w = img.shape[:2]
    targets_upper = None
    if target_letters is not None:
        targets_upper = {l.upper() for l in target_letters}

    all_hits: List[LabelHit] = []

    # Fast path: tesseract is much cheaper than initializing EasyOCR in each
    # evaluator subprocess. It works well for the single-letter labels used here.
    all_hits.extend(_ocr_single_letters_tesseract(img, min_conf=min_conf))
    crop_circles = sorted(circles, key=lambda c: float(c[2]), reverse=True)[:4]
    for (ccx, ccy, cr) in crop_circles:
        ccx, ccy, cr = int(ccx), int(ccy), int(cr)
        half = int(cr * crop_ratio)
        x1, y1 = max(0, ccx - half), max(0, ccy - half)
        x2, y2 = min(w, ccx + half), min(h, ccy + half)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crop = img[y1:y2, x1:x2]
        all_hits.extend(_ocr_single_letters_tesseract(crop, x1, y1, min_conf))

    if targets_upper is not None:
        all_hits = [h for h in all_hits if h[0] in targets_upper]
    hits = _dedupe_hits(all_hits, merge_dist=60.0)
    expected = len(targets_upper) if targets_upper is not None else 1
    if len({h[0] for h in hits}) >= expected:
        _set_cached_labels(cache_key, hits)
        return hits

    if not _allow_easyocr_fallback():
        _set_cached_labels(cache_key, hits)
        return hits

    reader = get_reader()
    all_hits = []

    # Pass 1: full image (original + inverted)
    all_hits.extend(_ocr_single_letters(reader, img, min_conf=min_conf))
    all_hits.extend(_ocr_single_letters(reader, cv2.bitwise_not(img), min_conf=min_conf))

    # Pass 2: per-circle crops (original + inverted + contrast-enhanced)
    for (ccx, ccy, cr) in crop_circles:
        ccx, ccy, cr = int(ccx), int(ccy), int(cr)
        half = int(cr * crop_ratio)
        x1, y1 = max(0, ccx - half), max(0, ccy - half)
        x2, y2 = min(w, ccx + half), min(h, ccy + half)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue

        crop = img[y1:y2, x1:x2]

        # crop original
        all_hits.extend(_ocr_single_letters(reader, crop, x1, y1, min_conf,
                                             text_threshold=0.3, low_text=0.3))

        # crop inverted
        all_hits.extend(_ocr_single_letters(reader, cv2.bitwise_not(crop), x1, y1, min_conf,
                                             text_threshold=0.3, low_text=0.3))

        # crop contrast-enhanced
        gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = cv2.cvtColor(clahe.apply(gray_c), cv2.COLOR_GRAY2BGR)
        all_hits.extend(_ocr_single_letters(reader, enhanced, x1, y1, min_conf,
                                             text_threshold=0.3, low_text=0.3))

    # Filter by target
    if targets_upper is not None:
        all_hits = [h for h in all_hits if h[0] in targets_upper]

    hits = _dedupe_hits(all_hits, merge_dist=60.0)
    _set_cached_labels(cache_key, hits)
    return hits


def _dedupe_hits(hits: List[LabelHit], merge_dist: float = 60.0) -> List[LabelHit]:
    """Merge nearby detections of the same letter, keeping highest confidence."""
    hits_sorted = sorted(hits, key=lambda h: h[3], reverse=True)
    kept: List[LabelHit] = []
    for letter, cx, cy, conf in hits_sorted:
        if any(kl == letter and np.hypot(cx - kx, cy - ky) < merge_dist
               for kl, kx, ky, _ in kept):
            continue
        kept.append((letter, cx, cy, conf))
    return kept


# ──────────────── circle-label verification ────────────────

Circle = Tuple[float, float, float]  # (cx, cy, r)


def point_in_circle(px: float, py: float, circle: Circle) -> bool:
    """Check if a point is inside a circle."""
    cx, cy, r = circle
    return np.hypot(px - cx, py - cy) <= r


def verify_circle_labels(
    image_path: str,
    circles: list,
    expected_labels: List[str],
    *,
    min_conf: float = 0.30,
) -> Tuple[bool, Dict[str, str]]:
    """Verify that each expected label appears exactly once inside a distinct circle.

    Uses the find_labels_in_circles strategy (full image + per-circle crops).

    Args:
        image_path: Path to the image.
        circles: List of (cx, cy, r) circles.
        expected_labels: Labels that must each appear once inside a unique circle.
        min_conf: Minimum OCR confidence.

    Returns:
        (all_ok, details_dict).
    """
    hits = find_labels_in_circles(image_path, circles,
                                   target_letters=expected_labels,
                                   min_conf=min_conf)

    details: Dict[str, str] = {}
    all_ok = True
    used_circles = set()

    for letter in expected_labels:
        letter_up = letter.upper()
        letter_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == letter_up]
        details[f"label_{letter}_count"] = str(len(letter_hits))

        if len(letter_hits) == 0:
            details[f"label_{letter}_ok"] = "False (not detected)"
            all_ok = False
            continue

        # Take highest confidence
        letter_hits.sort(key=lambda h: h[3], reverse=True)
        _, lx, ly, conf = letter_hits[0]
        details[f"label_{letter}_conf"] = f"{conf:.3f}"

        # Check inside exactly one circle
        inside = [i for i, c in enumerate(circles) if point_in_circle(lx, ly, c)]

        if len(inside) == 0:
            details[f"label_{letter}_ok"] = "False (not inside any circle)"
            all_ok = False
        elif inside[0] in used_circles:
            details[f"label_{letter}_ok"] = "False (duplicate circle)"
            all_ok = False
        else:
            used_circles.add(inside[0])
            details[f"label_{letter}_ok"] = "True"

    details["labels_all_ok"] = str(all_ok)
    return all_ok, details


def verify_circle_and_rect_labels(
    image_path: str,
    circles: list,
    circle_labels: List[str],
    *,
    rect_label: str = "U",
    min_conf: float = 0.30,
) -> Tuple[bool, Dict[str, str]]:
    """Verify circle labels + a rectangle label (U) that should NOT be inside any circle.

    Args:
        image_path: Path to the image.
        circles: List of (cx, cy, r) circles.
        circle_labels: Letters inside circles (e.g. ["A", "B", "C"]).
        rect_label: Letter for the rectangle (default "U").
        min_conf: Minimum OCR confidence.

    Returns:
        (all_ok, details_dict).
    """
    all_labels = list(circle_labels) + [rect_label]
    hits = find_labels_in_circles(image_path, circles,
                                   target_letters=all_labels,
                                   min_conf=min_conf)

    details: Dict[str, str] = {}
    all_ok = True
    used_circles = set()

    # Circle labels
    for letter in circle_labels:
        letter_up = letter.upper()
        letter_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == letter_up]
        details[f"label_{letter}_count"] = str(len(letter_hits))

        if len(letter_hits) == 0:
            details[f"label_{letter}_ok"] = "False (not detected)"
            all_ok = False
            continue

        letter_hits.sort(key=lambda h: h[3], reverse=True)
        _, lx, ly, conf = letter_hits[0]
        details[f"label_{letter}_conf"] = f"{conf:.3f}"

        inside = [i for i, c in enumerate(circles) if point_in_circle(lx, ly, c)]
        if len(inside) == 0:
            details[f"label_{letter}_ok"] = "False (not inside any circle)"
            all_ok = False
        elif inside[0] in used_circles:
            details[f"label_{letter}_ok"] = "False (duplicate circle)"
            all_ok = False
        else:
            used_circles.add(inside[0])
            details[f"label_{letter}_ok"] = "True"

    # Rect label
    rect_hits = [(l, cx, cy, c) for l, cx, cy, c in hits if l == rect_label.upper()]
    details[f"label_{rect_label}_count"] = str(len(rect_hits))

    if len(rect_hits) == 0:
        details[f"label_{rect_label}_ok"] = "False (not detected)"
        all_ok = False
    else:
        rect_hits.sort(key=lambda h: h[3], reverse=True)
        _, ux, uy, conf = rect_hits[0]
        details[f"label_{rect_label}_conf"] = f"{conf:.3f}"
        inside_any = any(point_in_circle(ux, uy, c) for c in circles)
        if inside_any:
            details[f"label_{rect_label}_ok"] = "False (inside a circle)"
            all_ok = False
        else:
            details[f"label_{rect_label}_ok"] = "True"

    details["labels_all_ok"] = str(all_ok)
    return all_ok, details
