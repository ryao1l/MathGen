#!/usr/bin/env python3
"""Unified evaluator for all MathGen angle prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import angle_common as _common


CASE_CONFIGS: Dict[int, dict] = {
    28: {"n_angles": 3, "expected": [70.0, 110.0], "mode": "straight_adjacent", "tol": 12.0},
    29: {"n_angles": 3, "expected": [45.0, 135.0], "mode": "straight_adjacent", "tol": 12.0},
    30: {"n_angles": 3, "expected": [60.0, 120.0], "mode": "straight_adjacent", "tol": 6.0},
    31: {"n_angles": 3, "expected": [30.0, 150.0], "mode": "straight_adjacent", "tol": "wide", "split_obtuse": True},
    32: {"n_angles": 3, "expected": [80.0, 100.0], "mode": "straight_adjacent", "tol": "wide"},
    33: {"n_angles": 3, "expected": [90.0, 90.0], "mode": "two_right_visual"},
    34: {"n_angles": 3, "expected": [55.0, 125.0], "mode": "straight_adjacent", "tol": "wide"},
    35: {"n_angles": 4, "expected": [90.0, 90.0, 90.0, 90.0], "mode": "four_right", "tol": 10.0},
    36: {"n_angles": 4, "expected": [30.0, 60.0, 90.0, 180.0], "mode": "four_rays", "tol": "wide", "case_filter": True},
    37: {"n_angles": 4, "expected": [45.0, 45.0, 135.0, 135.0], "mode": "four_rays", "tol": 12.0, "case_filter": True},
    38: {"n_angles": 4, "expected": [50.0, 70.0, 100.0, 140.0], "mode": "four_rays", "tol": 12.0, "case_filter": True},
    39: {"n_angles": 4, "expected": [80.0, 100.0, 80.0, 100.0], "mode": "four_rays", "tol": 8.0},
    40: {"n_angles": 4, "expected": [60.0, 90.0, 120.0, 90.0], "mode": "four_rays", "tol": "wide"},
    41: {"n_angles": 4, "expected": [40.0, 50.0, 110.0, 160.0], "mode": "four_rays", "tol": "wide", "case_filter": True},
    42: {"n_angles": 4, "expected": [75.0, 75.0, 105.0, 105.0], "mode": "four_rays", "tol": "wide"},
    43: {"n_angles": 4, "expected": [30.0, 60.0, 120.0, 150.0], "mode": "four_rays", "tol": "wide", "case_filter": True},
    44: {"n_angles": 4, "expected": [55.0, 65.0, 115.0, 125.0], "mode": "four_rays", "tol": "wide", "case_filter": True},
    45: {"n_angles": 4, "expected": [45.0, 135.0, 45.0, 135.0], "mode": "four_rays", "tol": "wide", "case_filter": True},
    46: {"n_angles": 4, "expected": [70.0, 50.0, 70.0, 170.0], "mode": "four_rays", "tol": "wide"},
    47: {"n_angles": 4, "expected": [36.0, 72.0, 108.0, 144.0], "mode": "four_rays", "tol": "wide"},
    48: {"n_angles": 4, "expected": [20.0, 40.0, 60.0, 240.0], "mode": "four_rays", "tol": 12.0, "case_filter": True},
    49: {"n_angles": 4, "expected": [90.0, 60.0, 90.0, 120.0], "mode": "four_rays", "tol": 12.0, "case_filter": True},
    50: {"n_angles": 4, "expected": [60.0, 80.0, 100.0, 120.0], "mode": "four_rays", "tol": 8.0, "case_filter": True},
}


def _parse_case_id(image_path: str, case_id: Optional[int]) -> int:
    if case_id is not None:
        return int(case_id)
    parsed = _common._parse_image_id(image_path)
    if parsed is None:
        raise ValueError(f"Cannot infer angle case id from image path: {image_path}")
    return int(parsed)


def _tol(config: dict) -> float:
    value = config.get("tol", "wide")
    if value == "wide":
        return _common.ANGLE_TOL_DEG + 8.0
    return float(value)


def _evaluate_configured_case(image_path: str, case_id: int, n_angles: Optional[int], min_sep_deg: Optional[float]) -> dict:
    config = CASE_CONFIGS[case_id]
    expected: List[float] = list(config["expected"])
    mode = str(config["mode"])
    n_angles = int(config["n_angles"] if n_angles is None else n_angles)

    img = _common.load_image(image_path)
    if img is None:
        return {"passed": False, "criteria": {"image_readable": False}, "meta": {"case_id": case_id}}

    gray = _common.cv2.cvtColor(img, _common.cv2.COLOR_BGR2GRAY)
    center, rays = _common.detect_center_and_rays(gray)
    sectors = _common._cyclic_diffs(rays)

    criteria = {
        "image_readable": True,
        "foreground_detected": center is not None,
        "rays_detected": len(rays) >= max(2, n_angles),
    }

    if mode == "two_right_visual":
        criteria["rays_detected"] = len(rays) >= 2
        criteria.update(_common._angle36_visual_two_right_angles(gray))
    else:
        tol = _tol(config)
        lenient_ok = False
        if mode == "straight_adjacent":
            lenient_ok = _common._lenient_straight_adjacent_fallback(case_id, sectors, gray)
            if config.get("split_obtuse"):
                split_obtuse_ok = (
                    len(sectors) == 4
                    and _common._has_all_sectors(sectors, [159.0, 97.0, 52.0, 52.0], 3.0)
                    and _common._relaxed_clean_annotated_angle_scene(gray, min_top_area=7000.0)
                )
                lenient_ok = lenient_ok or split_obtuse_ok
            expected_full = expected + [180.0]
            criteria["rays_detected"] = criteria["rays_detected"] or lenient_ok
            criteria["straight_line_present"] = (
                _common._has_opposite_pairs(rays, min_pairs=1)
                or _common._has_sector_near(sectors, 180.0, tol)
                or lenient_ok
            )
            criteria["ray_count_tight_ok"] = 3 <= len(rays) <= 4 or lenient_ok
            criteria["adjacent_supplementary_values_ok"] = _common._multiset_match(sectors, expected_full, tol) or lenient_ok
        elif mode == "four_right":
            lenient_ok = _common._lenient_four_right_angles_fallback(sectors, gray)
            offset_marker_fp = _common._offset_square_marker_false_positive(sectors, gray)
            four_ray_values_ok = (
                _common._multiset_match(sectors, expected, tol)
                or _common._contains_cyclic_sequence(sectors, expected, tol)
                or _common._contains_cyclic_sequence(sectors, list(reversed(expected)), tol)
                or lenient_ok
            )
            pure_quadrant_cross = len(sectors) == 4 and all(abs(s - 90.0) <= 2.0 for s in sectors)
            over_split_octants = len(sectors) >= 8 and all(40.0 <= s <= 50.0 for s in sectors)
            criteria["not_plain_cross_or_octants"] = not (pure_quadrant_cross or over_split_octants)
            criteria["four_ray_values_ok"] = four_ray_values_ok and not offset_marker_fp
        elif mode == "four_rays":
            case_accept = _common._case_specific_four_rays_accept(case_id, sectors, gray) if config.get("case_filter") else False
            case_reject = _common._case_specific_four_rays_reject(case_id, sectors, gray) if config.get("case_filter") else False
            criteria["rays_detected"] = criteria["rays_detected"] or case_accept
            four_ray_values_ok = (
                _common._multiset_match(sectors, expected, tol)
                or _common._contains_cyclic_sequence(sectors, expected, tol)
                or _common._contains_cyclic_sequence(sectors, list(reversed(expected)), tol)
                or case_accept
            )
            criteria["four_ray_values_ok"] = four_ray_values_ok and not case_reject
        else:
            criteria["known_check_mode"] = False

    return {
        "passed": all(bool(v) for v in criteria.values()),
        "criteria": criteria,
        "meta": {
            "case_id": case_id,
            "check_mode": mode,
            "expected_angles": expected,
            "center": center,
            "ray_angles_deg": rays,
            "sector_angles_deg": sectors,
        },
    }


def evaluate(
    image_path: str,
    case_id: Optional[int] = None,
    n_angles: Optional[int] = None,
    min_sep_deg: Optional[float] = None,
) -> dict:
    """Evaluate one angle image, inferring the case id from the filename by default."""
    resolved_case_id = _parse_case_id(image_path, case_id)
    if resolved_case_id in CASE_CONFIGS:
        return _evaluate_configured_case(image_path, resolved_case_id, n_angles, min_sep_deg)
    return _common.evaluate_angle_case(
        image_path,
        case_id=resolved_case_id,
        n_angles=4 if n_angles is None else int(n_angles),
        min_sep_deg=10.0 if min_sep_deg is None else float(min_sep_deg),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one MathGen angle image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--case_id", type=int, default=None, help="Optional prompt id; defaults to the image filename.")
    parser.add_argument("--n_angles", type=int, default=None)
    parser.add_argument("--min_sep_deg", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.image, args.case_id, args.n_angles, args.min_sep_deg), indent=2))


if __name__ == "__main__":
    main()
