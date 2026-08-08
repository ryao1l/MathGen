"""Shared utilities for MathGen deterministic evaluator scripts.

These helpers standardize evaluator I/O without changing task-level decisions.
Prompt-specific scripts should keep their mathematical logic local or in a
topic-level common module.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


def parse_image_id(image_path: str) -> Optional[int]:
    """Return the numeric id at the end of an image filename."""
    match = re.search(r"(\d+)$", Path(image_path).stem)
    return int(match.group(1)) if match else None


def to_jsonable(value: Any) -> Any:
    """Convert common numpy/scipy scalar and array values to JSON-safe types."""
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def make_report(passed: bool, criteria: Optional[Mapping[str, Any]] = None, meta: Optional[Mapping[str, Any]] = None) -> dict:
    """Create the canonical evaluator report dictionary."""
    clean_criteria = {str(k): bool(v) for k, v in (criteria or {}).items()}
    return {
        "passed": bool(passed),
        "criteria": clean_criteria,
        "meta": to_jsonable(dict(meta or {})),
    }


def normalize_report(report: Any) -> dict:
    """Normalize legacy bool/list/dict evaluator outputs to the report schema."""
    report = to_jsonable(report)
    if isinstance(report, dict):
        passed = bool(report.get("passed", False))
        criteria = report.get("criteria", {})
        meta = report.get("meta", {})
        extra = {k: v for k, v in report.items() if k not in {"passed", "criteria", "meta"}}
        if extra:
            meta = {**dict(meta or {}), **extra}
        return make_report(passed, criteria, meta)
    if isinstance(report, (list, tuple)) and report:
        return make_report(bool(report[0]), meta={"raw_report": report})
    return make_report(bool(report), meta={"raw_report": report})


def print_report(report: Any, *, indent: int = 2) -> None:
    """Print a normalized evaluator report as JSON."""
    print(json.dumps(normalize_report(report), ensure_ascii=False, indent=indent))


def load_sibling_module(module_name: str, file_name: str, current_file: str):
    """Load a helper module from the current script's directory."""
    module_path = os.path.join(os.path.dirname(os.path.abspath(current_file)), file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_simple_cli(evaluate_fn: Callable[[str], Any], description: str = "Run a MathGen evaluator.") -> None:
    """Run a standard --image CLI for simple one-argument evaluators."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--image", type=str, required=True, help="Path to the generated image.")
    args = parser.parse_args()
    print_report(evaluate_fn(args.image))

