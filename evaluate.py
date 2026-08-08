#!/usr/bin/env python3
"""Evaluate generated MathGen images with deterministic local judges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import run_evaluation as evaluation_runner


def _split_optional_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def run_evaluation(args) -> None:
    """Run the evaluator scripts using a parsed argument namespace."""
    project_root = Path(__file__).resolve().parent
    prompt_dir = Path(getattr(args, "prompt_dir", project_root / "data" / "prompt_data"))
    generated_img_dir = Path(getattr(args, "images_dir", project_root / "data" / "generated_img"))
    manifest = Path(getattr(args, "manifest", project_root / "data" / "mapping.csv"))
    output_dir = Path(getattr(args, "output_dir", project_root / "results" / "current" / "evaluation"))

    runner_args = argparse.Namespace(
        prompt_dir=prompt_dir,
        generated_img_dir=generated_img_dir,
        manifest=manifest,
        output_dir=output_dir,
        python_bin=getattr(args, "python_bin", sys.executable),
        topics=_split_optional_list(getattr(args, "topics", None))
        or _split_optional_list(getattr(args, "topic", None))
        or evaluation_runner.DEFAULT_TOPICS,
        models=_split_optional_list(getattr(args, "models", None))
        or _split_optional_list(getattr(args, "model", None)),
        workers=int(getattr(args, "workers", 4)),
        timeout=float(getattr(args, "timeout", 20.0)),
        preflight_timeout=float(getattr(args, "preflight_timeout", 10.0)),
        no_preflight=bool(getattr(args, "no_preflight", False)),
        preflight_topics=_split_optional_list(getattr(args, "preflight_topics", None)) or ["counting"],
        resume=bool(getattr(args, "resume", False)),
        limit=int(getattr(args, "limit", 0)),
    )
    evaluation_runner.run_evaluation(runner_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated MathGen images with deterministic local judges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--prompt-dir", "--prompt_dir", dest="prompt_dir", type=Path, default=project_root / "data" / "prompt_data")
    parser.add_argument("--images-dir", "--images_dir", dest="images_dir", type=Path, default=project_root / "data" / "generated_img")
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "mapping.csv")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=project_root / "results" / "current" / "evaluation")
    parser.add_argument("--python-bin", "--python_bin", dest="python_bin", default=sys.executable)
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--topic", default=None, help="Single-topic alias for --topics.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--model", default=None, help="Single-model alias for --models.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--preflight-timeout", "--preflight_timeout", dest="preflight_timeout", type=float, default=10.0)
    parser.add_argument("--no-preflight", "--no_preflight", dest="no_preflight", action="store_true")
    parser.add_argument("--preflight-topics", "--preflight_topics", dest="preflight_topics", nargs="*", default=["counting"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
