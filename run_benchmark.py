#!/usr/bin/env python3
"""Run the MathGen generation-and-evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import evaluate
import generate
from utils import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MathGen: generate images, then evaluate them with deterministic local judges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    project_root = Path(__file__).resolve().parent

    parser.add_argument("--prompt-dir", "--prompt_dir", dest="prompt_dir", type=Path, default=project_root / "data" / "prompt_data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=project_root / "data" / "generated_img", help="Generation output root.")
    parser.add_argument("--images-dir", "--images_dir", dest="images_dir", type=Path, default=None, help="Evaluation image root; defaults to --output-dir.")
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "mapping.csv")
    parser.add_argument("--eval-output-dir", "--eval_output_dir", dest="eval_output_dir", type=Path, default=project_root / "results" / "current" / "evaluation")

    parser.add_argument("--backend", choices=["dry-run", "openai", "replicate"], default="dry-run")
    parser.add_argument("--model", default="gpt-image-1")
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--topic", default=None, help="Single-topic alias for --topics.")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-prompt-constraints", "--no_prompt_constraints", dest="no_prompt_constraints", action="store_true")
    parser.add_argument("--generation-manifest", "--generation_manifest", dest="generation_manifest", type=Path, default=None)

    parser.add_argument("--replicate-model", "--replicate_model", dest="replicate_model", default=None)
    parser.add_argument("--replicate-prompt-key", "--replicate_prompt_key", dest="replicate_prompt_key", default="prompt")
    parser.add_argument("--replicate-extra-input", "--replicate_extra_input", dest="replicate_extra_input", default=None)
    parser.add_argument("--replicate-min-interval", "--replicate_min_interval", dest="replicate_min_interval", type=float, default=0.0)
    parser.add_argument("--replicate-max-retries", "--replicate_max_retries", dest="replicate_max_retries", type=int, default=3)

    parser.add_argument("--python-bin", "--python_bin", dest="python_bin", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--preflight-timeout", "--preflight_timeout", dest="preflight_timeout", type=float, default=10.0)
    parser.add_argument("--no-preflight", "--no_preflight", dest="no_preflight", action="store_true")
    parser.add_argument("--preflight-topics", "--preflight_topics", dest="preflight_topics", nargs="*", default=["counting"])

    parser.add_argument("--skip-generate", "--skip_generate", dest="skip_generate", action="store_true")
    parser.add_argument("--skip-evaluate", "--skip_evaluate", dest="skip_evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.images_dir is None:
        args.images_dir = args.output_dir
    args.output_dir = Path(args.output_dir)
    args.images_dir = Path(args.images_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        generate.run_generation(args)

    if not args.skip_evaluate:
        eval_args = argparse.Namespace(
            prompt_dir=args.prompt_dir,
            images_dir=args.images_dir,
            manifest=args.manifest,
            output_dir=args.eval_output_dir,
            python_bin=args.python_bin,
            topics=args.topics or ([args.topic] if args.topic else run_evaluation.DEFAULT_TOPICS),
            models=[args.model],
            workers=args.workers,
            timeout=args.timeout,
            preflight_timeout=args.preflight_timeout,
            no_preflight=args.no_preflight,
            preflight_topics=args.preflight_topics,
            resume=args.resume,
            limit=args.limit,
        )
        evaluate.run_evaluation(eval_args)


if __name__ == "__main__":
    main()
