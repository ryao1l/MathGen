#!/usr/bin/env python3
"""Generate images for MathGen prompts.

The default layout is compatible with multi-model evaluation:

``data/generated_img/<model>/<topic>/<id>.png``
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

from utils.math_global_config import GLOBAL_CONSTRAINTS, get_topic_constraints
from utils.io_utils import load_jsonl, write_jsonl


def _rule_name(item: dict) -> str:
    return item.get("rule") or item.get("criterion") or "Rule"


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-").lower()


def build_enhanced_prompt(user_prompt: str, topic: str, include_constraints: bool = True) -> str:
    """Attach shared MathGen drawing constraints to a task prompt."""
    if not include_constraints:
        return user_prompt

    parts = [
        "You are an expert mathematical visualization engine.",
        "Generate a precise, high-quality academic image for this task:",
        "",
        user_prompt,
        "",
        "General requirements:",
    ]
    for item in GLOBAL_CONSTRAINTS:
        parts.append(f"- {_rule_name(item)}: {item['description']}")

    topic_rules = get_topic_constraints(topic)
    if topic_rules:
        parts.append("")
        parts.append(f"{topic} requirements:")
        for item in topic_rules:
            parts.append(f"- {_rule_name(item)}: {item['description']}")

    parts.append("")
    parts.append("Use a clean composition and avoid extra objects or labels unless requested.")
    return "\n".join(parts)


def iter_prompt_rows(prompt_dir: Path, topics: Optional[list[str]]) -> list[dict]:
    rows: list[dict] = []
    files = [prompt_dir / f"{topic}.jsonl" for topic in topics] if topics else sorted(prompt_dir.glob("*.jsonl"))
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        rows.extend(load_jsonl(path))
    return rows


def generate_openai_image(prompt: str, save_path: Path, model: str, size: str) -> bool:
    """Generate one image through the OpenAI Images API."""
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from exc

    client = OpenAI()
    response = client.images.generate(model=model, prompt=prompt, size=size)
    image_b64 = response.data[0].b64_json
    save_path.write_bytes(base64.b64decode(image_b64))
    return True


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def generate_replicate_image(
    prompt: str,
    save_path: Path,
    replicate_model: str,
    prompt_key: str,
    extra_input: Optional[str],
    max_retries: int,
    min_interval: float,
) -> bool:
    """Generate one image through Replicate."""
    try:
        import replicate
    except Exception as exc:
        raise RuntimeError("replicate package is not installed. Run: pip install replicate") from exc

    model_input = {prompt_key: prompt}
    if extra_input:
        parsed = json.loads(extra_input)
        if not isinstance(parsed, dict):
            raise ValueError("--replicate-extra-input must be a JSON object.")
        model_input.update(parsed)

    for attempt in range(max_retries + 1):
        try:
            output = replicate.run(replicate_model, input=model_input)
            data = None
            if hasattr(output, "read"):
                data = output.read()
            elif hasattr(output, "url"):
                data = _download_bytes(str(output.url))
            elif isinstance(output, list) and output:
                first = output[0]
                if hasattr(first, "read"):
                    data = first.read()
                elif hasattr(first, "url"):
                    data = _download_bytes(str(first.url))
                elif isinstance(first, str):
                    data = _download_bytes(first)
            elif isinstance(output, str):
                data = _download_bytes(output)

            if not data:
                raise RuntimeError(f"Unsupported Replicate output type: {type(output)!r}")
            save_path.write_bytes(data)
            time.sleep(max(0.0, min_interval))
            return True
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(60.0, max(1.0, min_interval) * (attempt + 1)))
    return False


def run_generation(args) -> None:
    """Generate images or write a dry-run manifest from a parsed namespace."""
    project_root = Path(__file__).resolve().parent
    prompt_dir = Path(getattr(args, "prompt_dir", project_root / "data" / "prompt_data"))
    output_dir = Path(getattr(args, "output_dir", project_root / "data" / "generated_img"))
    backend = getattr(args, "backend", "dry-run")
    topics = getattr(args, "topics", None) or ([getattr(args, "topic")] if getattr(args, "topic", None) else None)
    model = getattr(args, "model", "gpt-image-1")
    model_dir_name = sanitize_name(model)
    size = getattr(args, "size", "1024x1024")
    include_constraints = not bool(getattr(args, "no_prompt_constraints", False))
    limit = int(getattr(args, "limit", 0))
    resume = bool(getattr(args, "resume", False))

    rows = iter_prompt_rows(prompt_dir, topics)
    if limit:
        rows = rows[:limit]

    manifest_rows = []
    failures = []
    for index, row in enumerate(rows, 1):
        topic = row["topic"]
        prompt_id = int(row["id"])
        out_path = output_dir / model_dir_name / topic / f"{prompt_id}.png"
        enhanced_prompt = build_enhanced_prompt(row["prompt"], topic, include_constraints)
        manifest_row = {
            "topic": topic,
            "id": prompt_id,
            "model": model,
            "backend": backend,
            "prompt": row["prompt"],
            "output_path": str(out_path),
            "status": "pending",
        }

        if resume and out_path.exists():
            manifest_row["status"] = "exists"
            manifest_rows.append(manifest_row)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if backend == "dry-run":
                manifest_row["status"] = "dry_run"
            elif backend == "openai":
                generate_openai_image(enhanced_prompt, out_path, model, size)
                manifest_row["status"] = "ok"
            elif backend == "replicate":
                replicate_model = getattr(args, "replicate_model", model)
                generate_replicate_image(
                    enhanced_prompt,
                    out_path,
                    replicate_model=replicate_model,
                    prompt_key=getattr(args, "replicate_prompt_key", "prompt"),
                    extra_input=getattr(args, "replicate_extra_input", None),
                    max_retries=int(getattr(args, "replicate_max_retries", 3)),
                    min_interval=float(getattr(args, "replicate_min_interval", 0.0)),
                )
                manifest_row["status"] = "ok"
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        except Exception as exc:
            manifest_row["status"] = "error"
            manifest_row["error"] = str(exc)
            failures.append(manifest_row)

        manifest_rows.append(manifest_row)
        if index % 25 == 0 or index == len(rows):
            print(f"generation progress: {index}/{len(rows)}")

    manifest_path = Path(getattr(args, "generation_manifest", output_dir / model_dir_name / "generation_manifest.jsonl"))
    write_jsonl(manifest_path, manifest_rows)
    print(f"Wrote generation manifest: {manifest_path}")
    if failures:
        print(f"Generation finished with {len(failures)} failures.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images for MathGen prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--prompt-dir", "--prompt_dir", dest="prompt_dir", type=Path, default=project_root / "data" / "prompt_data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=project_root / "data" / "generated_img")
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
    return parser.parse_args()


def main() -> None:
    run_generation(parse_args())


if __name__ == "__main__":
    main()
