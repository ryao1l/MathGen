#!/usr/bin/env python3
"""Run MathGen Script-as-a-Judge evaluators over generated images.

The runner keeps the release layout simple:

* prompts live in ``data/prompt_data/<topic>.jsonl``;
* images live in ``data/generated_img/<model>/<topic>/<id>.png``;
* ``data/mapping.csv`` maps each prompt id to its evaluator.

Each evaluator returns a JSON-serializable report with ``passed``, ``criteria``,
and ``meta`` fields. This script normalizes those reports and writes both
per-image decisions and aggregate summaries.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_TOPICS = [
    "counting",
    "angle",
    "fraction",
    "plane",
    "function",
    "solid",
    "set",
    "open_scene",
]

_MODULE_CACHE = {}
_MODULE_CACHE_LOCK = threading.Lock()


def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL prompt file."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path, base_dir: Optional[Path] = None) -> Dict[tuple, str]:
    """Load the prompt-id to evaluator-script mapping."""
    out: Dict[tuple, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("status")
            if status and status not in {"copied", "generated"}:
                continue
            topic = row["topic"]
            prompt_id = int(row.get("id") or row.get("new_id"))
            script = row.get("script") or ""
            if script:
                script_path = Path(script)
                if base_dir is not None and not script_path.is_absolute():
                    script = str(base_dir / script_path)
                out[(topic, prompt_id)] = script
    return out


def discover_models(generated_img_dir: Path, topics: List[str]) -> List[str]:
    """Return model directories that contain at least one requested topic."""
    models = []
    for path in sorted(generated_img_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if any((path / topic).is_dir() for topic in topics):
            models.append(path.name)
    return models


def evaluator_image_arg(script_path: Path) -> str:
    """Detect whether an older evaluator expects --image or --img_path."""
    try:
        text = script_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "--image"
    if "--img_path" in text and "--image" not in text:
        return "--img_path"
    return "--image"


def to_jsonable(value):
    if isinstance(value, dict):
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


def normalize_report(report) -> dict:
    """Convert evaluator output to the common report schema."""
    report = to_jsonable(report)
    if isinstance(report, dict):
        extra = {k: v for k, v in report.items() if k not in {"passed", "criteria", "meta"}}
        meta = report.get("meta") or {}
        if extra:
            meta = {**meta, **extra}
        return {
            "passed": bool(report.get("passed", False)),
            "criteria": to_jsonable(report.get("criteria") or {}),
            "meta": to_jsonable(meta),
        }
    if isinstance(report, (list, tuple)) and report:
        return {
            "passed": bool(report[0]),
            "criteria": {},
            "meta": {"raw_report": report},
        }
    return {"passed": False, "criteria": {}, "meta": {"raw_report": report}}


def parse_report_stdout(stdout: str) -> Optional[dict]:
    """Parse reports printed by standalone evaluator scripts."""
    text = stdout.strip()
    if not text:
        return None

    for candidate in [text, text.splitlines()[-1].strip()]:
        try:
            return normalize_report(json.loads(candidate))
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            parsed = None
        if parsed is not None:
            return normalize_report(parsed)

    for idx in [m.start() for m in re.finditer(r"\{", text)][::-1]:
        candidate = text[idx:].strip()
        try:
            return normalize_report(json.loads(candidate))
        except json.JSONDecodeError:
            pass
        try:
            return normalize_report(ast.literal_eval(candidate))
        except (SyntaxError, ValueError):
            pass

    match = re.search(r"(?:Final\s+result|Result):\s*(PASS|FAIL)", text, flags=re.IGNORECASE)
    if match:
        return {
            "passed": match.group(1).upper() == "PASS",
            "criteria": {},
            "meta": {"raw_stdout": text},
        }
    return None


def load_eval_module(script_path: Path):
    """Import an evaluator module while allowing sibling common.py imports."""
    key = str(script_path.resolve())
    with _MODULE_CACHE_LOCK:
        if key in _MODULE_CACHE:
            return _MODULE_CACHE[key]
        module_name = f"_mathgen_eval_{abs(hash(key))}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        old_path = list(sys.path)
        sys.path.insert(0, str(script_path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
        _MODULE_CACHE[key] = module
        return module


def evaluate_by_import(script_path: Path, image_path: Path, prompt_file: Optional[str] = None, topic: str = "") -> Optional[dict]:
    """Run an evaluator through its Python API when possible."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    module = load_eval_module(script_path)
    if module is None:
        return None
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        return None
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        if topic == "counting":
            raw_report = evaluate(str(image_path), counting_prompt_file=prompt_file)
        else:
            raw_report = evaluate(str(image_path))
    if raw_report is None:
        return parse_report_stdout(log_buffer.getvalue())
    return normalize_report(raw_report)


def fill_report(base: dict, report: dict) -> dict:
    """Attach normalized evaluator results to a CSV/JSONL row."""
    passed = bool(report.get("passed", False))
    base.update(
        {
            "status": "ok",
            "passed": "1" if passed else "0",
            "criteria_json": json.dumps(to_jsonable(report.get("criteria", {})), ensure_ascii=False, sort_keys=True),
            "meta_json": json.dumps(to_jsonable(report.get("meta", {})), ensure_ascii=False, sort_keys=True),
        }
    )
    return base


def run_one(task: dict, python_bin: str, timeout: float) -> dict:
    """Evaluate one model/topic/id image."""
    image_path = Path(task["image_path"])
    script_path = Path(task["script_path"])
    base = {
        "topic": task["topic"],
        "id": task["id"],
        "subtopic": task.get("subtopic", ""),
        "model": task["model"],
        "image_path": str(image_path),
        "script_path": str(script_path),
        "status": "unknown",
        "passed": "",
        "returncode": "",
        "elapsed_sec": "",
        "error": "",
        "criteria_json": "",
        "meta_json": "",
        "stdout": "",
        "stderr": "",
    }

    if not image_path.exists():
        base.update({"status": "missing_image", "passed": ""})
        return base
    if not script_path.exists():
        base.update({"status": "missing_script", "passed": ""})
        return base

    forced_status = task.get("forced_status")
    if forced_status:
        base.update(
            {
                "status": forced_status,
                "passed": "",
                "error": task.get("forced_error", ""),
            }
        )
        return base

    if task["topic"] == "counting":
        start = time.monotonic()
        try:
            report = evaluate_by_import(script_path, image_path, task.get("prompt_file"), task["topic"])
        except Exception as exc:
            base.update(
                {
                    "status": "error",
                    "elapsed_sec": f"{time.monotonic() - start:.3f}",
                    "error": str(exc),
                }
            )
            return base
        base["elapsed_sec"] = f"{time.monotonic() - start:.3f}"
        if report is None:
            base.update({"status": "error", "error": "counting import fallback unavailable"})
            return base
        return fill_report(base, report)

    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    start = time.monotonic()
    try:
        cmd = [python_bin, str(script_path), evaluator_image_arg(script_path), str(image_path)]
        prompt_file = task.get("prompt_file")
        if prompt_file and task["topic"] == "counting":
            cmd.extend(["--prompt_file", str(prompt_file)])

        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        base.update(
            {
                "status": "timeout",
                "elapsed_sec": f"{time.monotonic() - start:.3f}",
                "error": f"timeout after {timeout}s",
                "stdout": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            }
        )
        return base

    elapsed = time.monotonic() - start
    base["returncode"] = str(proc.returncode)
    base["elapsed_sec"] = f"{elapsed:.3f}"
    base["stdout"] = proc.stdout[-2000:]
    base["stderr"] = proc.stderr[-2000:]

    if proc.returncode != 0:
        if "is not JSON serializable" in proc.stderr:
            try:
                report = evaluate_by_import(script_path, image_path, task.get("prompt_file"), task["topic"])
            except Exception as exc:
                report = None
                base["error"] = f"import fallback failed: {exc}"
            if report is not None:
                return fill_report(base, report)
        base.update({"status": "error", "error": proc.stderr.strip().splitlines()[-1:] and proc.stderr.strip().splitlines()[-1] or "nonzero exit"})
        return base

    report = parse_report_stdout(proc.stdout)
    if report is None:
        try:
            report = evaluate_by_import(script_path, image_path, task.get("prompt_file"), task["topic"])
        except Exception:
            report = None
    if report is None:
        base.update({"status": "bad_json", "error": "unable to parse evaluator output"})
        return base

    return fill_report(base, report)


def build_tasks(
    prompt_dir: Path,
    generated_img_dir: Path,
    manifest: Dict[tuple, str],
    topics: List[str],
    models: Optional[List[str]],
) -> List[dict]:
    """Construct all image/evaluator jobs requested by CLI filters."""
    if models is None:
        models = discover_models(generated_img_dir, topics)

    tasks = []
    for topic in topics:
        prompt_path = prompt_dir / f"{topic}.jsonl"
        if not prompt_path.exists():
            continue
        rows = read_jsonl(prompt_path)
        topic_models = [m for m in models if (generated_img_dir / m / topic).is_dir()]
        for row in rows:
            prompt_id = int(row["id"])
            script = manifest.get((topic, prompt_id), "")
            for model in topic_models:
                tasks.append(
                    {
                        "topic": topic,
                        "id": prompt_id,
                        "subtopic": row.get("subtopic", ""),
                        "model": model,
                        "image_path": str(generated_img_dir / model / topic / f"{prompt_id}.png"),
                        "script_path": script,
                        "prompt_file": str(prompt_path),
                    }
                )
    return tasks


def preflight_script(script_path: str, image_path: str, prompt_file: str, topic: str, python_bin: str, timeout: float) -> Optional[str]:
    """Probe special evaluators so missing optional dependencies are reported."""
    if not script_path or not image_path:
        return None
    cmd = [python_bin, script_path, evaluator_image_arg(Path(script_path)), image_path]
    if topic == "counting":
        cmd.extend(["--prompt_file", prompt_file])
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"preflight timeout after {timeout}s"
    if proc.returncode == 0:
        return None
    stderr = proc.stderr.strip()
    if "RT-DETR dependencies are missing" in stderr or "torchvision::nms" in stderr:
        return "RT-DETR/torchvision dependency unavailable"
    return None


def mark_unavailable_scripts(tasks: List[dict], python_bin: str, timeout: float, topics: set) -> Dict[str, str]:
    """Mark tasks whose evaluator dependencies are unavailable."""
    by_script: Dict[str, dict] = {}
    for task in tasks:
        if task["topic"] not in topics:
            continue
        script = task.get("script_path", "")
        if script and script not in by_script and Path(task["image_path"]).exists():
            by_script[script] = task

    unavailable: Dict[str, str] = {}
    for script, task in sorted(by_script.items()):
        reason = preflight_script(
            script_path=script,
            image_path=task["image_path"],
            prompt_file=task.get("prompt_file", ""),
            topic=task["topic"],
            python_bin=python_bin,
            timeout=timeout,
        )
        if reason:
            unavailable[script] = reason

    for task in tasks:
        reason = unavailable.get(task.get("script_path", ""))
        if reason:
            task["forced_status"] = "script_unavailable"
            task["forced_error"] = reason
    return unavailable


def row_key(row: dict) -> tuple:
    return (row["topic"], str(row["id"]), row["model"])


def read_existing_jsonl(path: Path) -> List[dict]:
    """Load resumable per-image rows from a previous JSONL output."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_summary(rows: List[dict], output_dir: Path) -> None:
    """Write model-topic and topic-level summary CSV files."""
    by_topic_model = defaultdict(lambda: Counter())
    by_topic = defaultdict(lambda: Counter())

    for row in rows:
        topic = row["topic"]
        model = row["model"]
        status = row["status"]
        passed = row["passed"]
        by_topic_model[(topic, model)]["total"] += 1
        by_topic_model[(topic, model)][status] += 1
        by_topic[topic]["total"] += 1
        by_topic[topic][status] += 1
        if status == "ok":
            by_topic_model[(topic, model)]["evaluated"] += 1
            by_topic[topic]["evaluated"] += 1
            if passed == "1":
                by_topic_model[(topic, model)]["correct"] += 1
                by_topic[topic]["correct"] += 1

    summary_path = output_dir / "model_topic_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["topic", "model", "correct", "evaluated", "total", "accuracy", "ok", "missing_image", "missing_script", "script_unavailable", "error", "timeout", "bad_json"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (topic, model), stats in sorted(by_topic_model.items()):
            evaluated = stats["evaluated"]
            writer.writerow(
                {
                    "topic": topic,
                    "model": model,
                    "correct": stats["correct"],
                    "evaluated": evaluated,
                    "total": stats["total"],
                    "accuracy": f"{(stats['correct'] / evaluated):.6f}" if evaluated else "",
                    "ok": stats["ok"],
                    "missing_image": stats["missing_image"],
                    "missing_script": stats["missing_script"],
                    "script_unavailable": stats["script_unavailable"],
                    "error": stats["error"],
                    "timeout": stats["timeout"],
                    "bad_json": stats["bad_json"],
                }
            )

    topic_path = output_dir / "topic_summary.csv"
    with topic_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["topic", "correct", "evaluated", "total", "accuracy", "ok", "missing_image", "missing_script", "script_unavailable", "error", "timeout", "bad_json"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for topic, stats in sorted(by_topic.items()):
            evaluated = stats["evaluated"]
            writer.writerow(
                {
                    "topic": topic,
                    "correct": stats["correct"],
                    "evaluated": evaluated,
                    "total": stats["total"],
                    "accuracy": f"{(stats['correct'] / evaluated):.6f}" if evaluated else "",
                    "ok": stats["ok"],
                    "missing_image": stats["missing_image"],
                    "missing_script": stats["missing_script"],
                    "script_unavailable": stats["script_unavailable"],
                    "error": stats["error"],
                    "timeout": stats["timeout"],
                    "bad_json": stats["bad_json"],
                }
            )


def run_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation from a parsed namespace."""
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(args.manifest, project_root)
    tasks = build_tasks(args.prompt_dir, args.generated_img_dir, manifest, args.topics, args.models)
    if args.limit:
        tasks = tasks[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "per_question_model_results.jsonl"
    csv_path = args.output_dir / "per_question_model_results.csv"

    existing_rows = read_existing_jsonl(jsonl_path) if args.resume else []
    done_keys = {row_key(row) for row in existing_rows}
    if done_keys:
        tasks = [task for task in tasks if row_key(task) not in done_keys]

    unavailable = {}
    if not args.no_preflight:
        unavailable = mark_unavailable_scripts(tasks, args.python_bin, args.preflight_timeout, set(args.preflight_topics))

    fields = [
        "topic",
        "id",
        "subtopic",
        "model",
        "image_path",
        "script_path",
        "status",
        "passed",
        "returncode",
        "elapsed_sec",
        "error",
        "criteria_json",
        "meta_json",
        "stdout",
        "stderr",
    ]

    rows = list(existing_rows)
    print(f"Running {len(tasks)} eval jobs with {args.workers} workers")
    if existing_rows:
        print(f"Resuming with {len(existing_rows)} existing rows")
    if unavailable:
        print("Unavailable scripts:")
        for script, reason in sorted(unavailable.items()):
            print(f"  {script}: {reason}")
    jsonl_mode = "a" if args.resume and existing_rows else "w"
    with jsonl_path.open(jsonl_mode, encoding="utf-8") as jf:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(run_one, task, args.python_bin, args.timeout) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                if i % 100 == 0 or i == len(tasks):
                    print(f"completed {i}/{len(tasks)}")

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, args.output_dir)
    status_counts = Counter(row["status"] for row in rows)
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {args.output_dir / 'model_topic_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'topic_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MathGen deterministic evaluators over generated images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--prompt-dir", type=Path, default=project_root / "data" / "prompt_data", help="Directory containing topic JSONL prompt files.")
    parser.add_argument("--generated-img-dir", type=Path, default=project_root / "data" / "generated_img", help="Directory containing <model>/<topic>/<id>.png images.")
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "mapping.csv", help="CSV file that maps topic/id pairs to evaluator scripts.")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results" / "current" / "evaluation", help="Directory for per-image outputs and summaries.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for subprocess-based evaluators.")
    parser.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS, help="Topics to evaluate.")
    parser.add_argument("--models", nargs="*", default=None, help="Model directories to evaluate; defaults to all discovered models.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel evaluator workers.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-image evaluator timeout in seconds.")
    parser.add_argument("--preflight-timeout", type=float, default=10.0, help="Timeout for dependency preflight probes.")
    parser.add_argument("--no-preflight", action="store_true", help="Skip dependency preflight checks.")
    parser.add_argument("--preflight-topics", nargs="*", default=["counting"], help="Topics that need preflight dependency probes.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing JSONL output and skip completed rows.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of tasks for smoke testing.")
    return parser.parse_args()


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
