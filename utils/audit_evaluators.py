#!/usr/bin/env python3
"""Audit MathGen evaluator organization without running model-specific logic.

This script is intended as a release gate for the public evaluator bundle. It
checks that every mapped evaluator has the API used by the runner, imports
cleanly, and avoids common draft artifacts such as hard-coded debug mode.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_TOPICS = {
    "angle",
    "counting",
    "fraction",
    "function",
    "plane",
    "open_scene",
    "set",
    "solid",
}

HELPER_NAMES = {
    "__init__.py",
    "angle_common.py",
    "counting_common.py",
    "function_common.py",
    "evaluator_utils.py",
    "ocr_label_utils.py",
    "open_scene_common.py",
    "set_common.py",
    "solid_common.py",
}

DRAFT_PATTERNS = {
    "TODO": re.compile(r"\bTODO\b"),
    "FIXME": re.compile(r"\bFIXME\b"),
    "DEBUG_TRUE": re.compile(r"^\s*DEBUG\s*=\s*True\s*$", flags=re.MULTILINE),
    "HAN_COMMENT_OR_DOC": re.compile(r"[\u4e00-\u9fff]"),
}


def has_eval_like_function(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in {"evaluate", "evaluate_plot", "semantic_venn_evaluator"}:
            return True
        if node.name.startswith(("verify_", "check_")):
            return True
    return False


def has_eval_like_symbol(module) -> bool:
    if callable(getattr(module, "evaluate", None)):
        return True
    for name, value in vars(module).items():
        if not callable(value):
            continue
        if name in {"evaluate_plot", "semantic_venn_evaluator"}:
            return True
        if name.startswith(("verify_", "check_")):
            return True
    return False


def load_manifest_scripts(project_root: Path) -> tuple[set[Path], Counter[str]]:
    scripts: set[Path] = set()
    rows_by_topic: Counter[str] = Counter()
    manifest = project_root / "data" / "mapping.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_topic[row["topic"]] += 1
            script = row.get("script")
            if script:
                scripts.add((project_root / script).resolve())
    return scripts, rows_by_topic


def has_callable_evaluate(module) -> bool:
    return callable(getattr(module, "evaluate", None))


def accepts_single_image_argument(fn) -> bool:
    """Return whether evaluate can be called as evaluate(image_path)."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    required_positionals = []
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if param.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            continue
        if param.default is inspect.Parameter.empty:
            required_positionals.append(param.name)
    return len(required_positionals) <= 1


def import_evaluator(path: Path):
    """Import an evaluator with its directory on sys.path, matching the runner."""
    module_name = f"_mathgen_audit_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to create module spec")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    eval_root = project_root / "data" / "evaluate_script"
    manifest_scripts, manifest_rows_by_topic = load_manifest_scripts(project_root)

    by_topic: defaultdict[str, Counter[str]] = defaultdict(Counter)
    issues: list[str] = []
    observed_topics: set[str] = set()

    for path in sorted(eval_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(eval_root)
        if len(rel.parts) == 1:
            if path.name not in HELPER_NAMES and not path.name.startswith("_"):
                issues.append(f"{rel}: root-level evaluator file should live under a topic directory")
            continue
        topic = rel.parts[0]
        observed_topics.add(topic)
        if topic not in EXPECTED_TOPICS:
            issues.append(f"{rel}: unexpected evaluator topic directory")
        is_manifest_script = path.resolve() in manifest_scripts
        is_helper = (
            path.name in HELPER_NAMES
            or path.name.endswith("_common.py")
            or path.name.endswith("_evaluator.py")
            or path.name.startswith("_")
        )
        by_topic[topic]["python_files"] += 1
        by_topic[topic]["manifest_scripts"] += int(is_manifest_script)
        by_topic[topic]["helpers"] += int(is_helper)

        try:
            ast_eval_like = has_eval_like_function(path)
        except SyntaxError as exc:
            issues.append(f"{rel}: syntax error: {exc}")
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in DRAFT_PATTERNS.items():
            if pattern.search(text):
                issues.append(f"{rel}: contains draft marker {name}")
        if is_manifest_script and ("from common import" in text or "import common as" in text):
            issues.append(f"{rel}: use a uniquely named topic common module, not generic common")
        if path.name == "common.py":
            issues.append(f"{rel}: use a topic-specific helper name instead of common.py")
        if is_manifest_script and ("spec_from_file_location" in text or "_COMMON_PATH" in text):
            issues.append(f"{rel}: common-module loader boilerplate should live in a helper or use a normal import")

        module = None
        if is_manifest_script:
            try:
                module = import_evaluator(path)
            except Exception as exc:
                issues.append(f"{rel}: import failed: {type(exc).__name__}: {exc}")
            else:
                if not has_callable_evaluate(module):
                    issues.append(f"{rel}: mapped evaluator must expose callable evaluate(image_path)")
                elif not accepts_single_image_argument(module.evaluate):
                    issues.append(f"{rel}: evaluate must be callable as evaluate(image_path)")
        eval_like = ast_eval_like or bool(module is not None and has_eval_like_symbol(module))
        by_topic[topic]["eval_like"] += int(eval_like)
        if is_manifest_script and not eval_like:
            issues.append(f"{rel}: listed in manifest but has no evaluator function")
        if eval_like and not is_manifest_script and not is_helper:
            issues.append(f"{rel}: evaluator-like file is not listed in mapping.csv")
        if not eval_like and not is_helper:
            issues.append(f"{rel}: neither helper nor evaluator-like script")

    missing_topics = EXPECTED_TOPICS - observed_topics
    if missing_topics:
        issues.append(f"missing evaluator topic directories: {', '.join(sorted(missing_topics))}")

    print("Evaluator organization by topic:")
    for topic, counts in sorted(by_topic.items()):
        manifest_rows = manifest_rows_by_topic.get(topic, 0)
        print(
            f"  {topic:10s} files={counts['python_files']:3d} "
            f"mapped_rows={manifest_rows:3d} unique_scripts={counts['manifest_scripts']:3d} "
            f"helpers={counts['helpers']:2d} "
            f"eval_like={counts['eval_like']:3d}"
        )

    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("Evaluator audit passed.")


if __name__ == "__main__":
    main()
