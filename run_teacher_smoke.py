#!/usr/bin/env python3
"""Extract teacher source and run its unchanged synthetic PPO interface check.

No ROM, emulator, environment rollout, learning, or trained model is produced.
--prepare-only uses the standard library for extraction and syntax checks only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

STUDENT_FILES = {
    "student_observation.py", "student_actions.py",
    "student_reward.py", "student_model.py",
}
CONFIG_NAMES = {"COMMON_RULES", "PVE_AGENT_CONFIG", "PVE_TRAIN_CONFIG"}


def literal_assignments(source: str, wanted: set[str]) -> dict:
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    return found


def read_teacher_notebook(path: Path) -> tuple[dict, dict, dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    public, student, config = {}, {}, {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile $PVE_STUDENT_DIR/"):
            first, body = source.split("\n", 1)
            name = first.split("$PVE_STUDENT_DIR/", 1)[1].strip()
            if name not in STUDENT_FILES or name in student:
                raise ValueError(f"Unexpected or duplicate student source: {name}")
            compile(body, f"cell-{index}-{name}", "exec")
            student[name] = body
        elif source.startswith("%%"):
            # Independent PvP source is not executed by this check.
            continue
        else:
            compile(source, f"notebook-cell-{index}", "exec")
            values = literal_assignments(source, {"CORE_FILES"} | CONFIG_NAMES)
            if "CORE_FILES" in values:
                if public:
                    raise ValueError("Duplicate CORE_FILES definition")
                public = values.pop("CORE_FILES")
            config.update(values)
    if len(public) != 22 or "input_contract_smoke.py" not in public:
        raise ValueError("Expected the teacher release's 22 public Python files")
    if set(student) != STUDENT_FILES or not CONFIG_NAMES <= config.keys():
        raise ValueError("Missing teacher student source or configuration")
    for name, text in public.items():
        if Path(name).name != name or not name.endswith(".py") or not isinstance(text, str):
            raise ValueError(f"Invalid public source entry: {name}")
        compile(text, name, "exec")
    return public, student, config


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(notebook: Path, output: Path) -> tuple[Path, dict]:
    public, student, config = read_teacher_notebook(notebook)
    code_dir, student_dir = output / "code", output / "student/pve"
    for directory in (code_dir, student_dir, output / "tmp", output / "cache", output / "matplotlib"):
        directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for folder, files in ((code_dir, public), (student_dir, student)):
        for name, text in files.items():
            (folder / name).write_text(text, encoding="utf-8")
            hashes[str((folder / name).relative_to(output))] = hashlib.sha256(text.encode()).hexdigest()
    report_path = output / "smoke-report.json"
    job = {
        "sources": str(student_dir), "agent": config["PVE_AGENT_CONFIG"],
        "train": config["PVE_TRAIN_CONFIG"], "common": config["COMMON_RULES"],
        "report": str(report_path),
    }
    job_path = output / "smoke-job.json"
    write_json(job_path, job)
    prepared = {
        "status": "TEACHER_SOURCE_PREPARED", "notebook": str(notebook),
        "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        "public_file_count": len(public), "student_file_count": len(student),
        "source_sha256": hashes,
        "scope": "Source extraction and Python syntax only; no model, game, or learning executed",
    }
    write_json(output / "prepare-report.json", prepared)
    # Clear any stale success status before this invocation's actual check.
    write_json(report_path, {"status": "SYNTHETIC_CHECK_NOT_RUN", "scope": prepared["scope"]})
    return job_path, prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path,
                        default=Path(__file__).resolve().with_name("teacher_classroom_2026.ipynb"))
    parser.add_argument("--output", type=Path, default=Path("output/teacher-smoke"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    notebook, output = args.notebook.expanduser().resolve(), args.output.expanduser().resolve()
    if output == Path(__file__).resolve().parent:
        parser.error("--output must be an artifact subdirectory, not the repository root")
    try:
        job_path, prepared = prepare(notebook, output)
    except (OSError, ValueError, KeyError, SyntaxError, TypeError) as exc:
        print(f"SOURCE_PREPARATION_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": prepared["status"], "output": str(output),
                      "public_files": prepared["public_file_count"],
                      "student_files": prepared["student_file_count"]}), flush=True)
    if args.prepare_only:
        print("PREPARE_ONLY_COMPLETE: source/syntax only; synthetic model check NOT RUN", flush=True)
        return 0
    required = ("numpy", "cv2", "torch", "gymnasium", "stable_baselines3")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    report_path = output / "smoke-report.json"
    if missing:
        write_json(report_path, {"status": "DEPENDENCIES_MISSING", "missing": missing,
                                "scope": "No model check, game or learning executed"})
        print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
        print("Install requirements-demo.txt, then rerun without --prepare-only.", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(output / "tmp"),
                "TEMP": str(output / "tmp"), "TMP": str(output / "tmp"),
                "MPLCONFIGDIR": str(output / "matplotlib"), "XDG_CACHE_HOME": str(output / "cache")})
    command = [sys.executable, str(output / "code/input_contract_smoke.py"), "--job", str(job_path)]
    try:
        result = subprocess.run(command, cwd=output / "code", env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        write_json(report_path, {"status": "SYNTHETIC_INPUT_CONTRACT_FAILED", "error": str(exc),
                                "scope": "Synthetic interface only; not a game or training run"})
        print(f"SYNTHETIC_INPUT_CONTRACT_FAILED: {exc}", file=sys.stderr)
        return 1
    (output / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(result.stderr, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        write_json(report_path, {"status": "SYNTHETIC_INPUT_CONTRACT_FAILED", "returncode": result.returncode,
                                "scope": "Synthetic interface only; not a game or training run"})
        return result.returncode
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "SYNTHETIC_INPUT_CONTRACT_PASSED":
        print("Teacher process did not report the expected synthetic pass", file=sys.stderr)
        return 1
    print(f"TEACHER_SYNTHETIC_CHECK_COMPLETE: {report_path}", flush=True)
    print("No game was played, no PPO learning ran, and no useful trained model was saved.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
