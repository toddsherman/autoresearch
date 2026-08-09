"""
Experiment wrapper (read-only for the research agent).

Launches train.py with a fresh telemetry run directory, captures metadata
(commit, branch, train.py snapshot and diff), enforces the 10-minute kill
rule, and maintains the machine-readable night.jsonl index that the playback
website consumes.

Usage:
    uv run run_experiment.py --desc "increase matrix LR to 0.06"
    uv run run_experiment.py --record-status keep     # after comparing val_bpb
    uv run run_experiment.py --record-status discard  # also prunes checkpoint
    uv run run_experiment.py --record-status crash

The wrapper relays all train.py output to stdout, so redirect to run.log and
grep exactly as before:  uv run run_experiment.py --desc "..." > run.log 2>&1
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = os.environ.get("AUTORESEARCH_TELEMETRY_DIR", os.path.join(REPO_ROOT, "telemetry"))
RUNS_DIR = os.path.join(TELEMETRY_DIR, "runs")
NIGHT_INDEX = os.path.join(TELEMETRY_DIR, "night.jsonl")
TRAIN_CMD = os.environ.get("AUTORESEARCH_TRAIN_CMD", "uv run train.py")
TIMEOUT_SECONDS = 600  # hard kill per program.md

SUMMARY_KEYS = [
    "val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
    "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth",
]


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _read_index():
    if not os.path.exists(NIGHT_INDEX):
        return []
    with open(NIGHT_INDEX) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_index(records):
    tmp = NIGHT_INDEX + ".tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, NIGHT_INDEX)


def start_run(desc):
    os.makedirs(RUNS_DIR, exist_ok=True)
    seq = len([d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]) + 1
    commit = _git("rev-parse", "--short=7", "HEAD") or "nocommit"
    run_id = f"run_{seq:03d}_{commit}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Metadata the training process can't know about itself
    meta = {
        "run_id": run_id,
        "seq": seq,
        "description": desc,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit_subject": _git("log", "-1", "--format=%s"),
        "started_at": time.time(),
        "train_py_diff": _git("diff", "HEAD^..HEAD", "--", "train.py"),
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.copy(os.path.join(REPO_ROOT, "train.py"),
                os.path.join(run_dir, "train_snapshot.py"))

    # Launch training with output relayed to both stdout and train.log
    env = dict(os.environ, AUTORESEARCH_RUN_DIR=run_dir)
    log_path = os.path.join(run_dir, "train.log")
    t0 = time.time()
    timed_out = False
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            TRAIN_CMD.split(), cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                if time.time() - t0 > TIMEOUT_SECONDS:
                    timed_out = True
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    break
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    # Parse the final summary out of the log
    summary = {}
    with open(log_path) as f:
        for line in f:
            m = re.match(r"^(\w+):\s+(-?[\d.]+)\s*$", line)
            if m and m.group(1) in SUMMARY_KEYS:
                summary[m.group(1)] = float(m.group(2))

    crashed = timed_out or proc.returncode != 0 or "val_bpb" not in summary
    record = {
        "run_id": run_id,
        "seq": seq,
        "started_at": meta["started_at"],
        "wall_seconds": round(time.time() - t0, 1),
        "commit": commit,
        "description": desc,
        "status": "crash" if crashed else "pending",
        "val_bpb": summary.get("val_bpb", 0.0),
        "peak_vram_mb": summary.get("peak_vram_mb", 0.0),
        "timed_out": timed_out,
        "exit_code": proc.returncode,
    }
    records = _read_index()
    records.append(record)
    _write_index(records)

    if timed_out:
        print(f"\nrun_experiment: TIMED OUT after {TIMEOUT_SECONDS}s and was killed", flush=True)
    print(f"run_experiment: recorded {run_id} (status={record['status']})", flush=True)
    return 1 if crashed else 0


def record_status(status):
    records = _read_index()
    if not records:
        print("run_experiment: no runs recorded yet", file=sys.stderr)
        return 1
    record = records[-1]
    record["status"] = status
    _write_index(records)
    if status == "discard":
        ckpt = os.path.join(RUNS_DIR, record["run_id"], "checkpoint.pt")
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print(f"run_experiment: {record['run_id']} marked {status}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one autoresearch experiment with telemetry")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--desc", type=str, help="Short description of what this experiment tries")
    group.add_argument("--record-status", choices=["keep", "discard", "crash"],
                       help="Record the keep/discard decision for the most recent run")
    args = parser.parse_args()
    if args.record_status:
        sys.exit(record_status(args.record_status))
    sys.exit(start_run(args.desc))
