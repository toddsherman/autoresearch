"""
Bundle a real telemetry/ directory (see telemetry-spec.md) into site/data.js —
the same format site/demo_data.py emits, so the playback site works identically
on synthetic and real nights.

Usage:
    python site/build_data.py [--telemetry-dir telemetry] [--out site/data.js]
"""

import argparse
import json
import os
import sys
import time

GAMES_PER_MILESTONE = 3   # shipped per milestone (stats stay full-sample)
LOSS_CURVE_POINTS = 150


def subsample(curve, n):
    if len(curve) <= n:
        return curve
    step = (len(curve) - 1) / (n - 1)
    return [curve[round(i * step)] for i in range(n)]


def load_run(runs_dir, record):
    run_dir = os.path.join(runs_dir, record["run_id"])
    meta, steps, samples, result = {}, [], [], None
    p = os.path.join(run_dir, "meta.json")
    if os.path.exists(p):
        meta = json.load(open(p))
    p = os.path.join(run_dir, "steps.jsonl")
    if os.path.exists(p):
        steps = [json.loads(l) for l in open(p) if l.strip()]
    p = os.path.join(run_dir, "samples.jsonl")
    if os.path.exists(p):
        samples = [json.loads(l) for l in open(p) if l.strip()]
    p = os.path.join(run_dir, "result.json")
    if os.path.exists(p):
        result = json.load(open(p))

    startup = 25.0
    if steps:
        startup = max(5.0, steps[0].get("t", 25.0))

    loss_curve = subsample([[s["progress"], s["loss"]] for s in steps], LOSS_CURVE_POINTS)

    milestones = []
    for s in sorted(samples, key=lambda s: s["milestone"]):
        st = s.get("stats", {})
        milestones.append({
            "milestone": s["milestone"],
            "progress": s["progress"],
            "step": s["step"],
            "legality_rate": st.get("legality_rate", 0.0),
            "complete_rate": st.get("complete_rate", 0.0),
            "mean_legal_prefix": st.get("mean_legal_prefix", 0.0),
            "games": [
                {k: g.get(k) for k in
                 ("moves", "total_moves", "legal_moves", "is_complete", "black", "white", "first_illegal")}
                for g in s.get("games", [])[:GAMES_PER_MILESTONE]
            ],
        })

    final = None
    if result:
        fss = result.get("final_sample_stats", {})
        vr = result.get("vs_random", {})
        final = {
            "legality_rate": fss.get("legality_rate", 0.0),
            "complete_rate": fss.get("complete_rate", 0.0),
            "winrate": vr.get("winrate", 0.0),
            "vs_random": {k: vr.get(k) for k in ("n_games", "wins", "draws", "losses")},
        }

    return {
        "run_id": record["run_id"],
        "seq": record["seq"],
        "commit": record.get("commit", ""),
        "description": record.get("description", ""),
        "status": record.get("status", "pending"),
        "val_bpb": record.get("val_bpb", 0.0),
        "peak_vram_mb": record.get("peak_vram_mb", 0.0),
        "started_at": record["started_at"],
        "wall_seconds": record.get("wall_seconds", 330.0),
        "startup_seconds": startup,
        "num_params": round((meta.get("num_params") or 0)),
        "diff": meta.get("train_py_diff", ""),
        "loss_curve": loss_curve,
        "milestones": milestones,
        "final": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-dir", default="telemetry")
    parser.add_argument("--out", default=os.path.join("site", "data.js"))
    parser.add_argument("--title", default="autoresearch: othello")
    parser.add_argument("--hardware", default="1x H100")
    args = parser.parse_args()

    index_path = os.path.join(args.telemetry_dir, "night.jsonl")
    if not os.path.exists(index_path):
        sys.exit(f"No {index_path} found — has a night been run?")
    records = [json.loads(l) for l in open(index_path) if l.strip()]
    runs_dir = os.path.join(args.telemetry_dir, "runs")
    runs = [load_run(runs_dir, r) for r in sorted(records, key=lambda r: r["seq"])]

    branch = ""
    for r in runs:
        # branch lives in per-run meta; grab the first one available
        p = os.path.join(runs_dir, r["run_id"], "meta.json")
        if os.path.exists(p):
            branch = json.load(open(p)).get("branch", "")
            if branch:
                break

    bundle = {
        "meta": {
            "title": args.title,
            "subtitle": "an agent teaching a GPT the game of Othello, 5 minutes at a time",
            "hardware": args.hardware,
            "branch": branch or "unknown",
            "generated_at": time.time(),
            "is_demo": False,
        },
        "runs": runs,
    }
    with open(args.out, "w") as f:
        f.write("window.NIGHT_DATA = ")
        json.dump(bundle, f, separators=(",", ":"))
        f.write(";\n")
    print(f"Wrote {args.out} ({os.path.getsize(args.out) // 1024} KB): {len(runs)} runs")


if __name__ == "__main__":
    main()
