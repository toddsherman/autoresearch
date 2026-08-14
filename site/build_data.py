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

    # A run left "pending" that nevertheless produced a final result.json ran to
    # completion — its keep/discard verdict just never got recorded before the
    # agent moved on. For a finished night, present it as "discard" (it was not
    # advanced) rather than a stray "running" badge. Raw telemetry is untouched.
    status = record.get("status", "pending")
    if status == "pending" and result is not None:
        status = "discard"

    # Unified metrics so every run carries both objectives regardless of which
    # night recorded which as primary. Night 1 didn't record complete_game_rate
    # in the index, but each run's result.json has it as final_sample_stats.
    fss = (result or {}).get("final_sample_stats", {})
    complete_game_rate = record.get("complete_game_rate") or fss.get("complete_rate", 0.0)
    legal_move_rate = record.get("legal_move_rate") or fss.get("legality_rate", 0.0)

    return {
        "run_id": record["run_id"],
        "seq": record["seq"],
        "commit": record.get("commit", ""),
        "description": record.get("description", ""),
        "status": status,
        "val_bpb": record.get("val_bpb", 0.0),
        "strength": record.get("strength", 0.0),
        "complete_game_rate": complete_game_rate,
        "legal_move_rate": legal_move_rate,
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


# Each night declares which metric it optimized so the site renders the timeline
# and champions in the right direction. Add an entry here when a night finishes.
#   primary.direction: "min" (val_bpb, lower better) or "max" (rate, higher better)
NIGHTS = [
    {
        "id": "night1",
        "act": "Act I",
        "telemetry_dir": "telemetry",
        "title": "Night 1 — learning to compress",
        "subtitle": "the agent minimizes val_bpb; legal play emerges as a side effect, then diverges",
        "hardware": "1x H100",
        "primary": {"key": "val_bpb", "label": "val_bpb", "direction": "min", "unit": "bits/byte", "fmt": "bpb"},
    },
    {
        "id": "night2",
        "act": "Act II",
        "telemetry_dir": "telemetry-night2",
        "title": "Night 2 — learning to play",
        "subtitle": "the agent maximizes complete legal games; a different objective, a different model",
        "hardware": "1x H100",
        "primary": {"key": "complete_game_rate", "label": "complete games", "direction": "max", "unit": "%", "fmt": "pct"},
    },
    {
        "id": "night3",
        "act": "Act III",
        "telemetry_dir": "telemetry-night3/telemetry",
        "title": "Night 3 — learning to win",
        "subtitle": "the agent maximizes playing strength against a fixed ladder of opponents",
        "hardware": "1x H100 NVL",
        "primary": {"key": "strength", "label": "strength", "direction": "max", "unit": "score", "fmt": "score"},
        "lean": True,  # chart-only: drop per-run games/loss_curve to keep data.js small
    },
    {
        "id": "probe",
        "act": "Act IV",
        "type": "probe",
        "probe_file": "probe_results.json",
        "title": "World model — looking inside",
        "subtitle": "is the Othello board represented inside the trained model? A linear probe for what it knows.",
        "hardware": "1x H100",
    },
]


def build_probe_night(cfg):
    """Load the world-model probe results into a site bundle (Act III)."""
    night = {k: cfg[k] for k in ("id", "act", "title", "subtitle", "hardware")}
    night["type"] = "probe"
    path = cfg["probe_file"]
    if not os.path.exists(path):
        night.update({"status": "pending", "probe": None})
        return night
    night.update({"status": "complete", "probe": json.load(open(path))})
    return night


def build_night(cfg):
    """Load one night's telemetry into a site bundle, or a 'pending' placeholder
    if its telemetry doesn't exist yet."""
    tel = cfg["telemetry_dir"]
    index_path = os.path.join(tel, "night.jsonl")
    night = {k: cfg[k] for k in ("id", "act", "title", "subtitle", "hardware", "primary")}
    if not os.path.exists(index_path):
        night.update({"status": "pending", "runs": [], "branch": None})
        return night
    records = [json.loads(l) for l in open(index_path) if l.strip()]
    runs_dir = os.path.join(tel, "runs")
    runs = [load_run(runs_dir, r) for r in sorted(records, key=lambda r: r["seq"])]
    branch = None
    for r in runs:
        p = os.path.join(runs_dir, r["run_id"], "meta.json")
        if os.path.exists(p) and json.load(open(p)).get("branch"):
            branch = json.load(open(p))["branch"]
            break
    night.update({"status": "complete", "runs": runs, "branch": branch})
    return night


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("site", "data.js"))
    args = parser.parse_args()

    nights = []
    for cfg in NIGHTS:
        night = build_probe_night(cfg) if cfg.get("type") == "probe" else build_night(cfg)
        # "lean" nights are charted by metric only (no board playback), so drop the
        # heavy per-run payload (games, loss curves, diffs) to keep data.js small.
        if cfg.get("lean") and night.get("runs"):
            keep = ("run_id", "seq", "description", "status", "strength",
                    "val_bpb", "complete_game_rate", "legal_move_rate",
                    "started_at", "wall_seconds")
            night["runs"] = [{k: r[k] for k in keep if k in r} for r in night["runs"]]
        nights.append(night)
    if not any(n["status"] == "complete" for n in nights):
        sys.exit("No night telemetry found — has any night been run?")

    research = {
        "meta": {
            "title": "autoresearch: othello",
            "tagline": "an AI agent teaches a small GPT to play Othello, five minutes at a time — a multi-night research journey",
            "generated_at": time.time(),
        },
        "nights": nights,
    }
    with open(args.out, "w") as f:
        f.write("window.RESEARCH_DATA = ")
        json.dump(research, f, separators=(",", ":"))
        f.write(";\n")
    summary = ", ".join(f"{n['id']}:{len(n.get('runs', []))}runs/{n['status']}" for n in nights)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) // 1024} KB): {summary}")


if __name__ == "__main__":
    main()
