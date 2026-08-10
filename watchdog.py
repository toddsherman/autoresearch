"""
Convergence watchdog for an autoresearch night (read-only for the agent).

Tails telemetry/night.jsonl and stops the run when it has stopped learning —
early-stopping-with-patience on the primary metric — or when a hard ceiling is
reached, whichever comes first. This is what lets a night halt itself instead
of running to exhaustion (or until the money runs out).

Primary metric defaults to complete_game_rate (HIGHER is better). An experiment
counts as a real improvement only if it beats the best-so-far by at least
MIN_DELTA, so the sub-threshold twiddling that dominates a converged tail does
not reset the patience clock. Patience is set deliberately larger than the
longest pre-breakthrough plateau observed on night 1 (~7 experiments), so a
genuine plateau-before-breakthrough is not amputated.

On trip it:
  - writes telemetry/STOP.json with the reason and evidence,
  - halts the agent by killing its tmux session (no further experiments),
  - optionally stops the pod itself if RUNPOD_API_KEY + RUNPOD_POD_ID are set
    (fully hands-off); otherwise the compute idles until an external check
    stops the pod.

Usage:
    python watchdog.py                       # defaults
    python watchdog.py --patience 20 --min-delta 0.01 --max-experiments 120
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = os.environ.get("AUTORESEARCH_TELEMETRY_DIR", os.path.join(REPO_ROOT, "telemetry"))
NIGHT_INDEX = os.path.join(TELEMETRY_DIR, "night.jsonl")
STOP_MARKER = os.path.join(TELEMETRY_DIR, "STOP.json")


def read_index():
    if not os.path.exists(NIGHT_INDEX):
        return []
    with open(NIGHT_INDEX) as f:
        return [json.loads(line) for line in f if line.strip()]


def halt_agent(session="agent"):
    subprocess.run(["tmux", "kill-session", "-t", session],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_pod_via_api():
    """Best-effort pod stop through the RunPod REST API. Returns a status str.
    Only fires if the user provisioned RUNPOD_API_KEY + RUNPOD_POD_ID."""
    key = os.environ.get("RUNPOD_API_KEY")
    pod = os.environ.get("RUNPOD_POD_ID")
    if not (key and pod):
        return "no api key/pod id — leaving pod running (external stop required)"
    url = f"https://rest.runpod.io/v1/pods/{pod}/stop"
    req = urllib.request.Request(url, method="POST",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return f"pod stop requested via API (HTTP {resp.status})"
    except Exception as e:  # noqa: BLE001 — best effort, never crash the watchdog
        return f"pod stop API call failed: {e}"


def trip(reason, evidence, args):
    marker = {
        "reason": reason,
        "evidence": evidence,
        "stopped_at_unix": int(time.time()),
        "metric": args.metric,
    }
    with open(STOP_MARKER, "w") as f:
        json.dump(marker, f, indent=2)
    print(f"\nwatchdog: STOP — {reason}", flush=True)
    print(f"watchdog: {json.dumps(evidence)}", flush=True)
    halt_agent(args.agent_session)
    print(f"watchdog: agent halted; {stop_pod_via_api()}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="complete_game_rate")
    p.add_argument("--goal", choices=["max", "min"], default="max",
                   help="max: higher is better (complete_game_rate); min: val_bpb")
    p.add_argument("--min-delta", type=float, default=0.01,
                   help="minimum improvement over best-so-far that counts as real")
    p.add_argument("--patience", type=int, default=20,
                   help="experiments with no real improvement before stopping")
    p.add_argument("--max-experiments", type=int, default=150,
                   help="hard ceiling on total experiments")
    p.add_argument("--max-wall-hours", type=float, default=12.0,
                   help="hard ceiling on wall-clock hours since watchdog start")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--agent-session", default="agent")
    args = p.parse_args()

    better = (lambda a, b: a > b + args.min_delta) if args.goal == "max" \
        else (lambda a, b: a < b - args.min_delta)
    worst = float("-inf") if args.goal == "max" else float("inf")

    t_start = time.time()
    seen = set()
    best = worst
    best_seq = None
    since_improve = 0
    print(f"watchdog: watching {args.metric} (goal={args.goal}); patience={args.patience}, "
          f"min_delta={args.min_delta}, ceilings: {args.max_experiments} exp / "
          f"{args.max_wall_hours}h", flush=True)

    while True:
        if os.path.exists(STOP_MARKER):
            print("watchdog: STOP marker already present, exiting", flush=True)
            return
        records = [r for r in read_index() if r.get("status") != "crash"]
        # process any newly-recorded experiments in seq order
        for r in sorted(records, key=lambda x: x["seq"]):
            if r["seq"] in seen:
                continue
            seen.add(r["seq"])
            val = r.get(args.metric, 0.0)
            if best_seq is None or better(val, best):
                best, best_seq, since_improve = val, r["seq"], 0
            else:
                since_improve += 1

        total = len(seen)
        # hard ceilings
        wall_h = (time.time() - t_start) / 3600
        if total >= args.max_experiments:
            trip("hard ceiling: max experiments reached",
                 {"experiments": total, "best": best, "best_seq": best_seq}, args)
            return
        if wall_h >= args.max_wall_hours:
            trip("hard ceiling: max wall-clock reached",
                 {"wall_hours": round(wall_h, 2), "best": best, "best_seq": best_seq}, args)
            return
        # convergence
        if best_seq is not None and since_improve >= args.patience:
            trip(f"converged: no >{args.min_delta} improvement in {since_improve} experiments",
                 {"best": best, "best_seq": best_seq, "since_improve": since_improve,
                  "total_experiments": total}, args)
            return

        if total:
            print(f"\rwatchdog: {total} exp · best {args.metric}={best:.4f} (#{best_seq}) · "
                  f"{since_improve}/{args.patience} since improve · {wall_h:.1f}h    ",
                  end="", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
