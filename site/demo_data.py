"""
Generate synthetic-but-realistic overnight telemetry for developing the
playback site before any GPU run exists. Games are produced with the real
rules engine, so board playback, legality stats, and transcripts are all
internally consistent.

Usage: python site/demo_data.py   (writes site/data.js)
"""

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from othello.engine import Game, bb_to_moves, move_to_name, verify_transcript

GAMES_PER_MILESTONE = 3    # shipped to the site for playback
STAT_GAMES = 12            # sampled per milestone for stable legality stats
NIGHT_START = 1786240800  # ~23:00 local, arbitrary but fixed


def model_game(rng, p_legal):
    """Simulate a free-running sample from a model whose per-move legality is
    ~p_legal: legal random moves until a failure, then a short junk tail."""
    game = Game()
    tokens = []
    while not game.over and len(tokens) < 60:
        if rng.random() < p_legal:
            sq = rng.choice(bb_to_moves(game.legal_moves()))
            tokens.append(move_to_name(sq))
            game.play(sq)
        else:
            legal = set(bb_to_moves(game.legal_moves()))
            while True:
                sq = rng.randrange(64)
                if sq not in legal:
                    break
            tokens.append(move_to_name(sq))
            for _ in range(rng.randint(2, 6)):
                tokens.append(move_to_name(rng.randrange(64)))
            break
    text = " ".join(tokens)
    return {"moves": text, **verify_transcript(text)}


EXPERIMENTS = [
    # (description, delta_direction) — delta_direction: -1 improves, +1 regresses, 0 crash
    ("baseline", -1),
    ("raise matrix LR 0.04 -> 0.06", -1),
    ("depth 8 -> 10", 1),
    ("depth 8 -> 6, keep width", -1),
    ("double batch size to 2^20", 1),
    ("warmdown ratio 0.5 -> 0.7", -1),
    ("remove value embeddings", 1),
    ("window pattern SSSL -> SSLL", 1),
    ("head dim 128 -> 64", 1),
    ("raise embedding LR to 0.9", -1),
    ("width x1.5 at depth 6", 0),
    ("width x1.25 at depth 6", -1),
    ("relu^2 -> gelu in MLP", 1),
    ("adam betas (0.8,0.95) -> (0.9,0.97)", 1),
    ("matrix LR 0.06 -> 0.08", 1),
    ("matrix LR 0.06 -> 0.07", -1),
    ("weight decay 0.2 -> 0.1", -1),
    ("logit softcap 15 -> 30", 1),
    ("rotary base 10000 -> 40000", 1),
    ("seq len chunking: train at T=512", -1),
    ("T=512 + batch rebalance", -1),
    ("drop x0 residual lambdas", 1),
    ("scalar LR 0.5 -> 1.0", 1),
    ("muon ns_steps 5 -> 3", 0),
    ("muon momentum warmup 300 -> 100 steps", -1),
    ("final LR frac 0.0 -> 0.05", 1),
    ("embedding LR 0.9 -> 1.2", 1),
    ("combine: T=512 + wd 0.1 + mom warmup 100", -1),
]


def synth_diff(desc):
    return (f"--- a/train.py\n+++ b/train.py\n@@ hyperparameters @@\n"
            f"-# (previous setting)\n+# {desc}\n")


def main():
    rng = random.Random(20260809)
    runs = []
    t = NIGHT_START
    best = 1.02  # baseline-ish val_bpb for random-ish Othello transcripts
    val = best
    n_kept = 0
    total_kept = sum(1 for _, d in EXPERIMENTS if d == -1)

    for seq, (desc, direction) in enumerate(EXPERIMENTS, start=1):
        t += rng.uniform(40, 150)  # agent thinking/editing time
        startup = rng.uniform(18, 35)
        crashed = direction == 0
        wall = startup + (rng.uniform(20, 90) if crashed else 300 + rng.uniform(8, 20))

        if crashed:
            status, val_bpb = "crash", 0.0
        elif direction == -1:
            n_kept += 1
            # diminishing returns toward ~0.78
            val_bpb = best - (best - 0.78) * 0.35 - rng.uniform(0, 0.004)
            val_bpb = max(val_bpb, best - 0.06)
            status = "keep"
        else:
            val_bpb = best + rng.uniform(0.002, 0.05)
            status = "discard"

        run = {
            "run_id": f"run_{seq:03d}_{rng.getrandbits(28):07x}",
            "seq": seq,
            "commit": f"{rng.getrandbits(28):07x}",
            "description": desc,
            "status": status,
            "val_bpb": round(val_bpb, 6),
            "peak_vram_mb": 0.0 if crashed else round(rng.uniform(38000, 52000), 1),
            "started_at": round(t, 1),
            "wall_seconds": round(wall, 1),
            "startup_seconds": round(startup, 1),
            "num_params": rng.choice([42, 50, 61]) * 10**6,
            "diff": "" if seq == 1 else synth_diff(desc),
            "loss_curve": [],
            "milestones": [],
            "final": None,
        }

        if not crashed:
            # loss curve: exp decay toward a floor tied to val_bpb quality
            floor = 0.45 + (val_bpb - 0.78) * 2.2
            n_pts = 120
            for i in range(n_pts + 1):
                p = i / n_pts
                loss = floor + (5.96 - floor) * math.exp(-p * rng.uniform(23, 27)) \
                       + 0.12 * math.exp(-p * 3.2) + rng.gauss(0, 0.006)
                run["loss_curve"].append([round(p, 4), round(max(loss, floor), 4)])

            # milestones: legality improves within the run; better runs go higher
            quality = (1.02 - val_bpb) / (1.02 - 0.78)  # 0..1 across the night
            p_final = 0.30 + 0.69 * quality
            for m in range(11):
                frac = m / 10
                p_legal = 0.02 + (p_final - 0.02) * (frac ** 0.5)
                games = [model_game(rng, p_legal) for _ in range(STAT_GAMES)]
                total = sum(g["total_moves"] for g in games) or 1
                legal = sum(g["legal_moves"] for g in games)
                run["milestones"].append({
                    "milestone": m,
                    "progress": frac,
                    "step": int(frac * 950),
                    "legality_rate": round(legal / total, 4),
                    "complete_rate": round(sum(g["is_complete"] for g in games) / len(games), 3),
                    "mean_legal_prefix": round(legal / len(games), 1),
                    "games": games[:GAMES_PER_MILESTONE],
                })
            last = run["milestones"][-1]
            run["final"] = {
                "legality_rate": last["legality_rate"],
                "complete_rate": last["complete_rate"],
                "winrate": round(min(0.95, 0.45 + 0.5 * quality + rng.uniform(-0.05, 0.05)), 3),
                "vs_random": {"n_games": 20},
            }

        if status == "keep":
            best = val_bpb
        t += wall
        runs.append(run)

    bundle = {
        "meta": {
            "title": "autoresearch: othello",
            "subtitle": "an agent teaching a GPT the game of Othello, 5 minutes at a time",
            "hardware": "1x H100 (synthetic demo data)",
            "branch": "autoresearch/demo",
            "generated_at": time.time(),
            "is_demo": True,
        },
        "runs": runs,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.js")
    with open(out, "w") as f:
        f.write("window.NIGHT_DATA = ")
        json.dump(bundle, f, separators=(",", ":"))
        f.write(";\n")
    kept = [r for r in runs if r["status"] == "keep"]
    print(f"Wrote {out} ({os.path.getsize(out) // 1024} KB): "
          f"{len(runs)} runs, {len(kept)} kept, best val_bpb {min(r['val_bpb'] for r in kept):.4f}")


if __name__ == "__main__":
    main()
