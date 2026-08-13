"""
Measure a trained Othello model's playing strength (runs on the GPU pod).

Loads a run's checkpoint, builds the model into an arena move-function, and
plays it against the weak->strong opponent ladder (random, greedy, alpha-beta at
increasing depth). Reports per-opponent score, implied Elo gap, and the strongest
rung it still beats. This is the real strength metric the earlier objectives
(next-move prediction, complete-game rate) never measured.

Usage (on the pod):
    python gauntlet_run.py --run-dir telemetry/runs/run_016_80f660f \
        --games 60 --max-depth 5 --out gauntlet_016.json
"""

import argparse
import json
import os

import torch

from prepare import Tokenizer
from othello.modeleval import run_gauntlet


def load_model(run_dir):
    """Reconstruct the model class from the run's train_snapshot.py (definitions
    only) and load its checkpoint. Returns (model, config)."""
    snapshot = open(os.path.join(run_dir, "train_snapshot.py")).read()
    marker = "\nt_start = time.time()"
    assert marker in snapshot, "could not locate setup boundary in snapshot"
    ns = {}
    exec(compile(snapshot.split(marker)[0], "<snapshot_defs>", "exec"), ns)
    GPT, GPTConfig = ns["GPT"], ns["GPTConfig"]

    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cuda")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).cuda()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Match init_weights(): keep token + value embeddings in bf16 so all attention
    # inputs share a dtype under autocast (we loaded weights, not re-init).
    model.transformer.wte.to(dtype=torch.bfloat16)
    for ve in model.value_embeds.values():
        ve.to(dtype=torch.bfloat16)
    model.eval()
    return model, config


def main():
    ap = argparse.ArgumentParser(description="Measure Othello model playing strength")
    ap.add_argument("--run-dir", required=True, help="folder with train_snapshot.py + checkpoint.pt")
    ap.add_argument("--games", type=int, default=60, help="games per opponent (alternating colors)")
    ap.add_argument("--max-depth", type=int, default=5, help="deepest alpha-beta opponent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    model, config = load_model(args.run_dir)
    tokenizer = Tokenizer.from_directory()

    def model_fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x)

    res = run_gauntlet(model_fn, tokenizer, n_games=args.games,
                       max_depth=args.max_depth, seed=args.seed)

    print(f"\nStrength gauntlet for {os.path.basename(args.run_dir.rstrip('/'))} "
          f"({args.games} games/opponent):")
    print(f"  beats up to: {res['beats_up_to']}")
    for name, r in res["per_opponent"].items():
        print(f"  vs {name:10s}  score={r['score']:.2f}  "
              f"W/D/L={r['wins']}/{r['draws']}/{r['losses']}  eloGap={r['elo_gap']:+.0f}")

    if args.out:
        payload = {
            "run_dir": args.run_dir,
            "games_per_opponent": args.games,
            "max_depth": args.max_depth,
            "config": ckpt_config_dict(config),
            **res,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


def ckpt_config_dict(config):
    """Best-effort dict of the model config for the JSON record."""
    d = getattr(config, "__dict__", None)
    if d:
        return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
    return {}


if __name__ == "__main__":
    main()
