"""
Telemetry for autoresearch experiments (read-only for the research agent).

Records everything needed to reconstruct and replay a run after the fact:
  - meta.json      run metadata (config, params, start time; merged with the
                   wrapper-written fields like commit and diff)
  - steps.jsonl    one line per optimizer step (loss, lr, throughput, ...)
  - samples.jsonl  free-running games sampled at ~10% progress milestones
  - result.json    final metrics: val_bpb, legality, vs-random winrate, ...
  - checkpoint.pt  final model weights + config

train.py MUST keep calling the four hooks (init_run, log_step, maybe_sample,
finalize) and MUST keep the model callable as `model(idx) -> logits`. Runs
with broken or missing telemetry are invalid regardless of val_bpb.

The run directory comes from $AUTORESEARCH_RUN_DIR (set by run_experiment.py);
ad-hoc runs fall back to telemetry/runs/adhoc-<timestamp>/.
"""

import json
import os
import time

import torch

from othello.modeleval import sample_games, play_vs_random, evaluate_strength

# Sampling configuration (fixed)
MILESTONE_EVERY = 0.1     # sample games each time progress crosses a 10% boundary
SAMPLE_GAMES = 16         # free-running games per milestone
VS_RANDOM_GAMES = 20      # legality-masked games vs random opponent at finalize
FLUSH_EVERY = 50          # steps.jsonl flush cadence

_state = {
    "run_dir": None,
    "steps_file": None,
    "steps_since_flush": 0,
    "last_milestone": -1,
    "t_init": None,
}


def _run_dir():
    if _state["run_dir"] is None:
        run_dir = os.environ.get("AUTORESEARCH_RUN_DIR")
        if not run_dir:
            run_dir = os.path.join("telemetry", "runs", f"adhoc-{int(time.time())}")
        os.makedirs(run_dir, exist_ok=True)
        _state["run_dir"] = run_dir
    return _state["run_dir"]


def _model_fn(model, tokenizer):
    """Wrap the (possibly compiled) model as an eval-mode eager callable.
    Uses _orig_mod to sidestep torch.compile shape recompilation."""
    eager = getattr(model, "_orig_mod", model)
    device = next(eager.parameters()).device

    def fn(idx):
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                return eager(idx)
        return eager(idx)

    return fn, device


def init_run(config=None, num_params=None, flops_per_token=None):
    """Call once after model construction, before the training loop."""
    run_dir = _run_dir()
    _state["t_init"] = time.time()
    meta_path = os.path.join(run_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):  # wrapper wrote commit/diff/description already
        with open(meta_path) as f:
            meta = json.load(f)
    meta.update({
        "train_started_at": _state["t_init"],
        "config": config,
        "num_params": num_params,
        "flops_per_token": flops_per_token,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    _state["steps_file"] = open(os.path.join(run_dir, "steps.jsonl"), "a")


def log_step(step, progress, loss, lrm, dt, tok_per_sec, mfu, epoch):
    """Call once per optimizer step."""
    f = _state["steps_file"]
    if f is None:
        return
    f.write(json.dumps({
        "step": step,
        "t": round(time.time() - _state["t_init"], 3),
        "progress": round(progress, 5),
        "loss": round(loss, 6),
        "lrm": round(lrm, 4),
        "dt_ms": round(dt * 1000, 1),
        "tok_per_sec": tok_per_sec,
        "mfu": round(mfu, 2),
        "epoch": epoch,
    }) + "\n")
    _state["steps_since_flush"] += 1
    if _state["steps_since_flush"] >= FLUSH_EVERY:
        f.flush()
        _state["steps_since_flush"] = 0


def maybe_sample(model, tokenizer, progress, step):
    """Call once per optimizer step (after log_step). Samples free-running
    games whenever progress crosses a milestone boundary. Cheap: runs the
    eager model on tiny batches, well under a second per milestone."""
    milestone = int(progress / MILESTONE_EVERY)
    if milestone <= _state["last_milestone"]:
        return
    _state["last_milestone"] = milestone
    model_fn, device = _model_fn(model, tokenizer)
    was_training = getattr(model, "training", True)
    model.eval()
    try:
        games, stats = sample_games(
            model_fn, tokenizer, n_games=SAMPLE_GAMES,
            seed=1000 + milestone, device=device,
        )
    finally:
        if was_training:
            model.train()
    with open(os.path.join(_run_dir(), "samples.jsonl"), "a") as f:
        f.write(json.dumps({
            "milestone": milestone,
            "progress": round(progress, 5),
            "step": step,
            "t": round(time.time() - _state["t_init"], 3),
            "stats": stats,
            "games": games,
        }) + "\n")


def finalize(model, tokenizer, val_bpb, metrics=None):
    """Call once after the final evaluation. Runs the Othello strength eval,
    saves the checkpoint, and writes result.json."""
    run_dir = _run_dir()
    if _state["steps_file"] is not None:
        _state["steps_file"].flush()
        _state["steps_file"].close()
        _state["steps_file"] = None

    model_fn, device = _model_fn(model, tokenizer)
    model.eval()
    games, sample_stats = sample_games(
        model_fn, tokenizer, n_games=SAMPLE_GAMES * 2, seed=9999, device=device)
    vs_random = play_vs_random(
        model_fn, tokenizer, n_games=VS_RANDOM_GAMES, seed=7, device=device)

    eager = getattr(model, "_orig_mod", model)

    # Night 3: sacred playing-strength metric. Computed HERE (read-only
    # instrumentation) so no experiment can skip or alter it. Uses the eager
    # model to avoid recompilation across the varied sequence lengths of full
    # games. This is the run's optimization target (see program.md).
    strength, strength_by_opponent = evaluate_strength(eager, tokenizer, device=device)
    print(f"strength:         {strength:.6f}")          # parsed by run_experiment
    print(f"strength_detail:  {strength_by_opponent}")  # human-readable breakdown

    config = None
    if hasattr(eager, "config"):
        from dataclasses import asdict, is_dataclass
        config = asdict(eager.config) if is_dataclass(eager.config) else None
    torch.save({"model_state_dict": eager.state_dict(), "config": config},
               os.path.join(run_dir, "checkpoint.pt"))

    result = {
        "val_bpb": val_bpb,
        "strength": strength,
        "strength_by_opponent": strength_by_opponent,
        "finished_at": time.time(),
        "final_sample_stats": sample_stats,
        "final_sample_games": games,
        "vs_random": vs_random,
        "metrics": metrics or {},
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"telemetry: run artifacts written to {run_dir}")
