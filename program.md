# autoresearch — Night 3: learning to *win* at Othello

This is an experiment to have the LLM do its own research. This run's objective
is **playing strength** — teaching a small GPT to actually *win* games of
Othello — not night 1's `val_bpb` (compression) or night 2's
`complete_game_rate` (legality). Earlier `program.md` versions are preserved in
git history. (At launch this file becomes `program.md`.)

## Setup

1. **Agree on a run tag** and create branch `autoresearch/<tag>` from the current
   branch (the night-3 baseline you are checked out on — NOT `master`, which is the
   upstream base without any Othello code).
2. **Read the in-scope files**: `README.md`, `prepare.py` (fixed constants, data,
   tokenizer, eval — do not modify), `train.py` (the file you modify).
3. **Verify data exists**: `~/.cache/autoresearch/` must contain the data shards
   (a large, diverse *strong* corpus) and the tokenizer. If not, ask the human.
4. **Initialize `results.tsv`** with just the header row.
5. **Confirm and go**, then kick off experimentation.

## Experimentation

Each experiment runs on a single GPU for a **fixed time budget** (wall-clock
training time), launched via the telemetry wrapper:

```
uv run run_experiment.py --desc "short description of the idea" > run.log 2>&1
```

**The data is strong Othello games**: each document is one complete game as
space-separated moves ("d3 c5 f6 ..."), one token per move. Unlike earlier runs,
these games are played by an alpha-beta *searcher* (mixed depths, with opening
and light midgame randomization for positional variety), so **every move in the
data is a strong move** — the model is imitating good play, not weak play.

**The goal: get the highest `strength` score.** `strength` is the model's mean
score against a fixed weak-to-moderate opponent ladder — **random, greedy,
search@1, search@2** — 16 games per opponent at a fixed seed, colors
alternating, where score = (wins + 0.5·draws) / games. It is computed by
`othello.modeleval.evaluate_strength` and reported every run. 0.5 vs an opponent
is an even match; higher `strength` means the model beats stronger opponents and
plays well from the varied positions those opponents create. This measures
whether the model can actually *win*, not just play legally. (The full gauntlet,
including search@3–5, is run on the final champion only.)

**What you CAN do:**
- Modify `train.py` — the only file you edit. Architecture, optimizer,
  hyperparameters, training loop, batch size, model size are all fair game.
- **Shape how the model learns from the data (this is explicitly in scope).**
  You may augment and re-sample the loaded games inside `train.py`: e.g. Othello
  is symmetric, so **dihedral (8×) augmentation** of the transcripts is allowed
  and was a real win on night 1; curriculum / re-weighting / sampling-temperature
  choices over the corpus are also fair game. The base corpus is fixed and
  diverse; how you exploit it is yours to optimize.

**What you CANNOT do:**
- Modify `prepare.py`, `telemetry.py`, `run_experiment.py`, or anything in
  `othello/` (read-only instrumentation and rules/eval code).
- Change the strength metric. `othello.modeleval.evaluate_strength` and its fixed
  ladder, game count and seed are the ground-truth metric — do not reimplement it,
  call it with different settings, or change which opponents/how many games. A run
  that does not report the sacred `strength` is invalid.
- Change or regenerate the data files on disk. (You may transform the games in
  memory inside `train.py` per the augmentation/sampling allowance above, but the
  parquet shards themselves are fixed.)
- Remove or break the telemetry hooks in `train.py` (`telemetry.init_run`,
  `log_step`, `maybe_sample`, `finalize`); the model must stay callable as
  `model(idx) -> logits`. A run with broken telemetry is invalid — treat as crash.
- Install new packages.

**VRAM** is a soft constraint; some growth is fine for real gains.

**Simplicity criterion**: all else equal, simpler is better; removing something
for equal-or-better strength is a great outcome.

**Noise**: `strength` is measured over sampled games and is **noisier** than a
smooth loss (win-rate variance over 64 games ≈ a few points). Treat a change as a
real improvement only if it beats the current best by a clear margin (roughly
**≥ 0.02**), not by a hair. The convergence watchdog is configured accordingly
(loosened min-delta, longer patience).

**The first run** establishes the baseline: run the training script as-is.

## Output format

After each run, append one row to `results.tsv` and tell the human the `strength`
score, the per-opponent breakdown, and whether you keep or discard the change.
