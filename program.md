# autoresearch — Night 2: learning to *play* Othello

This is an experiment to have the LLM do its own research. This run's objective
is **complete_game_rate** — teaching a small GPT to play fully legal Othello
games — not the val_bpb compression objective of the first run. (Night 1's
`program.md` is preserved in git history.)

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it via the telemetry wrapper:

```
uv run run_experiment.py --desc "short description of the idea" > run.log 2>&1
```

The wrapper relays all training output, so `run.log` works exactly as before.

**The data is Othello games**: each document is one complete self-play game as
space-separated moves ("d3 c5 f6 ..."). The tokenizer is fixed at one token
per move. The model is learning the rules and strategy of Othello from game
transcripts alone.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `telemetry.py`, `run_experiment.py`, or anything in `othello/`. These are read-only instrumentation and rules-engine code.
- Remove or break the telemetry hooks in `train.py`. The calls to `telemetry.init_run`, `telemetry.log_step`, `telemetry.maybe_sample`, and `telemetry.finalize` must remain in place and functional, and the model must remain callable as `model(idx) -> logits` for a (B, T) tensor of token ids. **A run with broken or missing telemetry is invalid regardless of its score** — treat it as a crash, fix, and re-run.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` and `evaluate_complete_game_rate` functions in `prepare.py` are the ground truth metrics. In particular you may not change the sampling protocol (temperature, seed, number of games) used by the complete-game-rate eval, nor call it yourself with different settings.

**The goal: get the highest `complete_game_rate`.** This is the fraction of games the model plays — sampled freely from the opening, at a fixed temperature — that are *fully legal Othello games from start to a finished board*. Higher is better; 1.0 would mean every sampled game is a complete, legal game. This measures whether the model has actually learned to *play* Othello, not just to compress transcripts. Since the time budget is fixed (5 minutes), you don't need to worry about training time. Everything else is fair game: architecture, optimizer, hyperparameters, batch size, model size, and — very much in scope for this objective — how the model is trained to generate (e.g. addressing the gap between teacher-forced training and free-running play). The only hard constraint is that the code runs without crashing and finishes within the time budget.

`val_bpb` is still printed every run for continuity with earlier work, but it is **not** your objective this time — do not optimize it. If a change lowers val_bpb but also lowers complete_game_rate, discard it; complete_game_rate is what you keep or discard on.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it; removing something and getting equal or better results is a great outcome. Weigh complexity cost against improvement magnitude. Because complete_game_rate is measured over sampled games it is noisier than a smooth loss — treat a change as a real improvement only if it beats the current best by a clear margin (roughly ≥ 0.01, i.e. one more complete game in a hundred is noise, several are not), not by a hair.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
complete_game_rate: 0.687500
legal_move_rate:  0.907000
val_bpb:          0.731200
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

`complete_game_rate` is your objective (higher is better). `legal_move_rate`
(fraction of individual sampled moves that are legal) is a useful secondary
signal — it usually moves before complete_game_rate does, so it can tell you a
change is helping even when complete games are still rare. Note the script
always stops after 5 minutes, so absolute numbers depend on the platform. You
can extract the key metrics from the log file:

```
grep "^complete_game_rate:\|^legal_move_rate:\|^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	complete_game_rate	legal_move_rate	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. complete_game_rate achieved (e.g. 0.687500) — use 0.000000 for crashes
3. legal_move_rate achieved (e.g. 0.907000) — use 0.000000 for crashes
4. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	complete_game_rate	legal_move_rate	memory_gb	status	description
a1b2c3d	0.000000	0.210000	44.0	keep	baseline
b2c3d4e	0.250000	0.560000	44.2	keep	deeper, narrower model
c3d4e5f	0.180000	0.520000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run run_experiment.py --desc "what this tries" > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^complete_game_rate:\|^legal_move_rate:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If complete_game_rate improved (higher, by a clear margin): `uv run run_experiment.py --record-status keep`, and you "advance" the branch, keeping the git commit
9. If complete_game_rate is equal or worse: `uv run run_experiment.py --record-status discard`, and you git reset back to where you started
10. If the run crashed: `uv run run_experiment.py --record-status crash`

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
