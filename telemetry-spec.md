# Telemetry spec

Everything the playback website needs to reconstruct an overnight run lives
under `telemetry/` (untracked by git):

```
telemetry/
  night.jsonl              # append-only index: one line per experiment
  runs/
    run_001_a1b2c3d/
      meta.json            # metadata (wrapper + train-time, merged)
      train_snapshot.py    # exact train.py used for this run
      train.log            # full training stdout/stderr
      steps.jsonl          # one line per optimizer step
      samples.jsonl        # free-running games at ~10% progress milestones
      result.json          # final metrics + Othello evals
      checkpoint.pt        # final weights (pruned when status=discard)
```

All timestamps are Unix seconds. `t` fields are seconds since `init_run`.

## night.jsonl — one JSON object per line

| field | type | meaning |
|---|---|---|
| run_id | str | `run_<seq>_<commit7>` |
| seq | int | 1-based experiment number in this night |
| started_at | float | wall-clock start |
| wall_seconds | float | total wrapper-observed duration |
| commit | str | short git hash of the train.py state |
| description | str | agent's `--desc` for the experiment |
| status | str | `pending` → `keep` / `discard` / `crash` |
| val_bpb | float | 0.0 for crashes |
| peak_vram_mb | float | 0.0 for crashes |
| timed_out | bool | killed at the 10-minute limit |
| exit_code | int | train process exit code |

## meta.json

Wrapper-written: `run_id`, `seq`, `description`, `commit`, `branch`,
`commit_subject`, `started_at`, `train_py_diff` (unified diff vs parent
commit). Train-written (merged in by `telemetry.init_run`):
`train_started_at`, `config` (GPTConfig as dict), `num_params`,
`flops_per_token`.

## steps.jsonl — one JSON object per optimizer step

`step`, `t`, `progress` (0–1 fraction of the 5-minute budget), `loss`
(EMA-debiased train loss), `lrm` (LR multiplier), `dt_ms`, `tok_per_sec`,
`mfu`, `epoch`.

## samples.jsonl — one JSON object per milestone (~11 per run)

- `milestone` (int, progress // 10%), `progress`, `step`, `t`
- `stats`: `n_games`, `legality_rate` (fraction of sampled moves that are
  legal), `complete_rate` (fraction of games that are full legal games),
  `mean_legal_prefix` (avg #moves before the first illegal one)
- `games[]`: each with `moves` (transcript text), `total_moves`,
  `legal_moves`, `is_complete`, `black`, `white`, `first_illegal`

Sampling is unconstrained (temperature 1.0 from BOS) — illegal moves are
recorded, not masked. That is the improvement signal the website replays.

## result.json

- `val_bpb` — the ground-truth metric
- `final_sample_stats` / `final_sample_games` — same shape as samples.jsonl,
  32 games, sampled after training finished
- `vs_random` — legality-masked greedy model vs uniform-random opponent:
  `n_games`, `wins`, `draws`, `losses`, `winrate`, and `games[]` with
  `model_color`, `moves`, `model_score`, `opp_score`
- `metrics` — `training_seconds`, `total_seconds`, `peak_vram_mb`,
  `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth`

## Replay semantics for the website

- The **night timeline** is night.jsonl ordered by `seq`; best-so-far val_bpb
  is the running min over `keep` runs.
- The **within-run story** is steps.jsonl (loss curve) + samples.jsonl
  (games getting more legal as progress increases).
- Any game's board states are recomputed client-side from `moves` with the
  same rules as `othello/engine.py`; `first_illegal` marks where playback
  should flag the illegal move.
