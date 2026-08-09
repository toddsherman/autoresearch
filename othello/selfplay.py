"""
Othello self-play data generation.

Generates complete games with a mix of policies (pure random through
mostly-greedy positional play) for distributional diversity, and writes
them as parquet shards with a single `text` column — one game transcript
per row — matching the schema prepare.py's dataloader expects.

Usage:
    python -m othello.selfplay --out-dir DATA_DIR --num-shards 8 --games-per-shard 65536
"""

import argparse
import os
import random
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from othello.engine import Game, bb_to_moves, moves_to_text

# Classic positional weight table (corners prized, X/C squares penalized).
# Row 0 = rank 1. Symmetric top/bottom.
_W = [
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
]

# Policy mix: (epsilon, fraction of games). epsilon = probability of a
# uniformly random legal move instead of the greedy positional move.
DEFAULT_MIX = [
    (1.0, 0.30),   # pure random
    (0.5, 0.20),
    (0.25, 0.20),
    (0.1, 0.20),
    (0.05, 0.10),  # near-greedy
]


def _pick_epsilon(rng, mix):
    r = rng.random()
    acc = 0.0
    for eps, frac in mix:
        acc += frac
        if r < acc:
            return eps
    return mix[-1][0]


def play_one_game(seed, mix=None):
    """Play one full self-play game. Returns the transcript string."""
    rng = random.Random(seed)
    eps = _pick_epsilon(rng, mix or DEFAULT_MIX)
    game = Game()
    while not game.over:
        moves = bb_to_moves(game.legal_moves())
        if rng.random() < eps:
            sq = moves[rng.randrange(len(moves))]
        else:
            best = max(_W[m] for m in moves)
            candidates = [m for m in moves if _W[m] == best]
            sq = candidates[rng.randrange(len(candidates))]
        game.play(sq)
    return moves_to_text(game.moves)


def _generate_shard(args):
    shard_idx, out_dir, games_per_shard, base_seed = args
    filepath = os.path.join(out_dir, f"shard_{shard_idx:05d}.parquet")
    if os.path.exists(filepath):
        return filepath, 0
    t0 = time.time()
    seed0 = base_seed + shard_idx * games_per_shard
    texts = [play_one_game(seed0 + i) for i in range(games_per_shard)]
    table = pa.table({"text": texts})
    tmp = filepath + ".tmp"
    pq.write_table(table, tmp, row_group_size=8192)
    os.rename(tmp, filepath)
    return filepath, time.time() - t0


def generate_shards(out_dir, num_shards, games_per_shard, base_seed=1234, workers=None):
    """Generate parquet shards of self-play games. Deterministic in base_seed.
    Skips shards that already exist."""
    os.makedirs(out_dir, exist_ok=True)
    # Cap: cloud containers often report the host's core count (e.g. 224)
    # while the cgroup quota is far smaller.
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
    jobs = [(i, out_dir, games_per_shard, base_seed) for i in range(num_shards)]
    with Pool(processes=min(workers, num_shards)) as pool:
        for filepath, dt in pool.imap_unordered(_generate_shard, jobs):
            status = f"generated in {dt:.1f}s" if dt else "already exists"
            print(f"  {os.path.basename(filepath)}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Othello self-play parquet shards")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--games-per-shard", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    t0 = time.time()
    generate_shards(args.out_dir, args.num_shards, args.games_per_shard, args.seed, args.workers)
    print(f"Done in {time.time() - t0:.1f}s")
