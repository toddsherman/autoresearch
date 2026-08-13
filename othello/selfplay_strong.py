"""
Strong self-play data generation (alpha-beta policy).

Same parquet schema as othello/selfplay.py (one `text` transcript per row), so
it is a drop-in replacement for the training corpus in prepare.py. The only
difference is the move policy: instead of a one-ply greedy heuristic (which
barely beats random), moves come from the alpha-beta searcher in othello.search,
which actually looks ahead. A model trained to predict these games imitates
strong play, not weak play.

Diversity comes from (a) a few random opening plies and (b) a small epsilon of
random midgame moves, so games are not all identical.

Usage:
    python -m othello.selfplay_strong --out-dir DATA_DIR --num-shards 8 \
        --games-per-shard 8192 --depth 4 --opening-random 4 --epsilon 0.08
"""

import argparse
import os
import random
import time
from multiprocessing import Pool

from othello.engine import Game, bb_to_moves, moves_to_text
from othello.search import best_move

# Defaults chosen for a strong-but-affordable corpus (see search.py timings):
# depth 4 is ~0.5s/game single-core and crushes the greedy policy.
#
# depth_mix (optional) is a list of (depth, weight) pairs; each game picks a
# depth from it. Mixing depths + heavier opening randomization gives a DIVERSE
# corpus (varied positions and targets) while every move stays strong, which is
# what the night-3 strength objective needs. Both moves come from the searcher,
# so there are no weak targets (unlike search-vs-weak games).
DEFAULTS = dict(depth=4, exact_empties=8, opening_random=4, epsilon=0.08, depth_mix=None)


def _pick_depth(rng, depth, depth_mix):
    if not depth_mix:
        return depth
    total = sum(w for _, w in depth_mix)
    r = rng.random() * total
    acc = 0.0
    for d, w in depth_mix:
        acc += w
        if r < acc:
            return d
    return depth_mix[-1][0]


def play_one_game(seed, depth, exact_empties, opening_random, epsilon, depth_mix=None):
    """Play one full game with the search policy. Returns the transcript string."""
    rng = random.Random(seed)
    d = _pick_depth(rng, depth, depth_mix)
    game = Game()
    ply = 0
    while not game.over:
        if ply < opening_random:
            moves = bb_to_moves(game.legal_moves())
            sq = moves[rng.randrange(len(moves))]
        else:
            p = game.boards[game.player]
            o = game.boards[1 - game.player]
            sq = best_move(p, o, d, exact_empties, rng, epsilon)
        game.play(sq)
        ply += 1
    return moves_to_text(game.moves)


def _generate_shard(args):
    shard_idx, out_dir, games_per_shard, base_seed, cfg = args
    import pyarrow as pa                      # lazy: keep module importable w/o pyarrow
    import pyarrow.parquet as pq
    filepath = os.path.join(out_dir, f"shard_{shard_idx:05d}.parquet")
    if os.path.exists(filepath):
        return filepath, 0
    t0 = time.time()
    seed0 = base_seed + shard_idx * games_per_shard
    texts = [play_one_game(seed0 + i, **cfg) for i in range(games_per_shard)]
    table = pa.table({"text": texts})
    tmp = filepath + ".tmp"
    pq.write_table(table, tmp, row_group_size=8192)
    os.rename(tmp, filepath)
    return filepath, time.time() - t0


def generate_shards(out_dir, num_shards, games_per_shard, base_seed=1234,
                    workers=None, cfg=None):
    """Generate parquet shards of strong self-play games. Deterministic in
    base_seed; skips shards that already exist."""
    cfg = {**DEFAULTS, **(cfg or {})}
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
    jobs = [(i, out_dir, games_per_shard, base_seed, cfg) for i in range(num_shards)]
    with Pool(processes=min(workers, num_shards)) as pool:
        for filepath, dt in pool.imap_unordered(_generate_shard, jobs):
            status = f"generated in {dt:.1f}s" if dt else "already exists"
            print(f"  {os.path.basename(filepath)}: {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate strong (alpha-beta) Othello self-play shards")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--games-per-shard", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    ap.add_argument("--exact-empties", type=int, default=DEFAULTS["exact_empties"])
    ap.add_argument("--opening-random", type=int, default=DEFAULTS["opening_random"])
    ap.add_argument("--epsilon", type=float, default=DEFAULTS["epsilon"])
    ap.add_argument("--depth-mix", type=str, default=None,
                    help='weighted depth mix, e.g. "3:2,2:1,4:1" (overrides --depth per game)')
    args = ap.parse_args()
    depth_mix = None
    if args.depth_mix:
        depth_mix = [(int(d), float(w)) for d, w in
                     (pair.split(":") for pair in args.depth_mix.split(","))]
    cfg = dict(depth=args.depth, exact_empties=args.exact_empties,
               opening_random=args.opening_random, epsilon=args.epsilon,
               depth_mix=depth_mix)
    t0 = time.time()
    generate_shards(args.out_dir, args.num_shards, args.games_per_shard,
                    args.seed, args.workers, cfg)
    print(f"Done in {time.time() - t0:.1f}s")
