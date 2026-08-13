"""
Alpha-beta Othello search: a strong opponent and a strong-data policy.

Negamax + alpha-beta over raw bitboards, with a positional + mobility heuristic
and exact endgame solving for the last few empties. This is far stronger than
the one-ply greedy positional policy used to generate the original training data
(othello/selfplay.py), because it actually looks ahead.

It is used three ways:
  - as an opponent ladder in the strength gauntlet (search at increasing depth),
  - as the move policy for strong-data self-play generation (selfplay_strong.py),
  - later, as the test-time search wrapper around the model's own policy/value.

Everything here is pure Python (only depends on othello.engine), so it runs
anywhere without a GPU.
"""

import random

from othello.engine import legal_moves_bb, apply_move, bb_to_moves

# Classic positional weight table (mirrors the one in othello/selfplay.py):
# corners prized, the X/C squares next to them penalized. Index = row*8 + col.
POS_W = [
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
]

INF = 1 << 30
MOBILITY_WEIGHT = 8      # value of one extra legal move, in positional points
TERMINAL_SCALE = 1000    # a final disc lead dominates any heuristic score


def _positional(bb):
    """Sum of POS_W over the set squares of a bitboard."""
    s = 0
    while bb:
        lsb = bb & -bb
        s += POS_W[lsb.bit_length() - 1]
        bb &= bb - 1
    return s


def evaluate(p, o):
    """Heuristic value from the perspective of the side to move (discs `p`)."""
    pos = _positional(p) - _positional(o)
    mob = legal_moves_bb(p, o).bit_count() - legal_moves_bb(o, p).bit_count()
    return pos + MOBILITY_WEIGHT * mob


def _negamax(p, o, depth, alpha, beta):
    """Negamax value from the side-to-move's perspective. `p` = side to move."""
    moves_bb = legal_moves_bb(p, o)
    if moves_bb == 0:
        if legal_moves_bb(o, p) == 0:            # neither can move -> game over
            return (p.bit_count() - o.bit_count()) * TERMINAL_SCALE
        return -_negamax(o, p, depth, -beta, -alpha)   # pass (no ply spent)
    if depth <= 0:
        return evaluate(p, o)
    # Move ordering by static square value greatly improves pruning.
    moves = sorted(bb_to_moves(moves_bb), key=lambda m: POS_W[m], reverse=True)
    best = -INF
    for sq in moves:
        np_, no_ = apply_move(p, o, sq)
        val = -_negamax(no_, np_, depth - 1, -beta, -alpha)
        if val > best:
            best = val
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break
    return best


def best_move(p, o, depth, exact_empties=8, rng=None, epsilon=0.0):
    """Return the best square for the side to move (`p`).

    depth          : search depth in the midgame.
    exact_empties  : when this many or fewer squares are empty, solve to the end
                     exactly (optimizing final disc difference).
    rng            : optional random.Random; ties are broken randomly, and with
                     probability `epsilon` a uniformly random legal move is
                     played instead (opening/diversity noise for data generation).
    """
    moves = bb_to_moves(legal_moves_bb(p, o))
    if not moves:
        return None
    if rng is not None and epsilon > 0.0 and rng.random() < epsilon:
        return moves[rng.randrange(len(moves))]

    empties = 64 - (p | o).bit_count()
    d = empties if empties <= exact_empties else depth

    scored = []
    alpha = -INF
    for sq in sorted(moves, key=lambda m: POS_W[m], reverse=True):
        np_, no_ = apply_move(p, o, sq)
        val = -_negamax(no_, np_, d - 1, -INF, -alpha)
        scored.append((val, sq))
        if val > alpha:
            alpha = val
    best = max(v for v, _ in scored)
    winners = [sq for v, sq in scored if v == best]
    if rng is not None and len(winners) > 1:
        return winners[rng.randrange(len(winners))]
    return winners[0]


# ---- move-function factories for the gauntlet / self-play loops ----
# Each returns fn(game, rng) -> square index, reading the live Game state.

def search_player(depth, exact_empties=8, epsilon=0.0):
    def fn(game, rng):
        p = game.boards[game.player]
        o = game.boards[1 - game.player]
        return best_move(p, o, depth, exact_empties, rng, epsilon)
    return fn


def greedy_player():
    """One-ply greedy on the positional table (the old data policy's strong end)."""
    def fn(game, rng):
        moves = bb_to_moves(game.legal_moves())
        best = max(POS_W[m] for m in moves)
        cands = [m for m in moves if POS_W[m] == best]
        return cands[rng.randrange(len(cands))] if rng else cands[0]
    return fn


def random_player():
    def fn(game, rng):
        moves = bb_to_moves(game.legal_moves())
        return moves[rng.randrange(len(moves))]
    return fn


def standard_ladder(max_depth=5, exact_empties=8):
    """Ordered weakest -> strongest opponent dict for the strength gauntlet."""
    ladder = {"random": random_player(), "greedy": greedy_player()}
    for d in range(1, max_depth + 1):
        ladder[f"search@{d}"] = search_player(d, exact_empties)
    return ladder
