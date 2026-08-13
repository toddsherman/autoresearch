"""
Arena: play Othello games between two move-functions and score strength.

A move-function has the signature fn(game, rng) -> square_index and reads the
live Game state (including game.moves, the full history), so a token-based model
player can reconstruct its context each turn. This keeps the arena pure Python
and independent of torch: engine players (random / greedy / alpha-beta) can be
pitted against each other here, and a model player slots into the exact same
loop on the GPU box.
"""

import math
import random

from othello.engine import BLACK, Game


def play_game(black_fn, white_fn, rng):
    """Play one full game. Returns (black_score, white_score, moves)."""
    game = Game()
    while not game.over:
        fn = black_fn if game.player == BLACK else white_fn
        game.play(fn(game, rng))
    b, w = game.score()
    return b, w, game.moves


def match(a_fn, b_fn, n_games=40, seed=0):
    """Play a_fn vs b_fn over n_games, alternating colors. Results from a_fn's
    perspective: wins/draws/losses and score = (wins + 0.5*draws) / n."""
    wins = draws = losses = 0
    for g in range(n_games):
        rng = random.Random(seed * 100003 + g)
        a_is_black = g % 2 == 0
        black_fn, white_fn = (a_fn, b_fn) if a_is_black else (b_fn, a_fn)
        bs, ws, _ = play_game(black_fn, white_fn, rng)
        a_score = bs if a_is_black else ws
        o_score = ws if a_is_black else bs
        if a_score > o_score:
            wins += 1
        elif a_score < o_score:
            losses += 1
        else:
            draws += 1
    n = n_games
    return {
        "n": n, "wins": wins, "draws": draws, "losses": losses,
        "score": (wins + 0.5 * draws) / n if n else 0.0,
    }


def elo_diff(score):
    """Elo rating difference implied by a score in (0,1). Clamped for 0/1."""
    score = min(max(score, 0.5 / 1000), 1 - 0.5 / 1000)
    return -400.0 * math.log10(1.0 / score - 1.0)


def gauntlet(a_fn, opponents, n_games=40, seed=0):
    """Run a_fn against a dict {name: move_fn} of opponents. Returns per-opponent
    results plus the implied Elo gap vs each. `opponents` should be ordered from
    weakest to strongest so the crossover (last rung still won >= 50%) is clear.
    """
    results = {}
    crossover = None
    streak_alive = True
    for name, opp in opponents.items():
        r = match(a_fn, opp, n_games=n_games, seed=seed)
        r["elo_gap"] = round(elo_diff(r["score"]), 1)
        results[name] = r
        # "beats_up_to" is the strongest rung in an unbroken >=50% streak from
        # the weakest opponent, so noise higher up the ladder can't inflate it.
        if streak_alive and r["score"] >= 0.5:
            crossover = name
        elif r["score"] < 0.5:
            streak_alive = False
    return {"per_opponent": results, "beats_up_to": crossover}
