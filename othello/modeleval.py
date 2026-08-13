"""
Model-vs-engine evaluation utilities for Othello language models.

Works with any model exposing the contract `model(idx) -> logits` where idx is
a (B, T) long tensor of token ids and logits is (B, T, vocab). Uses the fixed
move-level tokenizer from prepare.py (one token per move).
"""

import random

import torch
import torch.nn.functional as F

from othello.engine import (
    BLACK, Game, bb_to_moves, move_to_name, verify_transcript,
)


def build_move_tables(tokenizer):
    """Token id lookup for moves. Returns (first_ids, cont_ids) dicts mapping
    square index -> token id, for game-first moves ("d3") and continuation
    moves (" d3") respectively."""
    enc = tokenizer.enc
    first_ids, cont_ids = {}, {}
    for sq in range(64):
        name = move_to_name(sq)
        if name in ("d4", "e4", "d5", "e5"):
            continue  # center squares are never playable
        first_ids[sq] = enc.encode_single_token(name)
        cont_ids[sq] = enc.encode_single_token(" " + name)
    return first_ids, cont_ids


@torch.no_grad()
def sample_games(model_fn, tokenizer, n_games=16, max_new_tokens=70,
                 temperature=1.0, seed=0, device="cuda"):
    """Sample free-running games from the model (unconstrained — the model may
    emit illegal moves; that's the point, we measure it).

    Generation stops per-game when the model emits BOS (the next-game boundary
    it saw during training) or at max_new_tokens.

    Returns (games, stats): games is a list of dicts with the transcript and
    its verify_transcript analysis; stats aggregates legality metrics.
    """
    enc = tokenizer.enc
    bos = tokenizer.get_bos_token_id()
    gen = torch.Generator().manual_seed(seed)

    buf = torch.full((n_games, max_new_tokens + 1), bos, dtype=torch.long, device=device)
    stopped_at = [max_new_tokens] * n_games
    for t in range(max_new_tokens):
        logits = model_fn(buf[:, :t + 1])[:, -1, :].float()
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1).cpu()
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        for i in range(n_games):
            if stopped_at[i] == max_new_tokens and nxt[i].item() == bos:
                stopped_at[i] = t
        buf[:, t + 1] = nxt.to(device)

    games = []
    for i in range(n_games):
        ids = buf[i, 1:stopped_at[i] + 1].tolist()
        ids = [t for t in ids if t != bos]
        text = enc.decode(ids)
        result = verify_transcript(text)
        games.append({"moves": text, **result})

    total_moves = sum(g["total_moves"] for g in games)
    legal_moves = sum(g["legal_moves"] for g in games)
    stats = {
        "n_games": n_games,
        "legality_rate": legal_moves / total_moves if total_moves else 0.0,
        "complete_rate": sum(g["is_complete"] for g in games) / n_games,
        "mean_legal_prefix": legal_moves / n_games,
    }
    return games, stats


def model_player(model_fn, tokenizer, device="cuda"):
    """Adapt a model into an arena move-function fn(game, rng) -> square.

    Reconstructs the token context from game.moves each turn (so it sees the
    opponent's moves too), then plays the highest-logit *legal* move
    (legality-masked greedy). Slots into othello.arena alongside engine players.
    """
    first_ids, cont_ids = build_move_tables(tokenizer)
    bos = tokenizer.get_bos_token_id()

    @torch.no_grad()
    def fn(game, rng):
        ids = [bos]
        for i, m in enumerate(game.moves):
            ids.append(first_ids[m] if i == 0 else cont_ids[m])
        legal = bb_to_moves(game.legal_moves())
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model_fn(x)[0, -1].float()
        table = first_ids if not game.moves else cont_ids
        return max(legal, key=lambda m: logits[table[m]].item())

    return fn


def run_gauntlet(model_fn, tokenizer, n_games=40, max_depth=5, seed=0, device="cuda"):
    """Play the model against the standard weak->strong opponent ladder and
    return per-opponent scores, implied Elo gaps, and the strongest rung it
    still beats. This is the real playing-strength metric."""
    from othello.arena import gauntlet
    from othello.search import standard_ladder
    player = model_player(model_fn, tokenizer, device=device)
    return gauntlet(player, standard_ladder(max_depth), n_games=n_games, seed=seed)


@torch.no_grad()
def play_vs_random(model_fn, tokenizer, n_games=20, seed=0, device="cuda"):
    """Play the model against a uniform-random opponent, alternating colors.
    The model's move is the highest-logit *legal* move (legality-masked greedy),
    so this measures play strength, not legality. Returns stats dict."""
    first_ids, cont_ids = build_move_tables(tokenizer)
    bos = tokenizer.get_bos_token_id()
    wins = draws = losses = 0
    records = []
    for g in range(n_games):
        model_is_black = g % 2 == 0
        rng = random.Random(seed * 100003 + g)
        game = Game()
        ids = [bos]
        while not game.over:
            legal = bb_to_moves(game.legal_moves())
            model_to_move = (game.player == BLACK) == model_is_black
            if model_to_move:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model_fn(x)[0, -1].float()
                table = first_ids if len(ids) == 1 else cont_ids
                sq = max(legal, key=lambda m: logits[table[m]].item())
            else:
                sq = rng.choice(legal)
            table = first_ids if len(ids) == 1 else cont_ids
            ids.append(table[sq])
            game.play(sq)
        black, white = game.score()
        model_score = black if model_is_black else white
        opp_score = white if model_is_black else black
        if model_score > opp_score:
            wins += 1
        elif model_score < opp_score:
            losses += 1
        else:
            draws += 1
        records.append({
            "model_color": "black" if model_is_black else "white",
            "moves": " ".join(move_to_name(m) for m in game.moves),
            "model_score": model_score,
            "opp_score": opp_score,
        })
    return {
        "n_games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winrate": wins / n_games,
        "games": records,
    }
