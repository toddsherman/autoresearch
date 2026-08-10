"""
Act III — Othello world-model probe (runs on the pod; needs GPU + the training env).

Question: does a model trained only on move sequences build an internal
representation of the board? Following Li et al. 2023 ("Emergent World
Representations") and Nanda 2023 (the representation is linear in *player-relative*
coordinates), we freeze a trained checkpoint, capture its residual-stream
activations as it reads games, and train a linear probe per square to decode
each square's state (mine / theirs / empty) from those activations. High probe
accuracy = the board is linearly represented inside the model.

Loads a night's checkpoint by reconstructing that run's exact model class from its
train_snapshot.py (config-only edits, so the base GPT class applies). Emits
site-ready JSON: per-square probe accuracy + sample games with the model's decoded
internal belief at each position, for the Act III visualization.

Usage (on the pod):
    python othello_probe.py --run-id run_016_80f660f --layers 12,16,20 \
        --games 600 --out /workspace/probe_results.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from prepare import Tokenizer, TOKENIZER_DIR
from othello.engine import Game, BLACK, WHITE, bb_to_moves, move_to_name
from othello.modeleval import build_move_tables

CENTER = {"d4", "e4", "d5", "e5"}
SQUARES = [f + r for r in "12345678" for f in "abcdefgh"]           # a1..h8, index = engine sq
PLAYABLE = [i for i, s in enumerate(SQUARES) if s not in CENTER]    # 60 squares


def load_model(run_id, telemetry_dir="telemetry"):
    """Reconstruct the run's model class from its train_snapshot.py (definitions
    only, no training) and load its checkpoint. Returns (model, config)."""
    run_dir = os.path.join(telemetry_dir, "runs", run_id)
    snapshot = open(os.path.join(run_dir, "train_snapshot.py")).read()
    # Everything before the training-setup section is pure definitions (imports,
    # the GPT model, the optimizer, hyperparameter constants) — safe to exec.
    marker = "\nt_start = time.time()"
    assert marker in snapshot, "could not locate setup boundary in snapshot"
    defs = snapshot.split(marker)[0]
    ns = {}
    exec(compile(defs, "<snapshot_defs>", "exec"), ns)
    GPT, GPTConfig = ns["GPT"], ns["GPTConfig"]

    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cuda")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).cuda()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # init_weights() casts the token + value embeddings to bf16 so that under
    # autocast every attention input shares a dtype; replicate that here (we
    # loaded weights instead of calling init_weights, which would reinitialize).
    model.transformer.wte.to(dtype=torch.bfloat16)
    for ve in model.value_embeds.values():
        ve.to(dtype=torch.bfloat16)
    model.eval()
    return model, config


def player_relative_board(game_black, game_white, player):
    """Per-square label from the mover's perspective: 0 empty, 1 mine, 2 theirs."""
    labels = np.zeros(64, dtype=np.int64)
    mine, theirs = (game_black, game_white) if player == BLACK else (game_white, game_black)
    for sq in bb_to_moves(mine):
        labels[sq] = 1
    for sq in bb_to_moves(theirs):
        labels[sq] = 2
    return labels


def collect(model, tokenizer, layer_set, n_games, seed, device="cuda"):
    """Play random-legal games; for each position capture the residual-stream
    activation (per hooked layer) at the just-played token and the board label."""
    caps = {}
    handles = []
    for li in layer_set:
        block = model.transformer.h[li]
        def mk(li):
            def hook(_m, _in, out):
                caps[li] = out.detach()
            return hook
        handles.append(block.register_forward_hook(mk(li)))

    bos = tokenizer.get_bos_token_id()
    first_ids, cont_ids = build_move_tables(tokenizer)
    data = {li: {"X": [], "y": []} for li in layer_set}
    rng = random.Random(seed)

    with torch.no_grad():
        for g in range(n_games):
            game = Game()
            ids = [bos]
            # after playing move at position t, activation at that token encodes
            # the board *after* the move; label = board from next player's view
            while not game.over and len(ids) <= 64:
                legal = bb_to_moves(game.legal_moves())
                sq = rng.choice(legal)
                table = first_ids if len(ids) == 1 else cont_ids
                ids.append(table[sq])
                game.play(sq)
                x = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(x)
                label = player_relative_board(game.boards[BLACK], game.boards[WHITE], game.player)
                for li in layer_set:
                    act = caps[li][0, -1].float().cpu().numpy()   # (d_model,)
                    data[li]["X"].append(act)
                    data[li]["y"].append(label)
    for h in handles:
        h.remove()
    for li in layer_set:
        data[li]["X"] = np.stack(data[li]["X"])
        data[li]["y"] = np.stack(data[li]["y"])
    return data


def train_probe(X, y, d_model, device="cuda", epochs=300, lr=0.01):
    """One linear probe over all squares at once: Linear(d_model -> 64*3), 3-class
    cross-entropy per square, center squares masked out. Returns (per-square acc,
    weight tensor) evaluated on a held-out split."""
    n = X.shape[0]
    idx = np.arange(n); np.random.RandomState(0).shuffle(idx)
    split = int(n * 0.85)
    tr, te = idx[:split], idx[split:]
    Xt = torch.tensor(X[tr], device=device)
    yt = torch.tensor(y[tr], device=device)
    Xe = torch.tensor(X[te], device=device)
    ye = torch.tensor(y[te], device=device)
    probe = torch.nn.Linear(d_model, 64 * 3).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    playable = torch.tensor(PLAYABLE, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(Xt).view(-1, 64, 3)[:, playable]           # (B, 60, 3)
        loss = F.cross_entropy(logits.reshape(-1, 3), yt[:, playable].reshape(-1))
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = probe(Xe).view(-1, 64, 3).argmax(-1)               # (B, 64)
        acc = np.zeros(64)
        for sq in PLAYABLE:
            acc[sq] = (pred[:, sq] == ye[:, sq]).float().mean().item()
    return acc, probe


def decode_game(model, probe, tokenizer, layer, game_moves, device="cuda"):
    """Replay one game; at each position, decode the model's internal belief board
    from the probe. Returns per-move {decoded, truth, player} for the animation."""
    caps = {}
    h = model.transformer.h[layer].register_forward_hook(lambda m, i, o: caps.__setitem__(0, o.detach()))
    bos = tokenizer.get_bos_token_id()
    first_ids, cont_ids = build_move_tables(tokenizer)
    game = Game(); ids = [bos]; frames = []
    with torch.no_grad():
        for name in game_moves.split():
            from othello.engine import name_to_move
            sq = name_to_move(name)
            if not (game.legal_moves() >> sq) & 1:
                break
            table = first_ids if len(ids) == 1 else cont_ids
            ids.append(table[sq]); game.play(sq)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(x)
            act = caps[0][0, -1].float()
            belief = probe(act).view(64, 3).argmax(-1).cpu().numpy()   # 0 empty,1 mine,2 theirs
            truth = player_relative_board(game.boards[BLACK], game.boards[WHITE], game.player)
            frames.append({
                "move": name, "player": int(game.player),
                "belief": belief.tolist(), "truth": truth.tolist(),
            })
    h.remove()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--telemetry-dir", default="telemetry")
    ap.add_argument("--layers", default="12,16,20")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="probe_results.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    layer_set = [int(x) for x in args.layers.split(",")]
    tokenizer = Tokenizer.from_directory(TOKENIZER_DIR)
    model, config = load_model(args.run_id, args.telemetry_dir)
    d_model = config.n_embd
    layer_set = [li for li in layer_set if li < config.n_layer]
    print(f"model: {config.n_layer} layers, d_model={d_model}; probing layers {layer_set}")

    data = collect(model, tokenizer, layer_set, args.games, args.seed)
    print(f"collected {data[layer_set[0]]['X'].shape[0]} positions")

    per_layer = {}
    best_layer, best_mean, best_probe = None, -1, None
    for li in layer_set:
        acc, probe = train_probe(data[li]["X"], data[li]["y"], d_model)
        mean = float(np.mean([acc[s] for s in PLAYABLE]))
        per_layer[li] = {"per_square": acc.tolist(), "mean_acc": mean}
        print(f"  layer {li}: mean probe accuracy = {mean:.3f}")
        if mean > best_mean:
            best_layer, best_mean, best_probe = li, mean, probe

    # sample games (from random legal play) decoded through the best layer
    rng = random.Random(999)
    sample_games = []
    for _ in range(4):
        game = Game()
        while not game.over:
            game.play(rng.choice(bb_to_moves(game.legal_moves())))
        moves = " ".join(move_to_name(m) for m in game.moves)
        sample_games.append(decode_game(model, best_probe, tokenizer, best_layer, moves))

    # A chance baseline: always predict the most common class (empty early on).
    result = {
        "run_id": args.run_id,
        "config": {"n_layer": config.n_layer, "n_embd": d_model},
        "best_layer": best_layer,
        "best_mean_acc": best_mean,
        "per_layer": per_layer,
        "playable_squares": PLAYABLE,
        "square_names": SQUARES,
        "decoded_games": sample_games,
        "n_games": args.games,
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}: best layer {best_layer}, mean accuracy {best_mean:.3f}")


if __name__ == "__main__":
    main()
