"""
Engine correctness tests.

- Perft counts vs published Othello values (Aart Bik / RvB perft tables)
- Bitboard engine cross-checked against an independent naive implementation
- Opening moves, transcript round-trips, verify_transcript edge cases
- Self-play games are always complete legal games
"""

import random

import pytest

from othello.engine import (
    BLACK, WHITE, Game, initial_state, legal_moves_bb, apply_move,
    bb_to_moves, move_to_name, name_to_move, moves_to_text, text_to_moves,
    verify_transcript,
)
from othello.selfplay import play_one_game

# ---------------------------------------------------------------------------
# Naive reference implementation (independent of the bitboard code)
# ---------------------------------------------------------------------------

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def naive_initial():
    board = [[None] * 8 for _ in range(8)]
    board[3][3] = WHITE  # d4
    board[4][4] = WHITE  # e5
    board[3][4] = BLACK  # e4
    board[4][3] = BLACK  # d5
    return board


def naive_legal(board, player):
    opp = 1 - player
    legal = set()
    for r in range(8):
        for c in range(8):
            if board[r][c] is not None:
                continue
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                seen_opp = False
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == opp:
                    seen_opp = True
                    rr, cc = rr + dr, cc + dc
                if seen_opp and 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == player:
                    legal.add(r * 8 + c)
                    break
    return legal


def naive_apply(board, player, sq):
    opp = 1 - player
    r, c = sq // 8, sq % 8
    board = [row[:] for row in board]
    board[r][c] = player
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        path = []
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == opp:
            path.append((rr, cc))
            rr, cc = rr + dr, cc + dc
        if path and 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == player:
            for pr, pc in path:
                board[pr][pc] = player
    return board


def bb_from_naive(board, player):
    bb = 0
    for r in range(8):
        for c in range(8):
            if board[r][c] == player:
                bb |= 1 << (r * 8 + c)
    return bb

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_position():
    black, white = initial_state()
    assert bb_to_moves(black) == sorted([name_to_move("e4"), name_to_move("d5")])
    assert bb_to_moves(white) == sorted([name_to_move("d4"), name_to_move("e5")])


def test_opening_moves():
    black, white = initial_state()
    legal = legal_moves_bb(black, white)
    names = sorted(move_to_name(m) for m in bb_to_moves(legal))
    assert names == ["c4", "d3", "e6", "f5"]


def test_move_name_roundtrip():
    for sq in range(64):
        assert name_to_move(move_to_name(sq)) == sq
    assert move_to_name(0) == "a1"
    assert move_to_name(63) == "h8"
    with pytest.raises(ValueError):
        name_to_move("i1")
    with pytest.raises(ValueError):
        name_to_move("a9")


# Published Othello perft (from initial position, passes counted as in
# standard perft — a position where the mover must pass counts via the pass).
PERFT = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1396, 6: 8200, 7: 55092}


def _perft(p, o, depth, passed=False):
    if depth == 0:
        return 1
    legal = legal_moves_bb(p, o)
    if legal == 0:
        if passed:
            return 1  # game over
        return _perft(o, p, depth - 1, passed=True)
    total = 0
    for sq in bb_to_moves(legal):
        new_p, new_o = apply_move(p, o, sq)
        total += _perft(new_o, new_p, depth - 1)
    return total


@pytest.mark.parametrize("depth,expected", sorted(PERFT.items()))
def test_perft(depth, expected):
    black, white = initial_state()
    assert _perft(black, white, depth) == expected


def test_bitboard_matches_naive_on_random_games():
    rng = random.Random(0)
    for _ in range(50):
        game = Game()
        board = naive_initial()
        while not game.over:
            player = game.player
            bb_legal = set(bb_to_moves(game.legal_moves()))
            assert bb_legal == naive_legal(board, player)
            sq = rng.choice(sorted(bb_legal))
            game.play(sq)
            board = naive_apply(board, player, sq)
            assert game.boards[BLACK] == bb_from_naive(board, BLACK)
            assert game.boards[WHITE] == bb_from_naive(board, WHITE)
        # terminal: neither side has a move
        assert not naive_legal(board, 0) and not naive_legal(board, 1)


def test_transcript_roundtrip():
    game = Game()
    rng = random.Random(7)
    while not game.over:
        game.play(rng.choice(bb_to_moves(game.legal_moves())))
    text = moves_to_text(game.moves)
    assert text_to_moves(text) == game.moves
    result = verify_transcript(text)
    assert result["is_complete"]
    assert result["legal_moves"] == result["total_moves"] == len(game.moves)
    assert result["black"] + result["white"] <= 64
    assert (result["black"], result["white"]) == game.score()


def test_verify_transcript_illegal():
    # e6 is legal for black; a1 is never legal on move 2
    result = verify_transcript("e6 a1 d3")
    assert result["legal_moves"] == 1
    assert result["first_illegal"] == "a1"
    assert not result["is_complete"]
    # malformed token
    result = verify_transcript("e6 z9")
    assert result["legal_moves"] == 1
    assert result["first_illegal"] == "z9"
    # empty
    result = verify_transcript("")
    assert result["legal_moves"] == 0 and result["total_moves"] == 0
    assert not result["is_complete"]


def test_selfplay_games_complete_and_diverse():
    texts = [play_one_game(seed) for seed in range(30)]
    assert len(set(texts)) == len(texts), "seeded games should be distinct"
    for text in texts:
        result = verify_transcript(text)
        assert result["is_complete"], f"incomplete game: {text}"
        # a full game with no passes has 60 moves; passes shorten it
        assert 9 <= result["total_moves"] <= 60
