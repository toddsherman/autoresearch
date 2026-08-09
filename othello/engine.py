"""
Bitboard Othello engine.

Square indexing: index = row * 8 + col, where col 0..7 = files a..h and
row 0..7 = ranks 1..8. Square names are file+rank, e.g. "d3".
Standard initial position: White on d4/e5, Black on e4/d5. Black moves first.

Transcripts are space-separated move names in play order ("d3 c5 f6 ...").
Passes are not recorded (standard Othello transcript convention); the engine
auto-passes when the side to move has no legal move.
"""

FULL = 0xFFFFFFFFFFFFFFFF
NOT_A_FILE = 0xFEFEFEFEFEFEFEFE  # col a cleared
NOT_H_FILE = 0x7F7F7F7F7F7F7F7F  # col h cleared

BLACK, WHITE = 0, 1


def _shift_n(bb):
    return bb >> 8

def _shift_s(bb):
    return (bb << 8) & FULL

def _shift_e(bb):
    return ((bb & NOT_H_FILE) << 1) & FULL

def _shift_w(bb):
    return (bb & NOT_A_FILE) >> 1

def _shift_ne(bb):
    return (bb & NOT_H_FILE) >> 7

def _shift_nw(bb):
    return (bb & NOT_A_FILE) >> 9

def _shift_se(bb):
    return ((bb & NOT_H_FILE) << 9) & FULL

def _shift_sw(bb):
    return ((bb & NOT_A_FILE) << 7) & FULL


SHIFTS = (_shift_n, _shift_s, _shift_e, _shift_w, _shift_ne, _shift_nw, _shift_se, _shift_sw)


def initial_state():
    """Returns (black_bb, white_bb). Black moves first."""
    black = (1 << 28) | (1 << 35)  # e4, d5
    white = (1 << 27) | (1 << 36)  # d4, e5
    return black, white


def legal_moves_bb(p, o):
    """Bitboard of legal moves for player with discs `p` against opponent `o`."""
    empty = ~(p | o) & FULL
    legal = 0
    for shift in SHIFTS:
        x = shift(p) & o
        x |= shift(x) & o
        x |= shift(x) & o
        x |= shift(x) & o
        x |= shift(x) & o
        x |= shift(x) & o
        legal |= shift(x) & empty
    return legal


def apply_move(p, o, sq):
    """Apply move at square index `sq` for player `p`. Returns (new_p, new_o).
    Assumes the move is legal (flips at least one disc)."""
    move_bb = 1 << sq
    flips = 0
    for shift in SHIFTS:
        captured = 0
        x = shift(move_bb)
        while x & o:
            captured |= x
            x = shift(x)
        if x & p:
            flips |= captured
    return p | move_bb | flips, o & ~flips


def bb_to_moves(bb):
    """List of square indices set in a bitboard."""
    moves = []
    while bb:
        lsb = bb & -bb
        moves.append(lsb.bit_length() - 1)
        bb &= bb - 1
    return moves


def move_to_name(sq):
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


def name_to_move(name):
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"Invalid move name: {name!r}")
    return (int(name[1]) - 1) * 8 + "abcdefgh".index(name[0])


def moves_to_text(moves):
    return " ".join(move_to_name(m) for m in moves)


def text_to_moves(text):
    return [name_to_move(t) for t in text.split()]


class Game:
    """Mutable game state with auto-pass and transcript tracking."""

    def __init__(self):
        self.boards = list(initial_state())  # [black_bb, white_bb]
        self.player = BLACK
        self.moves = []       # square indices, in play order
        self.over = False

    def legal_moves(self):
        return legal_moves_bb(self.boards[self.player], self.boards[1 - self.player])

    def play(self, sq):
        """Play a move for the side to move. Raises ValueError if illegal.
        Auto-passes afterwards; sets self.over when neither side can move."""
        if self.over:
            raise ValueError("Game is over")
        if not (self.legal_moves() >> sq) & 1:
            raise ValueError(f"Illegal move: {move_to_name(sq)}")
        p, o = self.boards[self.player], self.boards[1 - self.player]
        new_p, new_o = apply_move(p, o, sq)
        self.boards[self.player] = new_p
        self.boards[1 - self.player] = new_o
        self.moves.append(sq)
        # Advance turn with auto-pass
        self.player = 1 - self.player
        if self.legal_moves() == 0:
            self.player = 1 - self.player
            if self.legal_moves() == 0:
                self.over = True

    def score(self):
        """(black_discs, white_discs)"""
        return self.boards[BLACK].bit_count(), self.boards[WHITE].bit_count()


def verify_transcript(text):
    """Validate a transcript string. Never raises on bad input.

    Returns a dict:
      total_moves:   number of whitespace-separated tokens
      legal_moves:   length of the longest legal prefix (malformed token = illegal)
      is_complete:   True if all moves legal AND the game reached a terminal state
      black, white:  final disc counts at the end of the legal prefix
      first_illegal: the first offending token, or None
    """
    tokens = text.split()
    game = Game()
    legal = 0
    first_illegal = None
    for tok in tokens:
        if game.over:
            first_illegal = tok
            break
        try:
            game.play(name_to_move(tok))
        except ValueError:
            first_illegal = tok
            break
        legal += 1
    black, white = game.score()
    return {
        "total_moves": len(tokens),
        "legal_moves": legal,
        "is_complete": game.over and legal == len(tokens),
        "black": black,
        "white": white,
        "first_illegal": first_illegal,
    }
