"""
One-time data preparation for autoresearch experiments (Othello edition).

Generates Othello self-play game transcripts as parquet shards and builds a
deterministic move-level tokenizer (one token per move, e.g. " d3").

Usage:
    python prepare.py                  # full prep (generate data + tokenizer)
    python prepare.py --num-shards 4   # generate only 4 train shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle

import pyarrow.parquet as pq
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 8 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_SHARD = 9999  # pinned validation shard
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
GAMES_PER_SHARD = 65536
TRAIN_SEED = 1234
VAL_SEED = 999_000_000  # disjoint from any train shard's seed range

# Split pattern: each chunk is exactly one move (optionally with its leading
# space), with catch-all alternatives so no byte is ever silently dropped.
SPLIT_PATTERN = r" ?[a-h][1-8]|\s+|\S"

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(num_shards):
    """Generate training shards + pinned validation shard of self-play games."""
    from othello.selfplay import generate_shards
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data: generating {num_shards} train shards + 1 val shard...")
    generate_shards(DATA_DIR, num_shards, GAMES_PER_SHARD, base_seed=TRAIN_SEED)
    # Val shard: separate seed universe so it never overlaps train data
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if not os.path.exists(val_path):
        import pyarrow as pa
        from othello.selfplay import play_one_game
        t0 = time.time()
        texts = [play_one_game(VAL_SEED + i) for i in range(GAMES_PER_SHARD)]
        table = pa.table({"text": texts})
        tmp = val_path + ".tmp"
        pq.write_table(table, tmp, row_group_size=8192)
        os.rename(tmp, val_path)
        print(f"  {VAL_FILENAME}: generated in {time.time() - t0:.1f}s")
    else:
        print(f"  {VAL_FILENAME}: already exists")

# ---------------------------------------------------------------------------
# Tokenizer construction (deterministic, no training required)
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def build_tokenizer():
    """Build the fixed move-level tokenizer and save as tiktoken pickle.

    Vocabulary: 256 byte tokens, 8 " <file>" intermediates, 60 " <move>"
    tokens, 60 bare "<move>" tokens (game-first move has no leading space),
    plus 4 special tokens. Every legal move encodes to exactly one token.
    """
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already built at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # The four center squares (d4, e4, d5, e5) are occupied at game start and
    # can never be played, so they are excluded from the move vocabulary.
    center = {"d4", "e4", "d5", "e5"}
    squares = [f + r for r in "12345678" for f in "abcdefgh" if f + r not in center]
    assert len(squares) == 60

    mergeable_ranks = {bytes([i]): i for i in range(256)}
    rank = 256
    for f in "abcdefgh":                      # " d" intermediates
        mergeable_ranks[b" " + f.encode()] = rank
        rank += 1
    for sq in squares:                        # " d3" full moves
        mergeable_ranks[b" " + sq.encode()] = rank
        rank += 1
    for sq in squares:                        # "d3" game-first moves
        mergeable_ranks[sq.encode()] = rank
        rank += 1

    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="othello_moves",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: built (vocab_size={enc.n_vocab}), saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    special_ids = set(special_tokens.values())
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        if token_id in special_ids:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(enc.decode_single_token_bytes(token_id)))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check: one token per move, exact roundtrip
    test = "d3 c5 f6 e6 f5"
    encoded = enc.encode_ordinary(test)
    assert len(encoded) == 5, f"Expected 5 tokens, got {len(encoded)}: {encoded}"
    assert enc.decode(encoded) == test, f"Tokenizer roundtrip failed: {test!r}"
    print("Tokenizer: sanity check passed (1 token per move)")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Construction is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Othello data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=16, help="Number of training shards to generate. Val shard is always pinned.")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Generate data
    generate_data(args.num_shards)
    print()

    # Step 2: Build tokenizer
    build_tokenizer()
    print()
    print("Done! Ready to train.")
