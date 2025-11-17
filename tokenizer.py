"""
Simple byte-level BPE tokenizer that can be trained from the dataset text.
Inspired by the minBPE implementation (https://github.com/karpathy/minbpe)
but tailored for this project with a reserved [MASK] token at id 0.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


DEFAULT_TOKENIZER_PATH = Path("data/bpe_tokenizer.json")
DEFAULT_DATA_PATH = Path("data/tiny_shakespeare.txt")
DEFAULT_VOCAB_SIZE = 1024


@dataclass
class ByteLevelBPETokenizer:
    merges: List[Tuple[int, int, int]]
    id_to_bytes: Dict[int, bytes]
    mask_token_id: int = 0

    def __post_init__(self):
        self.merge_ranks = {}
        self.pair_to_token = {}
        for rank, (a, b, new_id) in enumerate(self.merges):
            pair = (a, b)
            self.merge_ranks[pair] = rank
            self.pair_to_token[pair] = new_id
        self.vocab_size = max(self.id_to_bytes.keys(), default=self.mask_token_id) + 1

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        mask_token_id: int = 0,
    ) -> "ByteLevelBPETokenizer":
        byte_data = list(text.encode("utf-8"))
        if not byte_data:
            raise ValueError("Cannot train tokenizer on empty text.")

        # Start vocabulary with all 256 byte values (offset by mask token)
        id_to_bytes: Dict[int, bytes] = {
            byte_val + 1: bytes([byte_val]) for byte_val in range(256)
        }
        ids = [byte_val + 1 for byte_val in byte_data]
        next_id = max(id_to_bytes.keys()) + 1
        merges: List[Tuple[int, int, int]] = []

        # Perform greedy BPE merges until we reach the target vocabulary size
        while len(id_to_bytes) + 1 < vocab_size:
            stats = Counter(zip(ids, ids[1:]))
            if not stats:
                break

            best_pair, freq = stats.most_common(1)[0]
            if freq < 2:
                break  # no meaningful pairs left to merge

            new_id = next_id
            next_id += 1
            merges.append((best_pair[0], best_pair[1], new_id))
            merged_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
            id_to_bytes[new_id] = merged_bytes
            ids = cls._merge_sequence(ids, best_pair, new_id)

        return cls(merges=merges, id_to_bytes=id_to_bytes, mask_token_id=mask_token_id)

    def encode(self, text: str) -> torch.Tensor:
        byte_ids = [b + 1 for b in text.encode("utf-8")]
        merged_ids = self._apply_bpe(byte_ids)
        return torch.tensor(merged_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        bytes_out = bytearray()
        for token_id in token_ids:
            if token_id == self.mask_token_id:
                continue
            symbol = self.id_to_bytes.get(int(token_id))
            if symbol is None:
                continue
            bytes_out.extend(symbol)
        return bytes_out.decode("utf-8", errors="ignore")

    def save(self, path: Path) -> None:
        payload = {
            "mask_token_id": self.mask_token_id,
            "merges": self.merges,
            "id_to_bytes": {
                str(token_id): list(value) for token_id, value in self.id_to_bytes.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "ByteLevelBPETokenizer":
        payload = json.loads(path.read_text())
        merges = [tuple(entry) for entry in payload["merges"]]
        id_to_bytes = {
            int(token_id): bytes(values)
            for token_id, values in payload["id_to_bytes"].items()
        }
        return cls(
            merges=merges,
            id_to_bytes=id_to_bytes,
            mask_token_id=payload.get("mask_token_id", 0),
        )

    def _apply_bpe(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) < 2:
            return token_ids
        token_ids = token_ids[:]
        while True:
            pairs = self._get_pairs(token_ids)
            candidate = self._select_best_pair(pairs)
            if candidate is None:
                break
            new_token = self.pair_to_token[candidate]
            token_ids = self._merge_pair(token_ids, candidate, new_token)
        return token_ids

    def _select_best_pair(
        self, pairs: Iterable[Tuple[int, int]]
    ) -> Tuple[int, int] | None:
        best_pair = None
        best_rank = None
        for pair in pairs:
            rank = self.merge_ranks.get(pair)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_pair = pair
        return best_pair

    @staticmethod
    def _merge_sequence(
        ids: List[int], pair: Tuple[int, int], new_id: int
    ) -> List[int]:
        return ByteLevelBPETokenizer._merge_pair(ids, pair, new_id)

    @staticmethod
    def _merge_pair(
        ids: List[int], pair: Tuple[int, int], new_id: int
    ) -> List[int]:
        result: List[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    @staticmethod
    def _get_pairs(ids: List[int]) -> Iterable[Tuple[int, int]]:
        for i in range(len(ids) - 1):
            yield ids[i], ids[i + 1]


_TOKENIZER: ByteLevelBPETokenizer | None = None


def get_tokenizer(
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> ByteLevelBPETokenizer:
    """
    Load a cached tokenizer or train a new one from the dataset text.
    """
    global _TOKENIZER
    tokenizer_path = Path(tokenizer_path)
    data_path = Path(data_path)
    if _TOKENIZER is not None:
        return _TOKENIZER

    if tokenizer_path.exists():
        _TOKENIZER = ByteLevelBPETokenizer.load(tokenizer_path)
    else:
        text = data_path.read_text(encoding="utf-8")
        _TOKENIZER = ByteLevelBPETokenizer.train(
            text, vocab_size=vocab_size, mask_token_id=0
        )
        _TOKENIZER.save(tokenizer_path)
    return _TOKENIZER
