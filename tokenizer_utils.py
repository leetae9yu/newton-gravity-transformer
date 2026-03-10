"""BPE tokenizer utilities used by the project."""

import json
import os
from abc import ABC, abstractmethod

import torch


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @abstractmethod
    def save_state(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> "BaseTokenizer":
        ...

    def encode_to_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)

class BPETokenizer(BaseTokenizer):
    def __init__(self, tokenizer_obj):
        self._tok = tokenizer_obj

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def save_state(self) -> dict:
        return {"type": "bpe", "tokenizer_json": self._tok.to_str()}

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(state["tokenizer_json"])
        return cls(tok)

    @classmethod
    def train_from_text(cls, text: str, vocab_size: int = 4000) -> "BPETokenizer":
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"],
            show_progress=True,
        )
        # Train from iterator (single string split into lines)
        tok.train_from_iterator(text.splitlines(), trainer=trainer)
        print(f"BPE tokenizer trained: vocab_size={tok.get_vocab_size()}")
        return cls(tok)

    @classmethod
    def train_from_iterator(cls, iterator, vocab_size: int = 4000) -> "BPETokenizer":
        """Train a BPE tokenizer from a text iterator (streaming-friendly)."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"],
            show_progress=True,
        )
        tok.train_from_iterator(iterator, trainer=trainer)
        print(f"BPE tokenizer trained: vocab_size={tok.get_vocab_size()}")
        return cls(tok)
def load_tokenizer(state: dict) -> BaseTokenizer:
    """Restore a BPE tokenizer from a checkpoint state dict."""
    tok_type = state.get("type")

    if tok_type == "bpe":
        return BPETokenizer.from_state(state)

    raise ValueError(f"Only BPE tokenizer checkpoints are supported, got type={tok_type!r}")


def save_tokenizer_to_path(tokenizer: BaseTokenizer, path: str) -> None:
    """Save tokenizer state to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = tokenizer.save_state()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def load_tokenizer_from_path(path: str) -> BaseTokenizer:
    """Load tokenizer state from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return load_tokenizer(state)


def train_bpe_tokenizer_from_iterator(iterator, vocab_size: int, tokenizer_path: str | None = None) -> BaseTokenizer:
    """Train a BPE tokenizer from an iterator and optionally save it."""
    tok = BPETokenizer.train_from_iterator(iterator, vocab_size=vocab_size)
    if tokenizer_path:
        save_tokenizer_to_path(tok, tokenizer_path)
    return tok
