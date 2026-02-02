"""
Unified tokenizer interface for char-level, BPE, and tiktoken tokenizers.

All tokenizers expose:
    encode(text) -> list[int]
    decode(ids)  -> str
    vocab_size   -> int
    save_state() -> dict
    from_state(state) -> Tokenizer  (classmethod)
"""

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


# ---------------------------------------------------------------------------
# Char-level tokenizer
# ---------------------------------------------------------------------------
class CharTokenizer(BaseTokenizer):
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self._stoi = stoi
        self._itos = itos

    def encode(self, text: str) -> list[int]:
        fallback = self._stoi.get(" ", 0)
        return [self._stoi.get(ch, fallback) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos.get(i, "") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    def save_state(self) -> dict:
        return {"type": "char", "stoi": self._stoi, "itos": self._itos}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        # itos keys may have been converted to strings by JSON round-trips;
        # ensure they are int.
        itos = {int(k): v for k, v in state["itos"].items()}
        return cls(stoi=state["stoi"], itos=itos)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)


# ---------------------------------------------------------------------------
# BPE tokenizer (HuggingFace tokenizers library)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tiktoken tokenizer (GPT-2 / cl100k_base wrapper)
# ---------------------------------------------------------------------------
class TiktokenTokenizer(BaseTokenizer):
    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)
        self._encoding_name = encoding_name

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def save_state(self) -> dict:
        return {"type": "tiktoken", "encoding_name": self._encoding_name}

    @classmethod
    def from_state(cls, state: dict) -> "TiktokenTokenizer":
        return cls(encoding_name=state.get("encoding_name", "gpt2"))


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_tokenizer(text: str, tokenizer_type: str = "char", bpe_vocab_size: int = 4000) -> BaseTokenizer:
    """Build a new tokenizer from training text."""
    if tokenizer_type == "char":
        return CharTokenizer.from_text(text)
    elif tokenizer_type == "bpe":
        return BPETokenizer.train_from_text(text, vocab_size=bpe_vocab_size)
    elif tokenizer_type == "tiktoken":
        return TiktokenTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def load_tokenizer(state: dict) -> BaseTokenizer:
    """Restore a tokenizer from a checkpoint state dict.

    Backwards compatible: if ``state`` has no ``type`` key but has ``stoi``/
    ``itos`` keys, it is treated as a legacy char tokenizer.
    """
    tok_type = state.get("type")

    if tok_type is None:
        # Legacy checkpoint — char tokenizer stored as {"stoi": ..., "itos": ...}
        if "stoi" in state and "itos" in state:
            return CharTokenizer.from_state({"stoi": state["stoi"], "itos": state["itos"]})
        raise ValueError("Cannot determine tokenizer type from checkpoint state.")

    if tok_type == "char":
        return CharTokenizer.from_state(state)
    elif tok_type == "bpe":
        return BPETokenizer.from_state(state)
    elif tok_type == "tiktoken":
        return TiktokenTokenizer.from_state(state)
    else:
        raise ValueError(f"Unknown tokenizer type in checkpoint: {tok_type}")


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
