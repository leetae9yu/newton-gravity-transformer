"""
Unified data loading module.

Supports:
  - shakespeare: char-level TinyShakespeare (read text → encode → 90/10 split)
  - wikitext103: WikiText-103-raw-v1 via HuggingFace datasets (train/val/test splits, cached)
"""

import json
import os
import hashlib
from array import array

import torch
import numpy as np


def _tokenizer_fingerprint(tokenizer) -> str:
    """Stable fingerprint for tokenizer state to validate cached encodings."""
    state = tokenizer.save_state()
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_shakespeare(tokenizer, data_path):
    """Load TinyShakespeare: read text, encode, 90/10 split."""
    if data_path is None:
        data_path = os.path.join("data", "input.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = tokenizer.encode_to_tensor(text)
    split_idx = int(0.9 * len(data))
    return {
        "train": data[:split_idx],
        "val": data[split_idx:],
        "test": None,
        "text": text,
    }


def _load_wikitext103(tokenizer, data_path):
    """Load WikiText-103-raw-v1 with caching."""
    if data_path is None:
        data_path = "data"
    os.makedirs(data_path, exist_ok=True)

    meta_path = os.path.join(data_path, "wikitext103_meta.json")
    tokenizer_state = tokenizer.save_state()
    tokenizer_type = tokenizer_state.get("type", "unknown")
    fingerprint = _tokenizer_fingerprint(tokenizer)
    tokenizer_key = f"{tokenizer_type}_{tokenizer.vocab_size}_{fingerprint[:12]}"

    # Check cache validity
    cache_valid = True
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cached_key = meta.get("tokenizer")
        if cached_key != tokenizer_key:
            print(f"Tokenizer mismatch (cached={cached_key}, current={tokenizer_key}). Re-encoding...")
            cache_valid = False
    else:
        cache_valid = False

    splits = {}
    hf_dataset = None
    for split_name in ("train", "validation", "test"):
        cache_file = os.path.join(data_path, f"wikitext103_{split_name}.pt")
        out_key = "val" if split_name == "validation" else split_name

        if cache_valid and os.path.exists(cache_file):
            print(f"Loading cached {split_name} split from {cache_file}...")
            splits[out_key] = torch.load(cache_file, weights_only=True)
        else:
            # Download and encode
            if hf_dataset is None:
                print("Loading WikiText-103-raw-v1 from HuggingFace datasets...")
                from datasets import load_dataset as hf_load_dataset
                hf_dataset = hf_load_dataset("wikitext", "wikitext-103-raw-v1")

            split_data = hf_dataset[split_name]

            # Stream-friendly encoding: avoid building one massive string or a huge Python int list.
            newline_ids = tokenizer.encode("\n")
            tok_buf = array("i")
            first = True
            total_chars = 0
            for line in split_data["text"]:
                if not line or not line.strip():
                    continue
                if not first and newline_ids:
                    tok_buf.extend(newline_ids)
                first = False
                total_chars += len(line) + 1
                tok_buf.extend(tokenizer.encode(line))

            print(f"Encoding {split_name} split (~{total_chars:,} chars)...")
            np_tokens = np.frombuffer(tok_buf, dtype=np.int32)
            tensor = torch.from_numpy(np_tokens)
            torch.save(tensor, cache_file)
            print(f"Cached {split_name} split: {tensor.numel():,} tokens -> {cache_file}")
            splits[out_key] = tensor
            continue
            # Concatenate all text, filtering empty lines
            lines = [line for line in split_data["text"] if line.strip()]
            full_text = "\n".join(lines)
            print(f"Encoding {split_name} split ({len(full_text):,} chars)...")
            tokens = tokenizer.encode(full_text)
            tensor = torch.tensor(tokens, dtype=torch.long)
            torch.save(tensor, cache_file)
            print(f"Cached {split_name} split: {len(tensor):,} tokens → {cache_file}")
            splits[out_key] = tensor

    if not cache_valid:
        with open(meta_path, "w") as f:
            json.dump({"tokenizer": tokenizer_key, "tokenizer_fingerprint": fingerprint}, f)

    return {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits.get("test"),
        "text": None,  # Too large to keep in memory as string
    }


def load_dataset(dataset_name, tokenizer, data_path=None):
    """
    Load a dataset and return tokenized splits.

    Returns: {"train": Tensor, "val": Tensor, "test": Tensor|None, "text": str|None}
    """
    if dataset_name == "shakespeare":
        return _load_shakespeare(tokenizer, data_path)
    elif dataset_name == "wikitext103":
        return _load_wikitext103(tokenizer, data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
