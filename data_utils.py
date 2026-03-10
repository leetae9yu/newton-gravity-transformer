"""WikiText data loading with tokenizer-aware caching."""

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

def _load_wikitext(tokenizer, data_path, hf_variant, cache_prefix, dataset_label):
    """Load a raw WikiText variant with tokenizer-aware caching."""
    if data_path is None:
        data_path = "data"
    os.makedirs(data_path, exist_ok=True)

    meta_path = os.path.join(data_path, f"{cache_prefix}_meta.json")
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
        cache_file = os.path.join(data_path, f"{cache_prefix}_{split_name}.pt")
        out_key = "val" if split_name == "validation" else split_name

        if cache_valid and os.path.exists(cache_file):
            print(f"Loading cached {split_name} split from {cache_file}...")
            splits[out_key] = torch.load(cache_file, weights_only=True)
        else:
            # Download and encode
            if hf_dataset is None:
                print(f"Loading {dataset_label} from HuggingFace datasets...")
                from datasets import load_dataset as hf_load_dataset
                hf_dataset = hf_load_dataset("wikitext", hf_variant)

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

    Returns: {"train": Tensor, "val": Tensor, "test": Tensor|None, "text": None}
    """
    if dataset_name == "wikitext2":
        return _load_wikitext(
            tokenizer,
            data_path,
            hf_variant="wikitext-2-raw-v1",
            cache_prefix="wikitext2",
            dataset_label="WikiText-2-raw-v1",
        )
    if dataset_name == "wikitext103":
        return _load_wikitext(
            tokenizer,
            data_path,
            hf_variant="wikitext-103-raw-v1",
            cache_prefix="wikitext103",
            dataset_label="WikiText-103-raw-v1",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
