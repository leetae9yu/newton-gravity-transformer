import argparse
import os
import urllib.request

DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def ensure_data(data_path, url=DEFAULT_URL):
    data_dir = os.path.dirname(data_path)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
        print(f"Data already present at {data_path}.")
        return data_path

    print(f"Downloading TinyShakespeare to {data_path}...")
    urllib.request.urlretrieve(url, data_path)
    size = os.path.getsize(data_path)
    print(f"Download complete ({size} bytes).")
    return data_path


def prepare_wikitext(data_path="data", hf_variant="wikitext-2-raw-v1"):
    """Download and cache a raw WikiText variant using HuggingFace datasets."""
    os.makedirs(data_path, exist_ok=True)
    print(f"Downloading {hf_variant} via HuggingFace datasets...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", hf_variant)
    # Print split sizes
    for split_name in ("train", "validation", "test"):
        num_lines = len(dataset[split_name])
        total_chars = sum(len(line) for line in dataset[split_name]["text"])
        print(f"  {split_name}: {num_lines:,} lines, {total_chars:,} chars")
    print(f"{hf_variant} downloaded and cached by HuggingFace datasets.")
    dataset_arg = "wikitext2" if hf_variant == "wikitext-2-raw-v1" else "wikitext103"
    print(f"To pre-encode with tiktoken, run training with --dataset {dataset_arg} --tokenizer tiktoken")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data.")
    parser.add_argument(
        "--dataset",
        default="wikitext2",
        choices=["shakespeare", "wikitext2", "wikitext103"],
        help="Dataset to prepare (default: wikitext2).",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Cache directory for WikiText or destination file path for TinyShakespeare.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source URL for TinyShakespeare input.txt.",
    )
    args = parser.parse_args()

    if args.dataset == "wikitext2":
        prepare_wikitext(data_path=args.data_path or "data", hf_variant="wikitext-2-raw-v1")
    elif args.dataset == "wikitext103":
        prepare_wikitext(data_path=args.data_path or "data", hf_variant="wikitext-103-raw-v1")
    else:
        ensure_data(args.data_path or os.path.join("data", "input.txt"), url=args.url)


if __name__ == "__main__":
    main()
