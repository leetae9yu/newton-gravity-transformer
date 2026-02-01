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


def prepare_wikitext103(data_path="data"):
    """Download and cache WikiText-103-raw-v1 using HuggingFace datasets."""
    os.makedirs(data_path, exist_ok=True)
    print("Downloading WikiText-103-raw-v1 via HuggingFace datasets...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    # Print split sizes
    for split_name in ("train", "validation", "test"):
        num_lines = len(dataset[split_name])
        total_chars = sum(len(line) for line in dataset[split_name]["text"])
        print(f"  {split_name}: {num_lines:,} lines, {total_chars:,} chars")
    print(f"WikiText-103 downloaded and cached by HuggingFace datasets.")
    print(f"To pre-encode with tiktoken, run training with --dataset wikitext103 --tokenizer tiktoken")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data.")
    parser.add_argument(
        "--dataset",
        default="shakespeare",
        choices=["shakespeare", "wikitext103"],
        help="Dataset to prepare (default: shakespeare).",
    )
    parser.add_argument(
        "--data-path",
        default=os.path.join("data", "input.txt"),
        help="Destination path for the TinyShakespeare input.txt file.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source URL for TinyShakespeare input.txt.",
    )
    args = parser.parse_args()

    if args.dataset == "wikitext103":
        prepare_wikitext103(data_path="data")
    else:
        ensure_data(args.data_path, url=args.url)


if __name__ == "__main__":
    main()
