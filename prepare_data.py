import argparse
import os


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
    print("Training uses the project BPE tokenizer by default.")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data.")
    parser.add_argument(
        "--dataset",
        default="wikitext2",
        choices=["wikitext2"],
        help="Dataset to prepare (fixed: wikitext2).",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Cache directory for WikiText downloads and encoded splits.",
    )
    args = parser.parse_args()

    prepare_wikitext(data_path=args.data_path or "data", hf_variant="wikitext-2-raw-v1")


if __name__ == "__main__":
    main()
