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


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyShakespeare data.")
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
    ensure_data(args.data_path, url=args.url)


if __name__ == "__main__":
    main()
