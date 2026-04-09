import os
import requests
import yaml

BASE_URL = "https://itic.longdo.com/opendata/probe-data"
OUTPUT_DIR = "data/raw/iTIC_probe_data"


def download_file(url, file_path, retries=5):
    for attempt in range(1, retries + 1):
        existing_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                mode = "ab" if existing_size else "wb"
                with open(file_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            return
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise


def load_config(config_path="configs/data_sources.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_config()
    years = [d["year"] for d in config["datasets"]]

    files = [
        f"PROBE-{year}{month:02d}.tar.bz2"
        for year in years
        for month in range(1, 13)
    ]

    print(f"Downloading {len(files)} files...")

    for filename in files:
        file_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(file_path):
            print(f"Skip (exists): {filename}")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"Downloading: {filename}...")

        try:
            download_file(url, file_path)
        except requests.HTTPError as e:
            print(f"  Not available: {e}")
            os.remove(file_path)

    print(f"\nDone → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
