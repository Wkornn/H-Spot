import os
import requests
import yaml

BASE_URL = "https://itic.longdo.com/opendata/probe-data"
OUTPUT_DIR = "data/raw/iTIC_probe_data"


def download_file(url, file_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


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
