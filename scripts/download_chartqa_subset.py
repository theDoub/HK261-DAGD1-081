from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import urllib.request

GITHUB_API_DIR = "https://api.github.com/repos/vis-nlp/ChartQA/contents/ChartQA%20Dataset/test"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test"


def fetch_github_file_list(subfolder: str) -> list[str]:
    url = f"{GITHUB_API_DIR}/{subfolder}"
    req = urllib.request.Request(url, headers={"User-Agent": "ChartQA-Downloader"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return sorted([item["name"] for item in data if item["type"] == "file"])


def download_single_file(url: str, dest_path: Path) -> bool:
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "ChartQA-Downloader"})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
            out_f.write(resp.read())
        return True
    except Exception as exc:
        sys.stderr.write(f"Failed to download {url}: {exc}\n")
        return False


def download_chartqa_subset(target_dir: Path, num_samples: int = 300, max_workers: int = 16) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    png_dir = target_dir / "png"
    tables_dir = target_dir / "tables"

    print("Fetching file manifest from GitHub...")
    all_tables = fetch_github_file_list("tables")
    target_stems = [Path(f).stem for f in all_tables[:num_samples]]
    print(f"Targeting first {len(target_stems)} samples...")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for stem in target_stems:
            csv_name = f"{stem}.csv"
            png_name = f"{stem}.png"
            tasks.append(
                executor.submit(
                    download_single_file,
                    f"{GITHUB_RAW_BASE}/tables/{csv_name}",
                    tables_dir / csv_name,
                )
            )
            tasks.append(
                executor.submit(
                    download_single_file,
                    f"{GITHUB_RAW_BASE}/png/{png_name}",
                    png_dir / png_name,
                )
            )
        tasks.append(
            executor.submit(
                download_single_file,
                f"{GITHUB_RAW_BASE}/test_human.json",
                target_dir / "test_human.json",
            )
        )
        tasks.append(
            executor.submit(
                download_single_file,
                f"{GITHUB_RAW_BASE}/test_augmented.json",
                target_dir / "test_augmented.json",
            )
        )

        completed = 0
        total_tasks = len(tasks)
        for future in as_completed(tasks):
            if future.result():
                completed += 1
            if completed % 100 == 0 or completed == total_tasks:
                print(f"Downloaded {completed}/{total_tasks} files...")

    print(f"Download complete: {completed}/{total_tasks} files saved to {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ChartQA test subset directly from GitHub.")
    parser.add_argument("--target-dir", default=Path("data/chartqa_test"), type=Path)
    parser.add_argument("--num-samples", default=300, type=int)
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()

    download_chartqa_subset(args.target_dir, args.num_samples, args.workers)


if __name__ == "__main__":
    main()
