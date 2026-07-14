#!/usr/bin/env python3
"""
Download and prepare benchmark datasets.

Usage:
    python3 -m benchmarks.datasets.download --corpus librispeech-test-clean
    python3 -m benchmarks.datasets.download --corpus all
    python3 -m benchmarks.datasets.download --corpus all --cache-dir /data/datasets
    python3 -m benchmarks.datasets.download --list
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.datasets.loader import save_manifest
from benchmarks.datasets.registry import CORPORA, DEFAULT_CACHE_DIR, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("datasets.download")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare benchmark datasets")
    parser.add_argument(
        "--corpus",
        default="all",
        help=f"Corpus to download: {', '.join(list(CORPORA.keys()) + ['all'])} (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Dataset cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per corpus (0=all)")
    parser.add_argument("--list", action="store_true", help="List available corpora and exit")
    parser.add_argument("--output-manifest", type=Path, help="Save combined manifest JSON to this path")
    args = parser.parse_args()

    if args.list:
        print("Available corpora:")
        for name, info in CORPORA.items():
            print(f"  {name}: {info['description']}")
        print(f"  all: Download all corpora")
        return

    manifest = load_dataset(args.corpus, cache_dir=args.cache_dir, max_samples=args.max_samples)

    print(f"\nDataset ready: {len(manifest)} files")
    print(f"Total duration: {sum(e['duration_s'] for e in manifest) / 3600:.1f}h")
    print(f"With references: {sum(1 for e in manifest if e.get('reference'))}")

    by_corpus = {}
    for e in manifest:
        by_corpus.setdefault(e["corpus"], []).append(e)
    for corpus, entries in sorted(by_corpus.items()):
        dur = sum(e["duration_s"] for e in entries) / 3600
        print(f"  {corpus}: {len(entries)} files, {dur:.1f}h")

    if args.output_manifest:
        save_manifest(manifest, args.output_manifest)
        print(f"\nManifest saved: {args.output_manifest}")


if __name__ == "__main__":
    main()
