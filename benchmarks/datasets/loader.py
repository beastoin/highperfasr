#!/usr/bin/env python3
"""
Round-robin dataset loader for concurrency benchmarks.

Guarantees unique audio per concurrent stream within each round.
Shuffles deterministically per round to defeat caching.
"""

import json
import logging
import random
from pathlib import Path

log = logging.getLogger("datasets.loader")


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load a manifest JSON file."""
    with open(manifest_path) as f:
        return json.load(f)


def save_manifest(manifest: list[dict], path: str | Path) -> None:
    """Save manifest to JSON."""
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


class RoundRobinLoader:
    """Assigns unique audio files to concurrent streams, round by round.

    Given N files and concurrency C:
    - Each round yields exactly C files, all unique within that round.
    - Files are shuffled deterministically (seed = round_index) to defeat caching.
    - When C > N, raises ValueError (not enough unique files).
    - Cycles through the full corpus across rounds before repeating.

    Usage:
        loader = RoundRobinLoader(manifest)
        for round_idx in range(num_rounds):
            batch = loader.next_round(concurrency=256)
            # batch is a list of 256 unique manifest entries
    """

    def __init__(self, manifest: list[dict], seed: int = 42):
        if not manifest:
            raise ValueError("Empty manifest")
        self._manifest = list(manifest)
        self._seed = seed
        self._round = 0
        self._pool: list[dict] = []
        self._rng = random.Random(seed)
        self._refill_pool()

    def _refill_pool(self):
        """Refill and shuffle the pool from the full manifest."""
        self._pool = list(self._manifest)
        self._rng.shuffle(self._pool)

    def next_round(self, concurrency: int) -> list[dict]:
        """Get the next round of unique files for the given concurrency.

        Args:
            concurrency: Number of unique files needed.

        Returns:
            List of manifest entries, length == concurrency.

        Raises:
            ValueError: If concurrency exceeds total manifest size.
        """
        if concurrency > len(self._manifest):
            raise ValueError(
                f"Concurrency {concurrency} exceeds dataset size {len(self._manifest)}. "
                f"Add more corpora or reduce concurrency."
            )

        if len(self._pool) < concurrency:
            self._refill_pool()

        batch = self._pool[:concurrency]
        self._pool = self._pool[concurrency:]
        self._round += 1
        return batch

    @property
    def total_files(self) -> int:
        return len(self._manifest)

    @property
    def total_duration_h(self) -> float:
        return sum(e.get("duration_s", 0) for e in self._manifest) / 3600

    @property
    def rounds_completed(self) -> int:
        return self._round

    def wav_paths(self, entries: list[dict]) -> list[str]:
        """Extract wav_path list from manifest entries."""
        return [e["wav_path"] for e in entries]

    def wer_entries(self, entries: list[dict]) -> list[dict]:
        """Filter entries that have reference text (for WER evaluation)."""
        return [e for e in entries if e.get("reference")]
