"""
Multi-corpus benchmark dataset infrastructure.

Provides download, caching, manifest generation, and a round-robin loader
that guarantees unique audio per concurrent stream within each round.

Corpora:
    - librispeech-test-clean: 2620 files, 5.4h — baseline WER reference
    - librispeech-test-other: 2939 files, 5.3h — harder speakers/accents
    - earnings22:             ~125 files, 44h  — real-world financial calls
    - commonvoice-en:        ~16k files, ~25h  — crowd-sourced diversity

Usage:
    from benchmarks.datasets import load_dataset, load_manifest, RoundRobinLoader

    # Download and prepare a corpus
    manifest = load_dataset("librispeech-test-clean")

    # Load all corpora into a single manifest
    manifest = load_dataset("all")

    # Round-robin loader for concurrency benchmarks
    loader = RoundRobinLoader(manifest)
    for round_idx in range(num_rounds):
        files = loader.next_round(concurrency=256)
        # Each of the 256 files is unique within this round
"""

from benchmarks.datasets.loader import RoundRobinLoader, load_manifest
from benchmarks.datasets.registry import CORPORA, TUNING_MANIFESTS, load_dataset

__all__ = ["CORPORA", "TUNING_MANIFESTS", "RoundRobinLoader", "load_dataset", "load_manifest"]
