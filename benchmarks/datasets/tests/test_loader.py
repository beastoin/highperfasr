"""Tests for RoundRobinLoader — unique audio per round, deterministic shuffling."""

import json
import struct
import tempfile
from pathlib import Path

import pytest

from benchmarks.datasets.loader import RoundRobinLoader, load_manifest, save_manifest


def _make_wav(path: Path, duration_s: float = 1.0, sr: int = 16000):
    """Create a minimal valid WAV file."""
    num_samples = int(sr * duration_s)
    data_size = num_samples * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


@pytest.fixture
def manifest_10(tmp_path):
    """Create 10 fake manifest entries with WAV files."""
    entries = []
    for i in range(10):
        wav_path = tmp_path / f"utt_{i:03d}.wav"
        _make_wav(wav_path, duration_s=1.0 + i * 0.5)
        entries.append({
            "utt_id": f"utt_{i:03d}",
            "wav_path": str(wav_path),
            "duration_s": 1.0 + i * 0.5,
            "corpus": "test",
            "reference": f"this is utterance number {i}",
        })
    return entries


class TestRoundRobinLoader:
    def test_basic_round(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        batch = loader.next_round(concurrency=5)
        assert len(batch) == 5

    def test_unique_within_round(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        batch = loader.next_round(concurrency=10)
        utt_ids = [e["utt_id"] for e in batch]
        assert len(set(utt_ids)) == 10

    def test_concurrency_exceeds_dataset_raises(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        with pytest.raises(ValueError, match="exceeds dataset size"):
            loader.next_round(concurrency=11)

    def test_multiple_rounds_no_reuse_within(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        for _ in range(5):
            batch = loader.next_round(concurrency=5)
            utt_ids = [e["utt_id"] for e in batch]
            assert len(set(utt_ids)) == 5

    def test_full_corpus_covered_across_rounds(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        all_ids = set()
        for _ in range(2):
            batch = loader.next_round(concurrency=5)
            all_ids.update(e["utt_id"] for e in batch)
        assert len(all_ids) == 10

    def test_deterministic_with_same_seed(self, manifest_10):
        loader1 = RoundRobinLoader(manifest_10, seed=123)
        loader2 = RoundRobinLoader(manifest_10, seed=123)
        batch1 = loader1.next_round(concurrency=5)
        batch2 = loader2.next_round(concurrency=5)
        assert [e["utt_id"] for e in batch1] == [e["utt_id"] for e in batch2]

    def test_different_seeds_different_order(self, manifest_10):
        loader1 = RoundRobinLoader(manifest_10, seed=1)
        loader2 = RoundRobinLoader(manifest_10, seed=2)
        batch1 = loader1.next_round(concurrency=10)
        batch2 = loader2.next_round(concurrency=10)
        ids1 = [e["utt_id"] for e in batch1]
        ids2 = [e["utt_id"] for e in batch2]
        assert set(ids1) == set(ids2)
        assert ids1 != ids2

    def test_pool_refills_after_exhaustion(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        loader.next_round(concurrency=8)
        batch = loader.next_round(concurrency=8)
        assert len(batch) == 8

    def test_empty_manifest_raises(self):
        with pytest.raises(ValueError, match="Empty manifest"):
            RoundRobinLoader([])

    def test_wav_paths_helper(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        batch = loader.next_round(concurrency=3)
        paths = loader.wav_paths(batch)
        assert len(paths) == 3
        assert all(isinstance(p, str) for p in paths)

    def test_wer_entries_filter(self, manifest_10):
        manifest_10[0].pop("reference")
        loader = RoundRobinLoader(manifest_10)
        batch = loader.next_round(concurrency=10)
        wer = loader.wer_entries(batch)
        assert len(wer) == 9

    def test_properties(self, manifest_10):
        loader = RoundRobinLoader(manifest_10)
        assert loader.total_files == 10
        assert loader.total_duration_h > 0
        assert loader.rounds_completed == 0
        loader.next_round(concurrency=5)
        assert loader.rounds_completed == 1


class TestManifestIO:
    def test_save_and_load(self, manifest_10, tmp_path):
        path = tmp_path / "manifest.json"
        save_manifest(manifest_10, path)
        loaded = load_manifest(path)
        assert len(loaded) == 10
        assert loaded[0]["utt_id"] == manifest_10[0]["utt_id"]
