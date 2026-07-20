"""Tests for corpus registry — metadata, FLAC conversion, manifest building."""

import hashlib
import io
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.datasets.registry import (
    BENCHMARK_CORPORA,
    CORPORA,
    TUNING_MANIFESTS,
    _build_manifest,
    _download_file,
    _get_wav_duration,
    load_dataset,
)


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


class TestCorpusRegistry:
    def test_required_corpora_present(self):
        assert "librispeech-test-clean" in CORPORA
        assert "librispeech-test-other" in CORPORA

    def test_corpus_has_required_fields(self):
        for name, info in CORPORA.items():
            assert "url" in info, f"{name} missing url"
            assert "description" in info, f"{name} missing description"
            assert "format" in info, f"{name} missing format"

    def test_benchmark_corpora_are_immutable_and_verified(self):
        for name in BENCHMARK_CORPORA:
            info = CORPORA[name]
            files = info.get("files") or [info]
            for file_info in files:
                assert "/resolve/main/" not in file_info["url"], f"{name} uses mutable source URL"
                assert file_info.get("sha256"), f"{name} missing SHA256"

    def test_tuning_manifests_are_registered(self):
        expected = {
            "tuning-very-short",
            "tuning-short",
            "tuning-medium",
            "tuning-long",
            "tuning-very-long",
            "tuning-noisy",
        }
        assert expected.issubset(TUNING_MANIFESTS)


class TestWavDuration:
    def test_duration_1s(self, tmp_path):
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=1.0)
        dur = _get_wav_duration(wav)
        assert abs(dur - 1.0) < 0.01

    def test_duration_5s(self, tmp_path):
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_s=5.0)
        dur = _get_wav_duration(wav)
        assert abs(dur - 5.0) < 0.01


class TestBuildManifest:
    def test_builds_from_wavs(self, tmp_path):
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        for i in range(3):
            _make_wav(wav_dir / f"utt{i}.wav", duration_s=2.0)

        ref_file = tmp_path / "refs.tsv"
        ref_file.write_text("utt0\thello world\nutt1\tgoodbye world\n")

        manifest = _build_manifest(wav_dir, ref_file, "test-corpus")
        assert len(manifest) == 3
        assert manifest[0]["corpus"] == "test-corpus"
        assert manifest[0]["reference"] == "hello world"
        assert manifest[2].get("reference") is None
        assert abs(manifest[0]["duration_s"] - 2.0) < 0.01

    def test_build_manifest_respects_max_samples(self, tmp_path):
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        for i in range(5):
            _make_wav(wav_dir / f"utt{i}.wav", duration_s=1.0)

        ref_file = tmp_path / "refs.tsv"
        ref_file.write_text("\n".join(f"utt{i}\ttext {i}" for i in range(5)))

        manifest = _build_manifest(wav_dir, ref_file, "test-corpus", max_samples=2)

        assert [e["utt_id"] for e in manifest] == ["utt0", "utt1"]

    def test_empty_dir_returns_empty(self, tmp_path):
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        ref_file = tmp_path / "refs.tsv"
        ref_file.write_text("")
        manifest = _build_manifest(wav_dir, ref_file, "test")
        assert manifest == []


class TestLoadDatasetCache:
    def _seed_cache(self, tmp_path, corpus_name: str, wav_count: int):
        corpus_dir = tmp_path / corpus_name
        wav_dir = corpus_dir / "wav"
        wav_dir.mkdir(parents=True)
        for i in range(wav_count):
            (wav_dir / f"utt{i}.wav").touch()
        (corpus_dir / "references.tsv").write_text("utt0\thello\n")

    def test_full_dataset_requires_expected_file_count(self, tmp_path):
        self._seed_cache(tmp_path, "tiny", wav_count=2)
        corpus = {
            "url": "https://example.com/tiny.tar.gz",
            "description": "Tiny test corpus",
            "format": "librispeech",
            "expected_files": 3,
        }

        with patch.dict(CORPORA, {"tiny": corpus}), \
             patch("benchmarks.datasets.registry._download_file"), \
             patch("benchmarks.datasets.registry._extract_librispeech") as extract, \
             patch("benchmarks.datasets.registry._build_manifest", return_value=[]):
            load_dataset("tiny", cache_dir=tmp_path)

        extract.assert_called_once()

    def test_limited_dataset_reuses_cache_when_limit_is_satisfied(self, tmp_path):
        self._seed_cache(tmp_path, "tiny", wav_count=2)
        corpus = {
            "url": "https://example.com/tiny.tar.gz",
            "description": "Tiny test corpus",
            "format": "librispeech",
            "expected_files": 3,
        }

        with patch.dict(CORPORA, {"tiny": corpus}), \
             patch("benchmarks.datasets.registry._download_file"), \
             patch("benchmarks.datasets.registry._extract_librispeech") as extract, \
             patch("benchmarks.datasets.registry._build_manifest", return_value=[]):
            load_dataset("tiny", cache_dir=tmp_path, max_samples=2)

        extract.assert_not_called()

    def test_cached_download_is_sha256_verified(self, tmp_path):
        cached = tmp_path / "cached.bin"
        cached.write_bytes(b"bad-cache")
        expected = hashlib.sha256(b"expected").hexdigest()

        with pytest.raises(ValueError, match="SHA256 mismatch"):
            _download_file("https://example.com/cached.bin", cached, expected_sha256=expected)

        assert not cached.exists()

    def test_limited_dataset_expands_partial_cache_when_limit_increases(self, tmp_path):
        self._seed_cache(tmp_path, "tiny", wav_count=2)
        corpus = {
            "url": "https://example.com/tiny.tar.gz",
            "description": "Tiny test corpus",
            "format": "librispeech",
            "expected_files": 4,
        }

        with patch.dict(CORPORA, {"tiny": corpus}), \
             patch("benchmarks.datasets.registry._download_file"), \
             patch("benchmarks.datasets.registry._extract_librispeech") as extract, \
             patch("benchmarks.datasets.registry._build_manifest", return_value=[]):
            load_dataset("tiny", cache_dir=tmp_path, max_samples=3)

        extract.assert_called_once()

    def test_earnings22_format_dispatches_to_extractor(self, tmp_path):
        corpus = {
            "url": "https://example.com/earnings.parquet",
            "description": "Tiny Earnings-22 test corpus",
            "format": "earnings22",
            "expected_files": 1,
        }

        with patch.dict(CORPORA, {"tiny-earnings": corpus}), \
             patch("benchmarks.datasets.registry._download_file"), \
             patch("benchmarks.datasets.registry._extract_earnings22") as extract, \
             patch("benchmarks.datasets.registry._build_manifest", return_value=[]):
            load_dataset("tiny-earnings", cache_dir=tmp_path)

        extract.assert_called_once()

    def test_ami_format_dispatches_to_extractor(self, tmp_path):
        corpus = {
            "url": "https://example.com/ami.tar.gz",
            "description": "Tiny AMI test corpus",
            "format": "ami",
            "expected_files": 1,
        }

        with patch.dict(CORPORA, {"tiny-ami": corpus}), \
             patch("benchmarks.datasets.registry._download_file"), \
             patch("benchmarks.datasets.registry._extract_ami") as extract, \
             patch("benchmarks.datasets.registry._build_manifest", return_value=[]):
            load_dataset("tiny-ami", cache_dir=tmp_path)

        extract.assert_called_once()

    def test_prepared_tuning_manifest_is_loadable(self, tmp_path):
        manifest_dir = tmp_path / "tuning-short"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.json").write_text(
            '[{"utt_id": "utt0", "wav_path": "wav/utt0.wav", "duration_s": 2.0, "reference": "hello"}]'
        )

        manifest = load_dataset("tuning-short", cache_dir=tmp_path)

        assert manifest == [
            {
                "utt_id": "utt0",
                "wav_path": str(manifest_dir / "wav/utt0.wav"),
                "duration_s": 2.0,
                "reference": "hello",
                "corpus": "tuning-short",
            }
        ]

    def test_tuning_alias_combines_prepared_manifests(self, tmp_path):
        for name in TUNING_MANIFESTS:
            manifest_dir = tmp_path / name
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "manifest.json").write_text(
                f'[{{"utt_id": "{name}", "wav_path": "/tmp/{name}.wav", "duration_s": 1.0}}]'
            )

        manifest = load_dataset("tuning", cache_dir=tmp_path)

        assert len(manifest) == len(TUNING_MANIFESTS)
