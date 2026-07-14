"""Tests for corpus registry — metadata, FLAC conversion, manifest building."""

import io
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.datasets.registry import CORPORA, _build_manifest, _get_wav_duration


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

    def test_empty_dir_returns_empty(self, tmp_path):
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        ref_file = tmp_path / "refs.tsv"
        ref_file.write_text("")
        manifest = _build_manifest(wav_dir, ref_file, "test")
        assert manifest == []
