#!/usr/bin/env python3
"""
Corpus registry and download infrastructure.

Each corpus defines a download URL, extraction logic, and WAV normalization.
All audio is converted to 16kHz mono PCM16 WAV during extraction.
"""

import io
import logging
import os
import struct
import tarfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("datasets.registry")

DEFAULT_CACHE_DIR = Path(os.environ.get("HPFASR_DATASET_DIR", "/tmp/hpfasr-datasets"))

CORPORA = {
    "librispeech-test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "description": "LibriSpeech test-clean — 2620 files, 5.4h, baseline WER reference",
        "format": "librispeech",
        "expected_files": 2620,
    },
    "librispeech-test-other": {
        "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
        "description": "LibriSpeech test-other — 2939 files, 5.3h, harder speakers/accents",
        "format": "librispeech",
        "expected_files": 2939,
    },
}


def _flac_to_wav_bytes(flac_bytes: bytes) -> tuple[bytes, int]:
    """Convert FLAC bytes to PCM16 WAV bytes. Returns (wav_bytes, sample_rate)."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(flac_bytes), dtype="int16")
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    num_samples = len(audio)
    data_size = num_samples * 2
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio.tobytes())
    return buf.getvalue(), sr


def _extract_librispeech(tar_path: Path, wav_dir: Path, ref_file: Path, max_samples: int = 0):
    """Extract LibriSpeech tar.gz into flat WAV dir + references TSV."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    refs = {}
    count = 0
    limit = max_samples if max_samples > 0 else 999999

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.name.endswith(".trans.txt"):
                f = tar.extractfile(member)
                if f:
                    for line in f.read().decode().strip().split("\n"):
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            refs[parts[0]] = parts[1]

            if member.name.endswith(".flac"):
                utt_id = Path(member.name).stem
                if count >= limit:
                    continue
                wav_path = wav_dir / f"{utt_id}.wav"
                if wav_path.exists():
                    count += 1
                    continue
                f = tar.extractfile(member)
                if f:
                    wav_bytes, _ = _flac_to_wav_bytes(f.read())
                    with open(wav_path, "wb") as wf:
                        wf.write(wav_bytes)
                    count += 1

    with open(ref_file, "w") as f:
        for utt_id in sorted(refs):
            f.write(f"{utt_id}\t{refs[utt_id]}\n")

    log.info(f"Extracted {count} WAV files, {len(refs)} references")


def _download_file(url: str, dest: Path) -> Path:
    """Download a file with progress logging."""
    if dest.exists():
        log.info(f"Cached: {dest} ({dest.stat().st_size / 1e6:.0f}MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    log.info(f"Downloaded {dest.stat().st_size / 1e6:.0f}MB")
    return dest


def _target_file_count(info: dict, max_samples: int) -> int:
    """Return the minimum cached WAV count needed for this request."""
    if max_samples > 0:
        return max_samples
    return int(info.get("expected_files", 0))


def _get_wav_duration(wav_path: Path) -> float:
    """Get WAV duration in seconds from header."""
    with open(wav_path, "rb") as f:
        f.read(24)
        sr = struct.unpack("<I", f.read(4))[0]
        f.read(8)
        data_marker = f.read(4)
        while data_marker != b"data":
            skip = struct.unpack("<I", f.read(4))[0]
            f.seek(skip, 1)
            data_marker = f.read(4)
        data_size = struct.unpack("<I", f.read(4))[0]
    return data_size / (sr * 2)


def _build_manifest(wav_dir: Path, ref_file: Path, corpus_name: str, max_samples: int = 0) -> list[dict]:
    """Build manifest entries from WAV dir + references TSV."""
    refs = {}
    if ref_file.exists():
        with open(ref_file) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    refs[parts[0]] = parts[1]

    entries = []
    wav_paths = sorted(wav_dir.glob("*.wav"))
    if max_samples > 0:
        wav_paths = wav_paths[:max_samples]

    for wav_path in wav_paths:
        utt_id = wav_path.stem
        dur = _get_wav_duration(wav_path)
        entry = {
            "utt_id": utt_id,
            "wav_path": str(wav_path),
            "duration_s": round(dur, 3),
            "corpus": corpus_name,
        }
        if utt_id in refs:
            entry["reference"] = refs[utt_id]
        entries.append(entry)

    return entries


def load_dataset(
    name: str,
    cache_dir: Path | None = None,
    max_samples: int = 0,
) -> list[dict]:
    """Download, extract, and return manifest for a corpus (or 'all').

    Args:
        name: Corpus name from CORPORA registry, or "all" for all corpora.
        cache_dir: Override default cache directory.
        max_samples: Max samples to extract per corpus. 0 = all.

    Returns:
        List of manifest entries with utt_id, wav_path, duration_s, corpus, reference.
    """
    base = cache_dir or DEFAULT_CACHE_DIR

    if name == "all":
        manifest = []
        for corpus_name in CORPORA:
            manifest.extend(load_dataset(corpus_name, cache_dir=base, max_samples=max_samples))
        return manifest

    if name not in CORPORA:
        raise ValueError(f"Unknown corpus: {name}. Available: {list(CORPORA.keys()) + ['all']}")

    info = CORPORA[name]
    corpus_dir = base / name
    wav_dir = corpus_dir / "wav"
    ref_file = corpus_dir / "references.tsv"
    manifest_file = corpus_dir / "manifest.json"

    existing_wavs = len(list(wav_dir.glob("*.wav"))) if wav_dir.exists() else 0
    target = _target_file_count(info, max_samples)

    if target > 0 and existing_wavs >= target and ref_file.exists():
        log.info(f"{name}: {existing_wavs} WAV files cached")
    else:
        archive_path = corpus_dir / Path(info["url"]).name
        _download_file(info["url"], archive_path)

        if info["format"] == "librispeech":
            _extract_librispeech(archive_path, wav_dir, ref_file, max_samples)
        else:
            raise ValueError(f"Unknown format: {info['format']}")

    manifest = _build_manifest(wav_dir, ref_file, name, max_samples=max_samples)
    log.info(f"{name}: {len(manifest)} entries, "
             f"{sum(e['duration_s'] for e in manifest) / 3600:.1f}h total")
    return manifest
