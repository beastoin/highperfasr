#!/usr/bin/env python3
"""
Corpus registry and download infrastructure.

Each corpus defines a download URL, extraction logic, and WAV normalization.
All audio is converted to 16kHz mono PCM16 WAV during extraction.
"""

import io
import json
import logging
import os
import struct
import tarfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("datasets.registry")

DEFAULT_CACHE_DIR = Path(os.environ.get("HPFASR_DATASET_DIR", "/tmp/hpfasr-datasets"))


def _hf_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


EARNINGS22_REVISION = "0a034f9ed86d33a3859d9025d3e621cf243773ab"
AMI_REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"


CORPORA = {
    "librispeech-test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "sha256": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
        "description": "LibriSpeech test-clean — 2620 files, 5.4h, baseline WER reference",
        "format": "librispeech",
        "expected_files": 2620,
    },
    "librispeech-test-other": {
        "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
        "sha256": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
        "description": "LibriSpeech test-other — 2939 files, 5.3h, harder speakers/accents",
        "format": "librispeech",
        "expected_files": 2939,
    },
    "earnings22-full": {
        "url": _hf_url(
            "distil-whisper/earnings22",
            EARNINGS22_REVISION,
            "full/test-00000-of-00004-4541e1972fe3ed48.parquet",
        ),
        "sha256": "c21146efeb0ccf7e57d8947f3c56d5f0e5b8467da0c207c38ca6db9f6ca60fd5",
        "files": [
            {
                "url": _hf_url(
                    "distil-whisper/earnings22",
                    EARNINGS22_REVISION,
                    "full/test-00000-of-00004-4541e1972fe3ed48.parquet",
                ),
                "sha256": "c21146efeb0ccf7e57d8947f3c56d5f0e5b8467da0c207c38ca6db9f6ca60fd5",
            },
            {
                "url": _hf_url(
                    "distil-whisper/earnings22",
                    EARNINGS22_REVISION,
                    "full/test-00001-of-00004-e6d772bca8e23981.parquet",
                ),
                "sha256": "f1571d78ed15c41b9bb4ea5c1315a5c71a606d3a6bdb85b3118d2fc142e0a8dc",
            },
            {
                "url": _hf_url(
                    "distil-whisper/earnings22",
                    EARNINGS22_REVISION,
                    "full/test-00002-of-00004-25462b81a4cd9f09.parquet",
                ),
                "sha256": "3970867653b91c1a6872a6b35fc9e6e74a6ec52732e991ff5160377a8776637c",
            },
            {
                "url": _hf_url(
                    "distil-whisper/earnings22",
                    EARNINGS22_REVISION,
                    "full/test-00003-of-00004-514196049d554a43.parquet",
                ),
                "sha256": "9063b7779ea27556dcde854db828b8efca15483786e76d02164c021886d07cda",
            },
        ],
        "description": "Earnings-22 — 125 earnings calls, 119h, long-form batch benchmark",
        "format": "earnings22",
        "expected_files": 125,
        "tier": "benchmark",
    },
    "ami-eval-ihm": {
        "url": _hf_url("edinburghcstr/ami", AMI_REVISION, "ihm/test-00000-of-00004.parquet"),
        "sha256": "d95920dccc6924c15215239461bf5d1152fe07c9c61add073bd12da26dd602e0",
        "files": [
            {
                "url": _hf_url("edinburghcstr/ami", AMI_REVISION, "ihm/test-00000-of-00004.parquet"),
                "sha256": "d95920dccc6924c15215239461bf5d1152fe07c9c61add073bd12da26dd602e0",
            },
            {
                "url": _hf_url("edinburghcstr/ami", AMI_REVISION, "ihm/test-00001-of-00004.parquet"),
                "sha256": "07a2b1c407cf36bc1e66c1362b0248ed32c897edc2a116549d1dc6b456dc2325",
            },
            {
                "url": _hf_url("edinburghcstr/ami", AMI_REVISION, "ihm/test-00002-of-00004.parquet"),
                "sha256": "83cde21e55fc884295585327b0ee334cba47ad53bdb787dd5f97437067aca5f8",
            },
            {
                "url": _hf_url("edinburghcstr/ami", AMI_REVISION, "ihm/test-00003-of-00004.parquet"),
                "sha256": "3312bb79400ddef62d5a0097e124dfabe86a693b4643e7fef7d72a62588280d7",
            },
        ],
        "description": "AMI eval — 20 meetings, ~11h, close-talk headset mix, streaming benchmark",
        "format": "ami-parquet",
        "expected_files": 20,
        "tier": "benchmark",
    },
}

BENCHMARK_CORPORA = ["librispeech-test-clean", "librispeech-test-other", "earnings22-full", "ami-eval-ihm"]
TUNING_BUCKETS = {
    "very-short": {"duration_range": (1, 5), "sources": ["librispeech-train", "common-voice-en"], "count": 400},
    "short": {"duration_range": (5, 15), "sources": ["common-voice-en", "spgispeech"], "count": 600},
    "medium": {"duration_range": (15, 60), "sources": ["gigaspeech", "tedlium3"], "count": 400},
    "long": {"duration_range": (60, 300), "sources": ["tedlium3", "ami-train"], "count": 300},
    "very-long": {"duration_range": (300, 5700), "sources": ["earnings21"], "count": 44},
    "noisy": {"duration_range": (30, 300), "sources": ["chime6", "ami-farfield"], "count": 200},
}
TUNING_MANIFESTS = {
    f"tuning-{name}": {
        "description": f"Tuning manifest bucket: {name}",
        "duration_range": bucket["duration_range"],
        "sources": bucket["sources"],
        "count": bucket["count"],
        "tier": "tuning",
        "format": "manifest",
    }
    for name, bucket in TUNING_BUCKETS.items()
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


def _write_audio_bytes_as_wav(audio_bytes: bytes, wav_path: Path) -> None:
    """Decode encoded audio bytes and write normalized PCM16 WAV."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio, sr, subtype="PCM_16", format="WAV")


def _write_audio_array_as_wav(audio, sr: int, wav_path: Path) -> None:
    """Write an audio array to normalized PCM16 WAV."""
    import soundfile as sf

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio, sr, subtype="PCM_16", format="WAV")


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


def _extract_hf_audio_parquet(
    parquet_paths: Path | list[Path],
    wav_dir: Path,
    ref_file: Path,
    max_samples: int = 0,
    fallback_prefix: str = "sample",
) -> None:
    """Extract Hugging Face parquet audio rows into WAV + references TSV."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "parquet dataset extraction requires pyarrow. Install benchmark extras: pip install -e '.[bench]'"
        ) from exc

    wav_dir.mkdir(parents=True, exist_ok=True)
    paths = [parquet_paths] if isinstance(parquet_paths, Path) else list(parquet_paths)
    limit = max_samples if max_samples > 0 else 999999999
    refs = {}
    count = 0

    for parquet_path in paths:
        if count >= limit:
            break
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            if count >= limit:
                break
            audio = row.get("audio")
            if not audio:
                continue

            utt_id = row.get("id") or row.get("file") or row.get("path") or f"{fallback_prefix}_{count:04d}"
            utt_id = Path(str(utt_id)).stem
            wav_path = wav_dir / f"{utt_id}.wav"

            if not wav_path.exists():
                if isinstance(audio, dict) and audio.get("bytes") is not None:
                    _write_audio_bytes_as_wav(audio["bytes"], wav_path)
                elif isinstance(audio, dict) and audio.get("array") is not None:
                    _write_audio_array_as_wav(audio["array"], int(audio.get("sampling_rate", 16000)), wav_path)
                else:
                    raise ValueError(f"Unsupported parquet audio row shape for {utt_id}")
            count += 1

            ref = row.get("transcription") or row.get("text") or row.get("sentence") or row.get("normalized_text")
            if ref:
                refs[utt_id] = str(ref)

    with open(ref_file, "w") as f:
        for utt_id in sorted(refs):
            f.write(f"{utt_id}\t{refs[utt_id]}\n")

    log.info(f"Extracted {count} parquet WAV files, {len(refs)} references")


def _extract_earnings22(parquet_paths: Path | list[Path], wav_dir: Path, ref_file: Path, max_samples: int = 0):
    """Extract Hugging Face Earnings-22 parquet rows into WAV + references TSV."""
    _extract_hf_audio_parquet(parquet_paths, wav_dir, ref_file, max_samples, fallback_prefix="earnings22")


def _extract_ami(archive_paths: Path | list[Path], wav_dir: Path, ref_file: Path, max_samples: int = 0):
    """Extract AMI audio archive into flat WAV dir.

    Supports both the older archive shape and the current Hugging Face parquet
    shards for AMI IHM eval.
    """
    paths = [archive_paths] if isinstance(archive_paths, Path) else list(archive_paths)
    if paths and all(path.suffix == ".parquet" for path in paths):
        _extract_hf_audio_parquet(paths, wav_dir, ref_file, max_samples, fallback_prefix="ami")
        return

    wav_dir.mkdir(parents=True, exist_ok=True)
    limit = max_samples if max_samples > 0 else 999999
    count = 0

    for tar_path in paths:
        if count >= limit:
            break
        mode = "r:gz" if tar_path.suffixes[-2:] == [".tar", ".gz"] else "r:*"
        with tarfile.open(tar_path, mode) as tar:
            for member in tar:
                if count >= limit:
                    break
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix not in {".wav", ".flac"}:
                    continue
                extracted = tar.extractfile(member)
                if not extracted:
                    continue
                utt_id = Path(member.name).stem
                wav_path = wav_dir / f"{utt_id}.wav"
                if suffix == ".flac":
                    wav_bytes, _ = _flac_to_wav_bytes(extracted.read())
                    wav_path.write_bytes(wav_bytes)
                else:
                    _write_audio_bytes_as_wav(extracted.read(), wav_path)
                count += 1

    ref_file.write_text("")
    log.info(f"Extracted {count} AMI WAV files")


def _download_file(url: str, dest: Path, expected_sha256: str = None) -> Path:
    """Download a file with progress logging and optional SHA256 verification."""
    if dest.exists():
        log.info(f"Cached: {dest} ({dest.stat().st_size / 1e6:.0f}MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    log.info(f"Downloaded {dest.stat().st_size / 1e6:.0f}MB")
    if expected_sha256:
        import hashlib
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != expected_sha256:
            dest.unlink()
            raise ValueError(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")
        log.info(f"SHA256 verified: {actual[:16]}...")
    return dest


def _download_corpus_files(info: dict, corpus_dir: Path) -> list[Path]:
    """Download one or more files for a corpus entry."""
    if "files" not in info:
        archive_path = corpus_dir / Path(info["url"]).name
        return [_download_file(info["url"], archive_path, expected_sha256=info.get("sha256"))]

    paths = []
    for file_info in info["files"]:
        path = corpus_dir / Path(file_info["url"]).name
        paths.append(_download_file(file_info["url"], path, expected_sha256=file_info.get("sha256")))
    return paths


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


def _load_manifest_file(manifest_file: Path, corpus_name: str, max_samples: int = 0) -> list[dict]:
    """Load a prepared JSON manifest and normalize relative WAV paths."""
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"{corpus_name} is a prepared manifest dataset, but {manifest_file} does not exist. "
            "Create the tuning manifest first or choose a downloadable benchmark corpus."
        )

    with open(manifest_file) as f:
        manifest = json.load(f)
    if max_samples > 0:
        manifest = manifest[:max_samples]

    base = manifest_file.parent
    entries = []
    for entry in manifest:
        normalized = dict(entry)
        normalized.setdefault("corpus", corpus_name)
        wav_path = normalized.get("wav_path")
        if wav_path and not Path(wav_path).is_absolute():
            normalized["wav_path"] = str(base / wav_path)
        entries.append(normalized)
    return entries


def _available_dataset_names() -> list[str]:
    return sorted(list(CORPORA) + list(TUNING_MANIFESTS) + ["all", "benchmark", "tuning"])


def load_dataset(
    name: str,
    cache_dir: Path | None = None,
    max_samples: int = 0,
) -> list[dict]:
    """Download, extract, and return manifest for a dataset alias.

    Args:
        name: Corpus name from CORPORA, tuning manifest name, "benchmark", "tuning", or "all".
        cache_dir: Override default cache directory.
        max_samples: Max samples to extract per corpus. 0 = all.

    Returns:
        List of manifest entries with utt_id, wav_path, duration_s, corpus, reference.
    """
    base = cache_dir or DEFAULT_CACHE_DIR

    if name in {"all", "benchmark"}:
        manifest = []
        corpus_names = CORPORA if name == "all" else BENCHMARK_CORPORA
        for corpus_name in corpus_names:
            manifest.extend(load_dataset(corpus_name, cache_dir=base, max_samples=max_samples))
        return manifest

    if name == "tuning":
        manifest = []
        for tuning_name in TUNING_MANIFESTS:
            manifest.extend(load_dataset(tuning_name, cache_dir=base, max_samples=max_samples))
        return manifest

    if name in TUNING_MANIFESTS:
        manifest_file = base / name / "manifest.json"
        manifest = _load_manifest_file(manifest_file, name, max_samples=max_samples)
        log.info(f"{name}: {len(manifest)} prepared tuning entries")
        return manifest

    if name not in CORPORA:
        raise ValueError(f"Unknown dataset: {name}. Available: {_available_dataset_names()}")

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
        archive_paths = _download_corpus_files(info, corpus_dir)

        if info["format"] == "librispeech":
            _extract_librispeech(archive_paths[0], wav_dir, ref_file, max_samples)
        elif info["format"] == "earnings22":
            _extract_earnings22(archive_paths, wav_dir, ref_file, max_samples)
        elif info["format"] in {"ami", "ami-parquet"}:
            _extract_ami(archive_paths, wav_dir, ref_file, max_samples)
        else:
            raise ValueError(f"Unknown format: {info['format']}")

    manifest = _build_manifest(wav_dir, ref_file, name, max_samples=max_samples)
    log.info(f"{name}: {len(manifest)} entries, "
             f"{sum(e['duration_s'] for e in manifest) / 3600:.1f}h total")
    return manifest
