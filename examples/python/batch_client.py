#!/usr/bin/env python3
"""Upload an audio file for batch transcription.

Usage:
    python batch_client.py audio.wav
    python batch_client.py audio.wav --url http://localhost:8000 --timestamps
"""

import argparse
import json
import sys

import requests


def transcribe(url: str, path: str, timestamps: bool = False) -> dict:
    with open(path, "rb") as f:
        resp = requests.post(
            f"{url}/v1/transcriptions",
            files={"file": (path, f)},
            params={"timestamps": str(timestamps).lower()},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Batch transcription client")
    parser.add_argument("file", help="Path to audio file (WAV, FLAC, MP3, etc.)")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--timestamps", action="store_true", help="Include word timestamps")
    args = parser.parse_args()

    print(f"Uploading {args.file} to {args.url} ...")
    result = transcribe(args.url, args.file, args.timestamps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
