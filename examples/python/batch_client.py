#!/usr/bin/env python3
"""Upload an audio file for batch transcription.

Usage:
    python batch_client.py audio.wav
    python batch_client.py audio.wav --server http://localhost:8000 --timestamps

Requires: requests (pip install requests)
"""

import argparse
import json

import requests


def transcribe(server: str, path: str, timestamps: bool = False) -> dict:
    with open(path, "rb") as f:
        resp = requests.post(
            f"{server}/v1/transcriptions",
            files={"file": (path, f)},
            params={"timestamps": str(timestamps).lower()},
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Batch transcription client")
    parser.add_argument("file", help="Audio file (WAV, FLAC, MP3)")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--timestamps", action="store_true", help="Include word timestamps")
    args = parser.parse_args()

    result = transcribe(args.server, args.file, args.timestamps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
