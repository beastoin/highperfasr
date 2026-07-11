#!/usr/bin/env python3
"""Batch transcription client for HighPerfASR.

Uploads an audio file to the REST endpoint and prints the transcript.

Usage:
    python batch_client.py audio.wav
    python batch_client.py --url http://myserver:8000 audio.wav
    python batch_client.py --timestamps audio.wav
"""

import argparse
import json
import sys

import requests


def transcribe(url: str, file_path: str, timestamps: bool = False) -> dict:
    """Upload an audio file and return the transcription result."""
    endpoint = f"{url.rstrip('/')}/v1/transcriptions"
    params = {}
    if timestamps:
        params["timestamps"] = "true"

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f)}
        resp = requests.post(endpoint, files=files, params=params, timeout=120)

    if resp.status_code == 413:
        print("Error: file too large (server returned 413)", file=sys.stderr)
        sys.exit(1)
    elif resp.status_code == 415:
        print("Error: unsupported audio format (server returned 415)", file=sys.stderr)
        sys.exit(1)
    elif resp.status_code == 503:
        print("Error: server overloaded (server returned 503)", file=sys.stderr)
        sys.exit(1)
    elif resp.status_code != 200:
        print(f"Error: server returned {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Batch transcription client")
    parser.add_argument("file", help="Path to audio file (WAV, FLAC, MP3)")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Request word-level timestamps",
    )
    args = parser.parse_args()

    print(f"Uploading {args.file} to {args.url} ...")
    result = transcribe(args.url, args.file, timestamps=args.timestamps)
    print(f"\nTranscript: {result['text']}")

    if "words" in result:
        print("\nWord timestamps:")
        for w in result["words"]:
            print(f"  [{w['start']:.2f} - {w['end']:.2f}] {w['word']}")


if __name__ == "__main__":
    main()
