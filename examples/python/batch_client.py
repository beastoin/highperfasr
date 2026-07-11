#!/usr/bin/env python3
"""Batch transcription client — upload a file, print the result."""

import argparse
import sys

import requests


def transcribe(server: str, file_path: str, timestamps: bool = False) -> dict:
    url = f"{server}/v1/transcriptions"
    params = {"timestamps": "true"} if timestamps else {}
    with open(file_path, "rb") as f:
        resp = requests.post(url, files={"file": f}, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Batch transcription client")
    parser.add_argument("file", help="Audio file to transcribe (WAV, FLAC, MP3)")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--timestamps", action="store_true", help="Include word timestamps")
    args = parser.parse_args()

    result = transcribe(args.server, args.file, args.timestamps)
    print(result.get("text", ""))
    if args.timestamps and "words" in result:
        for w in result["words"]:
            print(f"  [{w['start']:.2f} - {w['end']:.2f}] {w['word']}")


if __name__ == "__main__":
    main()
