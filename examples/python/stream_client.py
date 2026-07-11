#!/usr/bin/env python3
"""Stream a WAV file over WebSocket for real-time transcription.

Usage:
    python stream_client.py audio.wav
    python stream_client.py audio.wav --server ws://localhost:8001
    python stream_client.py audio.wav --chunk-ms 160

Requires: websockets (pip install websockets)

The server expects raw PCM16 mono 16kHz audio as binary WebSocket frames.
This script reads a WAV file and streams it in chunks, printing partial
and final transcripts as they arrive.
"""

import argparse
import json
import sys
import time
import wave

import websockets.sync.client as ws_sync


def stream_file(server: str, path: str, chunk_ms: int = 160):
    sample_rate = 16000
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample

    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            print(f"Error: expected 16-bit WAV, got {wf.getsampwidth() * 8}-bit", file=sys.stderr)
            sys.exit(1)
        if wf.getnchannels() != 1:
            print(f"Error: expected mono, got {wf.getnchannels()} channels", file=sys.stderr)
            sys.exit(1)
        if wf.getframerate() != sample_rate:
            print(f"Warning: expected {sample_rate}Hz, got {wf.getframerate()}Hz", file=sys.stderr)
        audio_data = wf.readframes(wf.getnframes())

    url = f"{server}/v1/stream"
    print(f"Connecting to {url} ...")
    with ws_sync.connect(url) as conn:
        ack = json.loads(conn.recv())
        if "error" in ack:
            print(f"Error: {ack['error']}", file=sys.stderr)
            sys.exit(1)
        stream_id = ack.get("stream_id", "unknown")
        print(f"Stream opened: {stream_id}")

        offset = 0
        chunk_count = 0
        confirmed = []
        while offset < len(audio_data):
            chunk = audio_data[offset : offset + chunk_bytes]
            conn.send(chunk)
            offset += chunk_bytes
            chunk_count += 1

            resp = json.loads(conn.recv())
            partial = resp.get("partial_transcript", "")
            final = resp.get("final_transcript", "")
            if final:
                confirmed.append(final)
                print(f"  [final] {final}")
            elif partial:
                print(f"  [partial] {partial}", end="\r")

            time.sleep(chunk_ms / 1000)

        conn.send(json.dumps({"action": "close"}))
        final_msg = json.loads(conn.recv())

        print()
        final_text = final_msg.get("final_text", "")
        if final_text:
            print(f"Transcript: {final_text}")
        elif confirmed:
            print(f"Transcript: {' '.join(confirmed)}")

        audio_seconds = len(audio_data) / sample_rate / 2
        print(f"Stats: {chunk_count} chunks, {audio_seconds:.1f}s audio")


def main():
    parser = argparse.ArgumentParser(description="WebSocket streaming transcription client")
    parser.add_argument("file", help="WAV file (16-bit mono 16kHz)")
    parser.add_argument("--server", default="ws://localhost:8001", help="Server URL")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk duration in ms")
    args = parser.parse_args()

    stream_file(args.server, args.file, args.chunk_ms)


if __name__ == "__main__":
    main()
