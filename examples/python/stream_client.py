#!/usr/bin/env python3
"""Streaming transcription client for HighPerfASR.

Opens a WebSocket to the streaming endpoint, sends PCM16 audio chunks
from a WAV file, prints partial transcripts as they arrive, and prints
the final transcript on close.

Usage:
    python stream_client.py audio.wav
    python stream_client.py --url ws://myserver:8001 audio.wav
    python stream_client.py --chunk-ms 200 audio.wav
"""

import argparse
import asyncio
import json
import sys
import wave

import websockets


SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit PCM = 2 bytes per sample
DEFAULT_CHUNK_MS = 100  # 100ms chunks


def read_pcm16_from_wav(file_path: str) -> bytes:
    """Read a WAV file and return raw PCM16 mono bytes.

    Validates format and warns on mismatches.
    """
    with wave.open(file_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        if sample_width != SAMPLE_WIDTH:
            print(
                f"Warning: WAV is {sample_width * 8}-bit, expected 16-bit. "
                f"Audio will be sent as-is.",
                file=sys.stderr,
            )
        if channels != 1:
            print(
                f"Warning: WAV has {channels} channels, expected mono. "
                f"Audio will be sent as-is.",
                file=sys.stderr,
            )
        if framerate != SAMPLE_RATE:
            print(
                f"Warning: WAV sample rate is {framerate} Hz, expected {SAMPLE_RATE} Hz. "
                f"Audio will be sent as-is.",
                file=sys.stderr,
            )

        pcm_data = wf.readframes(n_frames)

    return pcm_data


async def stream_audio(url: str, file_path: str, chunk_ms: int):
    """Stream a WAV file over WebSocket and print transcripts."""
    pcm_data = read_pcm16_from_wav(file_path)
    chunk_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH * (chunk_ms / 1000))

    endpoint = f"{url.rstrip('/')}/v1/stream"
    print(f"Connecting to {endpoint} ...")

    async with websockets.connect(endpoint) as ws:
        # Step 1: Wait for the "opened" message
        msg = json.loads(await ws.recv())
        if msg.get("status") != "opened":
            print(f"Unexpected initial message: {msg}", file=sys.stderr)
            return
        stream_id = msg["stream_id"]
        print(f"Stream opened (id: {stream_id})")

        # Step 2: Send audio chunks while receiving partial transcripts
        async def send_audio():
            offset = 0
            total = len(pcm_data)
            while offset < total:
                chunk = pcm_data[offset : offset + chunk_bytes]
                await ws.send(chunk)
                offset += len(chunk)
                # Pace the send to approximate real-time playback
                await asyncio.sleep(chunk_ms / 1000)
            # Step 3: Signal end of audio
            await ws.send(json.dumps({"action": "close"}))

        async def recv_transcripts():
            while True:
                try:
                    raw = await ws.recv()
                except websockets.ConnectionClosed:
                    break
                msg = json.loads(raw)

                if "error" in msg:
                    print(f"\nError: {msg['error']}", file=sys.stderr)
                    break

                # Final close message
                if msg.get("status") == "closed":
                    print(f"\n\nFinal transcript: {msg['final_text']}")
                    break

                # Partial transcript update
                partial = msg.get("partial_transcript", "")
                final = msg.get("final_transcript", "")
                if final:
                    sys.stdout.write(f"\r[final]   {final}\n")
                if partial:
                    sys.stdout.write(f"\r[partial] {partial}")
                    sys.stdout.flush()

        # Run sender and receiver concurrently
        await asyncio.gather(send_audio(), recv_transcripts())

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Streaming transcription client")
    parser.add_argument("file", help="Path to WAV file (16 kHz, mono, PCM16)")
    parser.add_argument(
        "--url",
        default="ws://localhost:8001",
        help="WebSocket server URL (default: ws://localhost:8001)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help=f"Chunk duration in milliseconds (default: {DEFAULT_CHUNK_MS})",
    )
    args = parser.parse_args()

    asyncio.run(stream_audio(args.url, args.file, args.chunk_ms))


if __name__ == "__main__":
    main()
