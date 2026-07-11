#!/usr/bin/env python3
"""Streaming transcription client — stream a WAV file over WebSocket."""

import argparse
import asyncio
import json
import struct
import wave

import websockets


async def stream_file(server: str, file_path: str, chunk_ms: int = 160):
    sample_rate = 16000
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2  # PCM16 = 2 bytes per sample

    with wave.open(file_path, "rb") as wf:
        assert wf.getsampwidth() == 2, "WAV must be 16-bit PCM"
        assert wf.getnchannels() == 1, "WAV must be mono"
        file_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())

    if file_rate != sample_rate:
        print(f"Warning: file is {file_rate} Hz, server expects {sample_rate} Hz")

    uri = f"{server}/v1/stream"
    async with websockets.connect(uri) as ws:
        # Wait for opened acknowledgment
        msg = json.loads(await ws.recv())
        stream_id = msg.get("stream_id", "?")
        print(f"Stream opened: {stream_id}")

        # Send audio in chunks
        offset = 0
        while offset < len(pcm_data):
            chunk = pcm_data[offset : offset + chunk_bytes]
            await ws.send(chunk)
            offset += chunk_bytes

            # Read any available transcript updates (non-blocking)
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    update = json.loads(raw)
                    partial = update.get("partial_transcript", "")
                    if partial:
                        print(f"  partial: {partial}", end="\r")
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

            # Pace at realtime to avoid overwhelming the server
            await asyncio.sleep(chunk_ms / 1000)

        # Signal end of audio
        await ws.send(json.dumps({"action": "close"}))

        # Collect final transcript
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("status") == "closed":
                print(f"\nFinal: {msg.get('final_text', '')}")
                break
            partial = msg.get("partial_transcript", "")
            if partial:
                print(f"  partial: {partial}", end="\r")


def main():
    parser = argparse.ArgumentParser(description="Streaming transcription client")
    parser.add_argument("file", help="WAV file to stream (16-bit PCM, mono)")
    parser.add_argument("--server", default="ws://localhost:8001", help="WebSocket server URL")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk duration in ms")
    args = parser.parse_args()

    asyncio.run(stream_file(args.server, args.file, args.chunk_ms))


if __name__ == "__main__":
    main()
