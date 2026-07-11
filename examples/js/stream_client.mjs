#!/usr/bin/env node
/**
 * Stream a WAV file over WebSocket for real-time transcription.
 *
 * Usage:
 *   node stream_client.mjs audio.wav
 *   node stream_client.mjs audio.wav --url ws://localhost:8001/v1/stream
 *
 * Requires: npm install ws
 */

import { readFileSync } from "fs";
import { WebSocket } from "ws";

const SAMPLE_RATE = 16000;
const CHUNK_MS = 160;
const CHUNK_BYTES = Math.floor((SAMPLE_RATE * CHUNK_MS) / 1000) * 2; // 16-bit PCM

function parseWav(buffer) {
  // Minimal WAV parser — expects PCM16 mono 16kHz
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const riff = String.fromCharCode(...buffer.slice(0, 4));
  if (riff !== "RIFF") throw new Error("Not a WAV file");

  // Find 'data' chunk
  let offset = 12;
  while (offset < buffer.length - 8) {
    const id = String.fromCharCode(...buffer.slice(offset, offset + 4));
    const size = view.getUint32(offset + 4, true);
    if (id === "data") {
      return buffer.slice(offset + 8, offset + 8 + size);
    }
    offset += 8 + size;
  }
  throw new Error("No data chunk found in WAV");
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error("Usage: node stream_client.mjs <audio.wav> [--url ws://...]");
    process.exit(1);
  }

  const file = args[0];
  let url = "ws://localhost:8001/v1/stream";
  const urlIdx = args.indexOf("--url");
  if (urlIdx !== -1 && args[urlIdx + 1]) url = args[urlIdx + 1];

  const raw = readFileSync(file);
  const pcmData = parseWav(raw);
  console.log(`Audio: ${(pcmData.length / SAMPLE_RATE / 2).toFixed(1)}s`);
  console.log(`Connecting to ${url} ...`);

  const ws = new WebSocket(url);
  let chunkCount = 0;

  ws.on("open", () => {
    // Wait for ack before sending
  });

  ws.on("message", (data) => {
    const msg = JSON.parse(data.toString());

    if (msg.stream_id && chunkCount === 0) {
      // Ack received — start sending chunks
      console.log(`Stream opened: ${msg.stream_id}`);
      let offset = 0;
      const sendNext = () => {
        if (offset < pcmData.length) {
          const chunk = pcmData.slice(offset, offset + CHUNK_BYTES);
          ws.send(chunk);
          offset += CHUNK_BYTES;
          chunkCount++;
          // Don't send next until we get the response
        } else {
          // Done sending — close stream
          ws.send(JSON.stringify({ action: "close" }));
        }
      };
      // Store sender for use in message handler
      ws._sendNext = sendNext;
      sendNext();
      return;
    }

    // Partial or final result
    const text = msg.partial || msg.text || msg.transcript || "";
    if (msg.partial !== undefined) {
      process.stdout.write(`\r  [${chunkCount}] ${text}`);
      ws._sendNext?.();
    } else {
      // Final transcript (after close)
      console.log(`\n\nFinal transcript:\n  ${text}`);
      console.log(`\nStats: ${chunkCount} chunks sent`);
      ws.close();
    }
  });

  ws.on("error", (err) => {
    console.error("WebSocket error:", err.message);
    process.exit(1);
  });
}

main();
