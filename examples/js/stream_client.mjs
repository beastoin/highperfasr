#!/usr/bin/env node
/**
 * Streaming transcription client — stream a WAV file over WebSocket.
 *
 * Usage: node stream_client.mjs audio.wav [ws://localhost:8001]
 *
 * Requires: npm install ws
 */

import { readFileSync } from "fs";
import WebSocket from "ws";

const file = process.argv[2];
const server = process.argv[3] || "ws://localhost:8001";

if (!file) {
  console.error("Usage: node stream_client.mjs <audio.wav> [ws://server:port]");
  process.exit(1);
}

const wav = readFileSync(file);
// Skip WAV header (44 bytes) to get raw PCM data
const pcm = wav.subarray(44);

const CHUNK_MS = 160;
const SAMPLE_RATE = 16000;
const CHUNK_BYTES = Math.floor((SAMPLE_RATE * CHUNK_MS) / 1000) * 2;

const ws = new WebSocket(`${server}/v1/stream`);

ws.on("open", () => {
  console.log("Connected");
});

ws.on("message", (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.status === "opened") {
    console.log(`Stream opened: ${msg.stream_id}`);
    sendAudio();
  } else if (msg.status === "closed") {
    console.log(`\nFinal: ${msg.final_text || ""}`);
    ws.close();
  } else if (msg.partial_transcript) {
    process.stdout.write(`\r  partial: ${msg.partial_transcript}`);
  }
});

ws.on("error", (err) => {
  console.error("WebSocket error:", err.message);
  process.exit(1);
});

async function sendAudio() {
  let offset = 0;
  while (offset < pcm.length) {
    const chunk = pcm.subarray(offset, offset + CHUNK_BYTES);
    ws.send(chunk);
    offset += CHUNK_BYTES;
    await new Promise((r) => setTimeout(r, CHUNK_MS));
  }
  ws.send(JSON.stringify({ action: "close" }));
}
