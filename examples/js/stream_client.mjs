#!/usr/bin/env node
/**
 * Stream a WAV file over WebSocket for real-time transcription.
 *
 * Usage:
 *   node stream_client.mjs audio.wav
 *   node stream_client.mjs audio.wav --server ws://localhost:8001
 *
 * Requires Node.js 18+ and the `ws` package: npm install ws
 */

import { readFileSync } from "fs";
import { WebSocket } from "ws";

const SAMPLE_RATE = 16000;
const CHUNK_MS = 160;
const CHUNK_BYTES = Math.floor((SAMPLE_RATE * CHUNK_MS) / 1000) * 2;

function parseWavData(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (String.fromCharCode(...buffer.slice(0, 4)) !== "RIFF") {
    throw new Error("Not a WAV file");
  }

  let offset = 12;
  while (offset < buffer.length - 8) {
    const id = String.fromCharCode(...buffer.slice(offset, offset + 4));
    const size = view.getUint32(offset + 4, true);
    if (id === "data") {
      return buffer.slice(offset + 8, offset + 8 + size);
    }
    offset += 8 + size;
  }
  throw new Error("No data chunk in WAV");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const args = process.argv.slice(2);
  let file = null;
  let server = "ws://localhost:8001";
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--server" && args[i + 1]) {
      server = args[++i];
    } else if (!file) {
      file = args[i];
    }
  }
  if (!file) {
    console.error("Usage: node stream_client.mjs <audio.wav> [--server ws://host:port]");
    process.exit(1);
  }

  const pcm = parseWavData(readFileSync(file));
  const audioSec = pcm.length / SAMPLE_RATE / 2;
  const url = `${server}/v1/stream`;
  console.log(`Audio: ${audioSec.toFixed(1)}s — connecting to ${url}`);

  const ws = new WebSocket(url);
  const messages = [];
  let resolve;

  ws.on("message", (data) => {
    const msg = JSON.parse(data.toString());
    if (resolve) {
      const r = resolve;
      resolve = null;
      r(msg);
    } else {
      messages.push(msg);
    }
  });

  function recv() {
    if (messages.length > 0) return Promise.resolve(messages.shift());
    return new Promise((r) => {
      resolve = r;
    });
  }

  await new Promise((r, reject) => {
    ws.on("open", r);
    ws.on("error", (e) => reject(new Error(`Connection failed: ${e.message}`)));
  });

  const ack = await recv();
  if (ack.error) {
    console.error(`Error: ${ack.error}`);
    ws.close();
    return;
  }
  console.log(`Stream opened: ${ack.stream_id}`);

  let offset = 0;
  let chunkCount = 0;
  const confirmed = [];

  while (offset < pcm.length) {
    const chunk = pcm.slice(offset, offset + CHUNK_BYTES);
    ws.send(chunk);
    offset += CHUNK_BYTES;
    chunkCount++;

    const resp = await recv();
    const partial = resp.partial_transcript || "";
    const final_ = resp.final_transcript || "";
    if (final_) {
      confirmed.push(final_);
      process.stdout.write(`\r  [final] ${final_}\n`);
    } else if (partial) {
      process.stdout.write(`\r  [partial] ${partial}`);
    }

    await sleep(CHUNK_MS);
  }

  ws.send(JSON.stringify({ action: "close" }));
  const finalMsg = await recv();
  const finalText = finalMsg.final_text || "";

  console.log();
  if (finalText) {
    console.log(`Transcript: ${finalText}`);
  } else if (confirmed.length) {
    console.log(`Transcript: ${confirmed.join(" ")}`);
  }
  console.log(`Stats: ${chunkCount} chunks, ${audioSec.toFixed(1)}s audio`);
  ws.close();
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
