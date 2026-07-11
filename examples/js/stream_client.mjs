#!/usr/bin/env node
/**
 * Streaming transcription client for HighPerfASR (Node.js).
 *
 * Opens a WebSocket to the streaming endpoint, sends PCM16 audio chunks
 * from a WAV file, prints partial transcripts as they arrive, and prints
 * the final transcript on close.
 *
 * Dependencies: npm install ws
 *
 * Usage:
 *   node stream_client.mjs audio.wav
 *   node stream_client.mjs --url ws://myserver:8001 audio.wav
 *   node stream_client.mjs --chunk-ms 200 audio.wav
 */

import { readFileSync } from "fs";
import { basename } from "path";
import WebSocket from "ws";

const SAMPLE_RATE = 16000;
const SAMPLE_WIDTH = 2; // 16-bit PCM = 2 bytes per sample
const DEFAULT_CHUNK_MS = 100;

/**
 * Parse a WAV file and return the raw PCM data (skips the header).
 * Validates basic format properties and warns on mismatches.
 */
function readPcm16FromWav(filePath) {
  const buf = readFileSync(filePath);

  // Minimal WAV header parsing
  const riff = buf.toString("ascii", 0, 4);
  const wave = buf.toString("ascii", 8, 12);
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error(`${filePath} is not a valid WAV file`);
  }

  // Walk chunks to find "fmt " and "data"
  let offset = 12;
  let fmtFound = false;
  let channels, sampleRate, bitsPerSample, dataStart, dataSize;

  while (offset < buf.length - 8) {
    const chunkId = buf.toString("ascii", offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);

    if (chunkId === "fmt ") {
      const audioFormat = buf.readUInt16LE(offset + 8);
      channels = buf.readUInt16LE(offset + 10);
      sampleRate = buf.readUInt32LE(offset + 12);
      bitsPerSample = buf.readUInt16LE(offset + 22);
      fmtFound = true;

      if (audioFormat !== 1) {
        console.warn(`Warning: WAV audio format is ${audioFormat}, expected 1 (PCM)`);
      }
    } else if (chunkId === "data") {
      dataStart = offset + 8;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize;
  }

  if (!fmtFound) throw new Error("WAV file missing fmt chunk");
  if (dataStart === undefined) throw new Error("WAV file missing data chunk");

  if (channels !== 1) {
    console.warn(`Warning: WAV has ${channels} channels, expected mono`);
  }
  if (sampleRate !== SAMPLE_RATE) {
    console.warn(`Warning: WAV sample rate is ${sampleRate} Hz, expected ${SAMPLE_RATE} Hz`);
  }
  if (bitsPerSample !== 16) {
    console.warn(`Warning: WAV is ${bitsPerSample}-bit, expected 16-bit`);
  }

  return buf.subarray(dataStart, dataStart + dataSize);
}

/**
 * Sleep for the given number of milliseconds.
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Stream a WAV file over WebSocket and print transcripts.
 */
async function streamAudio(url, filePath, chunkMs) {
  const pcmData = readPcm16FromWav(filePath);
  const chunkBytes = Math.floor(SAMPLE_RATE * SAMPLE_WIDTH * (chunkMs / 1000));
  const endpoint = `${url.replace(/\/$/, "")}/v1/stream`;

  console.log(`Connecting to ${endpoint} ...`);

  return new Promise((resolve, reject) => {
    const ws = new WebSocket(endpoint);

    let opened = false;

    ws.on("message", (raw) => {
      const msg = JSON.parse(raw.toString());

      // Handle initial opened message
      if (!opened && msg.status === "opened") {
        opened = true;
        console.log(`Stream opened (id: ${msg.stream_id})`);

        // Start sending audio in the background
        (async () => {
          let offset = 0;
          while (offset < pcmData.length) {
            const chunk = pcmData.subarray(offset, offset + chunkBytes);
            ws.send(chunk);
            offset += chunk.length;
            await sleep(chunkMs);
          }
          // Signal end of audio
          ws.send(JSON.stringify({ action: "close" }));
        })();
        return;
      }

      // Error message
      if (msg.error) {
        console.error(`\nError: ${msg.error}`);
        ws.close();
        return;
      }

      // Final close message
      if (msg.status === "closed") {
        console.log(`\n\nFinal transcript: ${msg.final_text}`);
        ws.close();
        return;
      }

      // Partial transcript update
      const partial = msg.partial_transcript || "";
      const finalText = msg.final_transcript || "";
      if (finalText) {
        process.stdout.write(`\r[final]   ${finalText}\n`);
      }
      if (partial) {
        process.stdout.write(`\r[partial] ${partial}`);
      }
    });

    ws.on("close", () => {
      console.log("\nDone.");
      resolve();
    });

    ws.on("error", (err) => {
      console.error(`WebSocket error: ${err.message}`);
      reject(err);
    });
  });
}

// --- CLI argument parsing ---

function printUsage() {
  console.log(`Usage: node ${basename(process.argv[1])} [options] <file>

Options:
  --url <url>        WebSocket server URL (default: ws://localhost:8001)
  --chunk-ms <ms>    Chunk duration in ms (default: ${DEFAULT_CHUNK_MS})
  --help             Show this help message`);
}

function parseArgs() {
  const args = process.argv.slice(2);
  let url = "ws://localhost:8001";
  let chunkMs = DEFAULT_CHUNK_MS;
  let file = null;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--url":
        url = args[++i];
        break;
      case "--chunk-ms":
        chunkMs = parseInt(args[++i], 10);
        break;
      case "--help":
      case "-h":
        printUsage();
        process.exit(0);
        break;
      default:
        if (args[i].startsWith("-")) {
          console.error(`Unknown option: ${args[i]}`);
          printUsage();
          process.exit(1);
        }
        file = args[i];
    }
  }

  if (!file) {
    console.error("Error: no audio file specified");
    printUsage();
    process.exit(1);
  }

  return { url, chunkMs, file };
}

const { url, chunkMs, file } = parseArgs();
streamAudio(url, file, chunkMs).catch((err) => {
  console.error(err);
  process.exit(1);
});
