# C.R.A.T.E.R
### Cryptographically-secured Rover Autonomy, Telemetry & Environmental Reconnaissance

<p align="left">
  <img alt="status" src="https://img.shields.io/badge/status-active-success">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="framework" src="https://img.shields.io/badge/backend-FastAPI-009688">
  <img alt="crypto" src="https://img.shields.io/badge/crypto-ChaCha20--Poly1305-critical">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

**A simulated lunar rover platform combining real-time computer-vision perception, an authenticated encrypted telemetry link over a physically-realistic degraded network, and a self-learning intrusion detection system — built for ATEC'26 at ESPRIT.**

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Security Design](#security-design)
5. [Intrusion Detection & Reinforcement Learning](#intrusion-detection--reinforcement-learning)
6. [AI Mission Advisor](#ai-mission-advisor)
7. [Dashboard](#dashboard)
8. [Tech Stack](#tech-stack)
9. [Project Structure](#project-structure)
10. [Getting Started](#getting-started)
11. [Testing the Attack & Defense System](#testing-the-attack--defense-system)
12. [Running the Test Suite](#running-the-test-suite)
13. [Known Limitations](#known-limitations)
14. [Author](#author)

---

## Overview

A rover operating in a permanently shadowed lunar crater must detect hazards visually, report their position precisely, and do so over a radio link that is simultaneously **unreliable** (bursty packet loss, bandwidth-starved, high latency) and **hostile** (open to tampering and impersonation). C.R.A.T.E.R implements every layer of that problem end-to-end — no component is mocked:

- Real computer-vision detection pipeline (YOLOv8)
- Real coordinate geometry with distance-aware uncertainty
- Real authenticated encryption (ChaCha20-Poly1305)
- A genuinely simulated network (real UDP sockets, Gilbert-Elliott burst loss, token-bucket bandwidth limiting)
- A reinforcement-learning-driven intrusion response system that improves its own defense policy over time
- A live operations dashboard with an AI advisor grounded in local inference

---

## System Architecture

```
 World Generation → Perception (YOLOv8) → Localization → Telemetry Encoding (FEC/CRC8)
                                                                  │
                                                     ChaCha20-Poly1305 AEAD Encrypt
                                                                  │
                                                                  ▼
                                                   ┌───────────────────────────┐
                                                   │      Link Emulator        │
                                                   │  Real UDP · Gilbert-      │
                                                   │  Elliott loss · Token-    │
                                                   │  bucket bandwidth cap     │
                                                   └─────────────┬─────────────┘
                                                                 │
                              ┌──────────────────────────────────┤
                              │                                  │
                     ┌────────▼─────────┐              ┌──────────▼──────────┐
                     │  Intrusion Guard  │◀── attacks ──│  External Attacker  │
                     │  L1 Origin check  │              │  (tools/attack_demo)│
                     │  L2 AEAD verify   │              └──────────────────────┘
                     │  L3 CRC8 check    │
                     │  + Q-learning     │
                     │    block policy   │
                     └────────┬──────────┘
                              │
                              ▼
                    ┌───────────────────┐        ┌─────────────────────┐
                    │  Mission Control   │───────▶│   Local AI Advisor   │
                    │  Dashboard          │        │ TF-IDF + rules +     │
                    │  (FastAPI/WS/JS)    │        │  local LLM (Ollama)  │
                    └───────────────────┘        └─────────────────────┘
```

---

## Core Components

### Perception
Renders a fixed, seeded hazard field and runs **YOLOv8** object detection on the synthetic camera frame to identify rocks and craters.

| File | Responsibility |
|---|---|
| `src/perception.py` | World/hazard rendering |
| `src/yolo_detector.py` | YOLOv8 inference |

### Localization
Converts 2D pixel detections into real-world (x, y) coordinates using a pinhole camera model and the rover's live pose. Models **distance-proportional uncertainty growth**, matching real stereo-vision error characteristics.

| File | Responsibility |
|---|---|
| `src/localization.py` | Pixel → world-frame transform, uncertainty modeling |

### Telemetry Encoding
| File | Responsibility |
|---|---|
| `src/telemetry.py` | DELTA/FULL packet encoding, XOR Forward Error Correction (1 recoverable loss per 4-packet block), CRC8 integrity check |

---

## Security Design

| Property | Implementation |
|---|---|
| **Confidentiality + Authenticity** | ChaCha20-Poly1305 (AEAD) |
| **Key Derivation** | HKDF-SHA256 |
| **Packet Format** | `[12-byte nonce] [ciphertext] [16-byte Poly1305 tag]` |
| **Source** | `src/security.py` |

AEAD gives a single verification step that proves **both** that the data is unreadable without the key **and** that it was produced by the key-holder and not altered in transit. This is what makes tampering and impersonation collapse into the same detectable failure mode — forging either requires the secret mission key, which no attacker possesses.

---

## Intrusion Detection & Reinforcement Learning

`src/intrusion.py` implements a **3-layer verification pipeline** on every received packet:

| Layer | Checks | Nature |
|---|---|---|
| **L1 — Origin** | Source address matches the expected sender | Heuristic, first-pass filter |
| **L2 — AEAD** | Decryption succeeds, Poly1305 tag verifies | **Cryptographic — the real security gate** |
| **L3 — CRC8** | Payload integrity after decryption | Data-quality check |

**Response logic:**
- A sliding 10-second window tracks rejected packets per source
- 3 rejections from an off-path source within that window → **the source is blocked**
- Strikes persist across recoveries, escalating future response severity
- When a block timer expires, the source is **automatically readmitted** — fully self-healing, no operator action required

**What makes the response adaptive — `RLBlockPolicy`, a tabular Q-learning agent:**

| RL Element | Definition |
|---|---|
| **State** | Strike count for the offending source (clipped to 1–4+) |
| **Action space** | Block duration ∈ `{8s, 16s, 32s, 60s}` |
| **Reward** | `-1` if the attacker re-offends within a 6s grace window after release (block too short); up to `+1` if the source stays clear (block effective), with a slight efficiency bonus for shorter-but-sufficient durations |
| **Update rule** | `Q[s,a] ← Q[s,a] + α(reward + γ·max(Q[s′]) − Q[s,a])` |
| **Exploration** | ε-greedy — favors exploration early, shifts toward the best-known action as evidence accumulates |
| **Persistence** | Q-table serialized to `rl_policy.json`, surviving process restarts |

This replaces a fixed exponential-backoff rule with a policy that **learns the effective block duration for the observed attacker behavior**, rather than guessing a hardcoded constant.

---

## AI Mission Advisor

`src/dashboard/local_advisor.py` implements a lightweight, fully local Retrieval-Augmented Generation pipeline:

1. **Retrieval** — TF-IDF vectorization + cosine similarity over a curated 7-entry mission knowledge base
2. **Generation** — optional local inference via **Ollama** (`qwen2.5:3b`), grounded in the retrieved context and live pipeline telemetry
3. **Fallback** — a deterministic rule engine activates instantly if the LLM is unreachable or exceeds a 4-second timeout, so the dashboard never stalls

No cloud dependency, no external API key, fully operable offline — appropriate for a mission-control context.

---

## Dashboard

FastAPI + WebSocket backend, vanilla JS/Canvas frontend (`src/dashboard/`), organized into four live views:

| View | Purpose |
|---|---|
| **Mission View** | Camera feed, rover telemetry, hazard tracking |
| **Orbital View** | Global map, link statistics, AI advisor |
| **Security View** | Encryption/authentication posture, event history |
| **Intrusion Monitor** | Attacker source address, detection layer, block status, live Q-learning policy metrics |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | YOLOv8 |
| Numerics | NumPy, SciPy (Hungarian algorithm for detection tracking) |
| Cryptography | `cryptography` — ChaCha20-Poly1305, HKDF-SHA256 |
| Networking | Python `socket` (real UDP), `threading`, `heapq` scheduler |
| Backend | FastAPI, Uvicorn, WebSockets |
| Frontend | Vanilla JavaScript, HTML5 Canvas |
| AI Advisor | TF-IDF/cosine similarity, Ollama (`qwen2.5:3b`) |
| Intrusion Defense | Custom tabular Q-learning |
| Testing | Pytest |

---

## Project Structure

```
ATEC HACKATHON 2026/
├── run_pipeline.py              Main entry point
├── requirements.txt
├── rl_policy.json                Learned Q-table (auto-generated)
├── src/
│   ├── perception.py              Hazard field rendering
│   ├── yolo_detector.py           YOLOv8 inference
│   ├── localization.py            Pixel → world transform
│   ├── telemetry.py               Packet encoding, FEC, CRC8
│   ├── security.py                ChaCha20-Poly1305, HKDF
│   ├── link_sim.py                UDP link emulator
│   ├── intrusion.py               IntrusionGuard + RL block policy
│   └── dashboard/
│       ├── server.py              FastAPI app & API routes
│       ├── local_advisor.py       RAG advisor (TF-IDF + rules + LLM)
│       └── static/
│           ├── index.html         Main dashboard (4 views)
│           └── security.html      Standalone intrusion monitor
├── tools/
│   └── attack_demo.py             External attacker simulator
└── tests/                         Pytest suite
```

---

## Getting Started

### 1. Install dependencies
```bash
cd "ATEC HACKATHON 2026"
pip install -r requirements.txt
```

### 2. (Optional) Enable the local LLM advisor
```bash
# Install Ollama from https://ollama.com
ollama pull qwen2.5:3b
```
If unavailable, the advisor transparently falls back to its rule engine — no setup required for core functionality.

### 3. Launch the mission pipeline
```bash
python run_pipeline.py
```

### 4. Open Mission Control
Navigate to **http://127.0.0.1:8000**.

---

## Testing the Attack & Defense System

In a separate terminal, from the same project directory:

```bash
# Tamper attack — flips a bit in a validly-encrypted packet
python tools/attack_demo.py --mode bitflip --count 5

# Forgery attack — sends unauthenticated random bytes
python tools/attack_demo.py --mode forge --count 5
```

Switch to the **Intrusion Monitor** tab to observe, live: the attacker's source address, the detection layer that caught it, the active block countdown, and the Q-learning policy's decision/reward counters updating in real time.

---

## Running the Test Suite

```bash
pytest tests/
```

---

## Known Limitations

- **No replay protection** — a previously valid, unmodified packet resent later would still pass AEAD verification; a sequence-number or timestamp check would be required to close this
- **L1 origin filtering is a heuristic**, not a cryptographic guarantee — the enforceable security boundary is entirely at L2-AEAD
- **The Q-learning policy benefits from extended interaction** — a short demo session demonstrates the learning mechanism correctly rather than a fully converged policy
- The demonstration mission key is derived from a fixed passphrase; a production deployment would use provisioned, periodically rotated keys

---

## Author

Built solo for **ATEC'26 — ESPRIT** by **Salim**.
