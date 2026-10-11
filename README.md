# C.R.A.T.E.R
### Cryptographically-secured Rover Autonomy, Telemetry & Environmental Reconnaissance

<p align="left">
  <img alt="status" src="https://img.shields.io/badge/status-active-success">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="framework" src="https://img.shields.io/badge/backend-FastAPI-009688">
  <img alt="crypto" src="https://img.shields.io/badge/crypto-ChaCha20--Poly1305-critical">
  <img alt="rl" src="https://img.shields.io/badge/defense-Q--learning-orange">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

**A simulated lunar rover platform — built for ATEC'26 at ESPRIT — that doesn't just report a mission, it defends one: real computer-vision perception, authenticated encrypted telemetry over a physically-realistic degraded link, and a self-learning intrusion response system, all wired together end-to-end.**

---

## Table of Contents

1. [The Mission Problem](#1-the-mission-problem)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Problem → Cause → Solution, Subsystem by Subsystem](#3-problem--cause--solution-subsystem-by-subsystem)
   - [3.1 Perceiving hazards](#31-perceiving-hazards)
   - [3.2 Knowing where a hazard actually is](#32-knowing-where-a-hazard-actually-is)
   - [3.3 Getting data home over a hostile, degraded link](#33-getting-data-home-over-a-hostile-degraded-link)
   - [3.4 Trusting what arrives](#34-trusting-what-arrives)
   - [3.5 Responding to an attacker, not just detecting one](#35-responding-to-an-attacker-not-just-detecting-one)
   - [3.6 Giving the operator a second opinion](#36-giving-the-operator-a-second-opinion)
   - [3.7 Seeing all of this, live](#37-seeing-all-of-this-live)
4. [Tech Stack](#4-tech-stack)
5. [Project Structure](#5-project-structure)
6. [Getting Started](#6-getting-started)
7. [Testing the Attack & Defense System](#7-testing-the-attack--defense-system)
8. [Running the Test Suite](#8-running-the-test-suite)
9. [Known Limitations & Honest Trade-offs](#9-known-limitations--honest-trade-offs)
10. [Author](#10-author)

---

## 1. The Mission Problem

A rover working inside a **permanently shadowed lunar crater** — the kind of site worth exploring precisely because it may hold water ice — has to operate somewhere that makes everything harder at once:

- **No direct line of sight** to Earth for long stretches → relayed, high-latency, bandwidth-starved communication
- **Radio shadowing and occlusion** from crater walls → packet loss that comes in *bursts*, not a steady trickle
- **An open RF link** → anyone who can transmit on the right frequency can inject traffic, whether by accident (interference) or on purpose (spoofing, jamming, replay)
- **No real-time human operator** watching every packet → the system has to decide for itself, in seconds, whether to trust what just arrived and what to do about it if it shouldn't

C.R.A.T.E.R is a from-scratch simulation of that entire stack: a rover perceives hazards, reports them home over a link that behaves like the real thing (not a mocked one), and defends that link against an active attacker — automatically, and adaptively.

---

## 2. Architecture at a Glance

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

## 3. Problem → Cause → Solution, Subsystem by Subsystem

### 3.1 Perceiving hazards

**Problem.** The rover needs to know a rock or crater is in its path before it drives into it.

**Cause.** Raw camera frames are just pixels — there's no hazard information until something interprets them, and that interpretation has to happen fast enough to matter for navigation.

**Solution.** `src/perception.py` renders a camera view of a fixed, seeded hazard field, and `src/yolo_detector.py` runs **YOLOv8**, a convolutional neural network object detector, on every frame to identify rocks and craters in real time.

---

### 3.2 Knowing where a hazard actually is

**Problem.** A detection in a 2D image isn't useful on its own — mission control needs a real-world (x, y) position, and it needs to know *how much to trust it*.

**Cause.** Converting pixels to world coordinates requires the camera's geometry and the rover's current pose, and the further away something is, the less precisely a camera can localize it — error doesn't stay constant with distance, it grows.

**Solution.** `src/localization.py` applies a **pinhole camera model** to transform pixel detections into the landing-site reference frame, and explicitly models **uncertainty that grows with distance**, so a far-off detection is reported with a wider error margin instead of false precision.

---

### 3.3 Getting data home over a hostile, degraded link

**Problem.** The link between rover and mission control isn't clean. Packets get lost, often in bursts rather than evenly; bandwidth is severely capped; and naively retransmitting everything would blow through that budget.

**Cause.** A permanently shadowed crater means intermittent line-of-sight and radio shadowing — this isn't a corner case, it's the normal operating condition. Most demo systems fake this with a simple "drop X% of messages" list, which doesn't produce the bursty, timing-sensitive behavior a real link has.

**Solution.**
- `src/link_sim.py` runs **real UDP sockets** across three roles (sender → emulator → receiver) — an actual network path, not a Python list standing in for one.
- Loss is modeled with a **Gilbert-Elliott burst-loss model**: a 2-state Markov chain (GOOD/BAD) tuned so the long-run average loss rate matches a configurable target, but losses cluster into realistic bursts.
- A **token-bucket bandwidth cap** drops packets that exceed the current budget, mirroring a saturated downlink with no infinite buffer.
- `src/telemetry.py` adds **XOR Forward Error Correction**: every block of 4 packets carries parity, so 1 lost packet per block is reconstructed without asking for a retransmit — critical when round-trip latency alone can be hundreds of milliseconds.
- A lightweight **CRC8** catches corruption in payloads that otherwise decoded fine.

---

### 3.4 Trusting what arrives

**Problem.** Even if a packet arrives intact, how does the receiver know it actually came from the rover — and wasn't quietly modified in transit by someone else on the link?

**Cause.** An open radio link has no inherent sender verification. Anyone transmitting on the right frequency can inject a packet that *looks* like valid telemetry, or intercept and alter a real one — without encryption-with-authentication, the receiver has no way to tell.

**Solution.** `src/security.py` encrypts every packet with **ChaCha20-Poly1305**, an AEAD (Authenticated Encryption with Associated Data) cipher:
- **ChaCha20** encrypts the payload (confidentiality)
- **Poly1305** produces a 16-byte tag that proves the ciphertext came from whoever holds the secret key and wasn't altered (authenticity) — `[12-byte nonce][ciphertext][16-byte tag]`
- The key itself comes from **HKDF-SHA256**, deriving a proper 256-bit key from a mission passphrase

The key insight: one AEAD check gives you both guarantees at once. That's what makes **tampering and impersonation collapse into the same failure mode** — forging either requires the secret key, which no attacker has, so both are caught at the same point in the pipeline with the same mechanism.

---

### 3.5 Responding to an attacker, not just detecting one

**Problem.** Detecting a bad packet is only half the job. If the system just silently drops forged packets forever, an attacker can keep hammering the link indefinitely at no cost to themselves, wasting bandwidth and CPU on every single attempt. The system needs to actively respond — and a fixed, hardcoded response (e.g. "always block for exactly 8 seconds") is either too short to matter against a persistent attacker or too long and wastes availability against a one-off glitch.

**Cause.** The "right" block duration depends on the attacker's actual behavior, which isn't known in advance and can change. A static rule can't adapt to that; it can only guess.

**Solution.** `src/intrusion.py` implements two layers of defense:

**Detection — `IntrusionGuard`, a 3-layer pipeline on every incoming packet:**

| Layer | Checks | Nature |
|---|---|---|
| L1 — Origin | Source address matches the expected sender | Heuristic, cheap first filter |
| L2 — AEAD | Decryption + Poly1305 tag verification | **Cryptographic — the real security gate** |
| L3 — CRC8 | Payload integrity after decryption | Data-quality check |

Three rejected packets from an off-path source within a 10-second window triggers a block. Strikes persist across recoveries, escalating future severity. When a block expires, the source is **automatically readmitted** — fully self-healing, no operator action required.

**Response — `RLBlockPolicy`, a tabular Q-learning agent that decides block duration:**

| Element | Definition |
|---|---|
| State | Strike count for the offending source (1–4+) |
| Action space | Block duration ∈ `{8s, 16s, 32s, 60s}` |
| Reward | `-1` if the attacker re-offends within a 6s grace window after release (block was too short); up to `+1` if they stay away (block worked), with a bonus for using the shortest sufficient duration |
| Update rule | `Q[s,a] ← Q[s,a] + α(reward + γ·max(Q[s′]) − Q[s,a])` |
| Exploration | ε-greedy — random early on, shifting to the best-known action as evidence accumulates |
| Persistence | Q-table saved to `rl_policy.json`, surviving restarts |

Instead of guessing a constant, the policy **learns the effective block duration for the attacker it's actually seeing**, and keeps that knowledge across runs.

---

### 3.6 Giving the operator a second opinion

**Problem.** Raw telemetry numbers (loss rate, latency, hazard counts) are hard to interpret quickly under pressure, and a mission context means there may be **no internet connection** to call out to for help.

**Cause.** A cloud-hosted AI advisor is a single point of failure the moment connectivity drops — exactly when an operator might need guidance most.

**Solution.** `src/dashboard/local_advisor.py` runs a fully local RAG pipeline:
1. **TF-IDF + cosine similarity** retrieves the most relevant entry from a curated mission knowledge base
2. An optional **local LLM** (`qwen2.5:3b` via **Ollama**, on the operator's own GPU) turns that context + live stats into natural-language advice
3. A **deterministic rule engine** takes over instantly if the LLM is unreachable or exceeds a 4-second timeout — the dashboard never stalls waiting on inference

No cloud dependency, no API key, fully operable offline.

---

### 3.7 Seeing all of this, live

**Problem.** None of the above matters to a human operator if it's buried in logs.

**Cause.** Mission-critical state (link health, security posture, who's attacking and from where) needs to be legible at a glance, in real time, without requiring the operator to parse JSON.

**Solution.** `src/dashboard/` — a FastAPI + WebSocket backend with a vanilla JS/Canvas frontend, four live views:

| View | Purpose |
|---|---|
| Mission View | Camera feed, rover telemetry, hazard tracking |
| Orbital View | Global map, link statistics, AI advisor |
| Security View | Encryption/authentication posture, event history |
| Intrusion Monitor | Attacker source address, detection layer, block status, live Q-learning policy metrics |

---

## 4. Tech Stack

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

## 5. Project Structure

```
ATEC HACKATHON 2026/
├── run_pipeline.py              Main entry point
├── requirements.txt
├── rl_policy.json                Learned Q-table (auto-generated)
├── src/
│   ├── perception.py              Hazard field rendering
│   ├── yolo_detector.py           YOLOv8 inference
│   ├── localization.py            Pixel → world transform, uncertainty modeling
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

## 6. Getting Started

### 6.1 Install dependencies
```bash
cd "ATEC HACKATHON 2026"
pip install -r requirements.txt
```

### 6.2 (Optional) Enable the local LLM advisor
```bash
# Install Ollama from https://ollama.com
ollama pull qwen2.5:3b
```
If unavailable, the advisor transparently falls back to its rule engine — no setup required for core functionality.

### 6.3 Launch the mission pipeline
```bash
python run_pipeline.py
```

### 6.4 Open Mission Control
Navigate to **http://127.0.0.1:8000**.

---

## 7. Testing the Attack & Defense System

In a separate terminal, from the same project directory:

```bash
# Tamper attack — flips a bit in a validly-encrypted packet
python tools/attack_demo.py --mode bitflip --count 5

# Forgery attack — sends unauthenticated random bytes
python tools/attack_demo.py --mode forge --count 5
```

Switch to the **Intrusion Monitor** tab to watch, live: the attacker's source address, the layer that caught it, the active block countdown, and the Q-learning policy's decision/reward counters updating.

---

## 8. Running the Test Suite

```bash
pytest tests/
```

---

## 9. Known Limitations & Honest Trade-offs

Being upfront about what's *not* solved is part of the engineering, not an afterthought:

- **No replay protection.** A captured, unmodified, validly-encrypted packet resent later would still pass AEAD verification — AEAD proves authenticity and integrity, not freshness. Closing this requires a sequence number or timestamp check, which isn't implemented.
- **L1 origin filtering is a heuristic, not a cryptographic guarantee.** The enforceable security boundary is entirely at L2-AEAD; L1 is a cheap early filter that could, in principle, be spoofed.
- **The Q-learning policy improves with more interaction.** In a short demo session it demonstrates the learning mechanism correctly — state/action/reward loop, persistence across restarts — rather than arriving at a fully converged, optimal policy.
- **The demo mission key is derived from a fixed passphrase** for reproducibility. A production deployment would provision and periodically rotate real key material through a pre-launch ground exchange, not hardcode it.
- **Six pre-existing test failures** in `tests/` (a `telemetry.py` signature mismatch and one localization uncertainty test) predate this security/RL work and were not introduced by it, but remain unresolved.

---

## 10. Author

Built solo for **ATEC'26 — ESPRIT** by **Salim**.
