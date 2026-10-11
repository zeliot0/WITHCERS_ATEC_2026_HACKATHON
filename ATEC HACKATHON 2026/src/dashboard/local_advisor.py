import json
import math
import os
import re
import urllib.request
from collections import Counter

KNOWLEDGE_BASE = [
    {
        "id": "ccsds_framing",
        "tags": "telemetry packet framing protocol overhead bandwidth",
        "text": (
            "CCSDS (Consultative Committee for Space Data Systems) telemetry "
            "framing is the real-world standard this project's binary "
            "FULL/DELTA packet split takes after: fixed-length frames, no "
            "length prefix needed, and the smallest possible header so "
            "every byte spent on overhead is a byte not spent on payload. "
            "On a bandwidth-starved link, packet overhead ratio matters as "
            "much as raw throughput."
        ),
    },
    {
        "id": "fec_heritage",
        "tags": "fec forward error correction parity loss recovery reed solomon voyager deep space",
        "text": (
            "Forward Error Correction has deep-space heritage going back to "
            "Voyager's Reed-Solomon coding: when a one-way light-time delay "
            "makes retransmission requests impractical, the receiver must "
            "be able to reconstruct lost data from redundancy sent "
            "alongside it. A single-parity XOR scheme (this project's "
            "choice) recovers exactly one erasure per coding block at very "
            "low overhead; a Reed-Solomon code trades more overhead for "
            "recovering multiple erasures per block -- the standard "
            "next step once XOR's one-loss-per-block ceiling is reached."
        ),
    },
    {
        "id": "dsn_link_budget",
        "tags": "link budget bandwidth latency loss degraded signal dsn deep space network",
        "text": (
            "Deep Space Network link budgets are planned around a worst-case "
            "signal margin, not an average one: latency, bandwidth, and "
            "packet loss are treated as a jointly degrading envelope, and "
            "operators plan for graceful degradation (lower-priority data "
            "dropped first) rather than a binary link-up/link-down model. "
            "A live loss-rate spike above roughly 50% is normally a sign "
            "either of a real outage or of a deliberately induced test "
            "condition, not of ordinary link noise."
        ),
    },
    {
        "id": "rover_hazard_doctrine",
        "tags": "hazard avoidance rover safety rock crater autonomy stop",
        "text": (
            "Planetary rover hazard-avoidance doctrine (as used on Mars "
            "rover missions) treats a newly detected, high-confidence "
            "nearby hazard as an immediate drive-stop condition rather "
            "than something to resolve later -- false positives cost time, "
            "but a missed hazard can end the mission. Hazard count and "
            "confidence trend are the two numbers operators watch most "
            "closely during an active traverse."
        ),
    },
    {
        "id": "radiation_effects",
        "tags": "solar flare radiation bit flip sensor noise electronics shielding",
        "text": (
            "Unshielded electronics in a high-radiation environment (a "
            "solar particle event, or prolonged exposure outside a "
            "planet's magnetosphere) see increased single-event upsets: "
            "bit flips in memory and transient sensor noise, exactly the "
            "kind of correlated salt-and-pepper corruption this project's "
            "flare-simulation models. Real missions respond with "
            "radiation-hardened hardware and redundant voting logic; here, "
            "the integrity check (CRC8) and FEC layer are what stand in "
            "for that defense at the software level."
        ),
    },
    {
        "id": "uncertainty_quantification",
        "tags": "uncertainty localization covariance position error confidence",
        "text": (
            "Standard practice in robotic localization is to report "
            "position estimates with an explicit uncertainty envelope "
            "rather than a bare point, because downstream planning "
            "decisions (how close to drive to a hazard) depend on how much "
            "to trust the estimate. An elongated uncertainty ellipse along "
            "the camera's line of sight versus across it reflects the real "
            "geometry of monocular ranging: depth is always less certain "
            "than lateral position."
        ),
    },
    {
        "id": "link_security",
        "tags": "encryption security integrity authentication tamper confidentiality key",
        "text": (
            "Mission telemetry links are typically secured with "
            "authenticated encryption so that a corrupted or tampered "
            "packet is rejected outright rather than silently accepted -- "
            "confidentiality and integrity are treated as one property "
            "(AEAD: Authenticated Encryption with Associated Data), not "
            "two separate mechanisms layered loosely together."
        ),
    },
]

_STOPWORDS = frozenset({"the", "a", "an", "is", "are", "to", "of", "and", "on", "in", "for", "this", "with", "as"})
_TOKEN_RE = re.compile(r"[a-z0-9]+")

QUERIES = {
    "flare": "radiation solar flare sensor noise bit flip electronics",
    "severe_no_fec": "link budget degraded severe loss bandwidth",
    "fec": "fec forward error correction parity recovery reed solomon",
    "severe": "link budget degraded severe loss bandwidth",
    "hazard": "hazard avoidance rover safety autonomy",
    "default": "telemetry packet framing overhead protocol",
}


def _tokenize(text):
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]


_N_DOCS = len(KNOWLEDGE_BASE)
_DOC_TOKENS = [_tokenize(e["tags"] + " " + e["text"]) for e in KNOWLEDGE_BASE]
_DOC_FREQ = Counter()
for _toks in _DOC_TOKENS:
    _DOC_FREQ.update(set(_toks))


def _tfidf(tokens):
    return {
        term: count * (math.log((_N_DOCS + 1) / (_DOC_FREQ.get(term, 0) + 1)) + 1.0)
        for term, count in Counter(tokens).items()
    }


def _norm(vec):
    return math.sqrt(sum(v * v for v in vec.values()))


_DOC_VECTORS = [_tfidf(t) for t in _DOC_TOKENS]
_DOC_NORMS = [_norm(v) for v in _DOC_VECTORS]


def _cosine(a, na, b, nb):
    if na == 0.0 or nb == 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[t] for t, v in a.items() if t in b) / (na * nb)


def retrieve(query, top_k=1):
    qvec = _tfidf(_tokenize(query))
    qnorm = _norm(qvec)
    scored = [
        (_cosine(qvec, qnorm, dv, dn), i)
        for i, (dv, dn) in enumerate(zip(_DOC_VECTORS, _DOC_NORMS))
    ]
    scored.sort(key=lambda p: p[0], reverse=True)
    return [KNOWLEDGE_BASE[i] for score, i in scored[:top_k] if score > 0]


def get_advice(stats):
    loss_rate = float(stats.get("loss_rate", 0.0) or 0.0)
    latency_ms = float(stats.get("latency_ms", 0.0) or 0.0)
    survival_rate = float(stats.get("survival_rate", 1.0) or 1.0)
    fec_recoveries = int(stats.get("fec_recoveries", 0) or 0)
    fec_enabled = bool(stats.get("fec_enabled", True))
    unique_hazards = int(stats.get("unique_hazards", 0) or 0)
    total_packets = int(stats.get("total_packets_sent", 0) or 0)
    flare_active = bool(stats.get("flare_active", False))

    if flare_active:
        key = "flare"
    elif loss_rate >= 0.3 and not fec_enabled:
        key = "severe_no_fec"
    elif fec_recoveries > 0:
        key = "fec"
    elif loss_rate >= 0.3:
        key = "severe"
    elif unique_hazards > 0:
        key = "hazard"
    else:
        key = "default"
    hits = retrieve(QUERIES[key])
    grounding = hits[0]["text"] if hits else ""

    survival_pct = f"{survival_rate * 100:.0f}"
    loss_pct = f"{loss_rate * 100:.0f}"
    fec_clause = f", with {fec_recoveries} packets reconstructed by FEC." if fec_enabled and fec_recoveries else "."

    if total_packets == 0:
        headline = "No telemetry observed yet -- mission has not started transmitting."
    elif flare_active:
        headline = (
            f"Solar flare event active: expect elevated sensor noise and bit-level "
            f"corruption. Current link survival is {survival_pct}%."
        )
    elif loss_rate >= 0.6:
        headline = f"Link severely degraded ({loss_pct}% configured loss, {latency_ms:.0f} ms latency). Survival is {survival_pct}%" + fec_clause
    elif loss_rate >= 0.3:
        headline = (
            f"Link degraded ({loss_pct}% loss). Survival holding at {survival_pct}%"
            + (f" thanks to {fec_recoveries} FEC recoveries." if fec_enabled and fec_recoveries else " -- FEC is currently disabled, so losses are not being recovered.")
        )
    else:
        headline = (
            f"Link nominal ({loss_pct}% loss, {latency_ms:.0f} ms latency). "
            f"Survival at {survival_pct}%, {unique_hazards} hazards tracked."
        )

    if not fec_enabled and loss_rate > 0.15:
        recommendation = " Recommendation: re-enable FEC -- at this loss rate it materially improves survival."
    elif flare_active:
        recommendation = " Recommendation: hold position until the flare clears; new detections are less reliable right now."
    elif loss_rate >= 0.6:
        recommendation = " Recommendation: prioritize high-confidence, near-range hazards only until the link recovers."
    else:
        recommendation = ""

    return f"{headline}{recommendation} ({grounding})"


OLLAMA_URL = os.environ.get("CRATER_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("CRATER_OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_S = float(os.environ.get("CRATER_OLLAMA_TIMEOUT", "4"))

_LLM_SYSTEM_PROMPT = (
    "You are the mission advisor for a lunar rover telemetry system. "
    "Reply with 2 to 3 short plain-English sentences. Use only the numbers "
    "and facts given to you; never invent figures. No markdown, no lists."
)


def _ask_ollama(assessment):
    prompt = (
        "Telemetry assessment (its numbers are authoritative, keep them exactly):\n"
        f"{assessment}\n\n"
        "Rewrite this as clear advice for the mission operator."
    )
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "system": _LLM_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 120},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip()[:600] or None


def get_advice_with_source(stats):
    rules_text = get_advice(stats)
    try:
        llm_text = _ask_ollama(rules_text)
    except (OSError, ValueError):
        llm_text = None
    if llm_text:
        return llm_text, "llm"
    return rules_text, "rules"
