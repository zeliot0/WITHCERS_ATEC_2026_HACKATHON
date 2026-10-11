"""
Lossy-link emulator for Project C.R.A.T.E.R.

Real UDP sockets, three independent roles, all on localhost:

    sender  --UDP-->  emulator  --UDP-->  receiver

The emulator is the stand-in for "regolith-absorbed radio through a
permanently shadowed crater": it applies packet loss, latency + jitter, and
a hard bandwidth cap (token bucket) between sender and receiver. All three
parameters are mutable at runtime (see set_config) specifically so the
Mission Control dashboard can crank packet loss to 90% live during a demo
and show the system degrade honestly instead of hanging or crashing.

This is deliberately *not* faked by just dropping entries from a Python
list -- using real sockets means the "network" the telemetry travels over
behaves like a network (reordering, real timing, a real receive loop),
which is the point when a jury is asked to evaluate protocol robustness.
"""
from __future__ import annotations

import heapq
import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class LinkConfig:
    loss_rate: float = 0.15       # probability a packet is dropped in the emulator
    latency_ms: float = 150.0     # base one-way latency
    jitter_ms: float = 60.0       # +- random jitter added to latency
    bandwidth_bps: float = 2000.0  # bytes/sec cap (deep-space-link-ish, not literal kbps radio)


@dataclass
class LinkStats:
    sent: int = 0
    dropped_loss: int = 0
    dropped_bandwidth: int = 0
    delivered: int = 0
    bytes_sent: int = 0
    bytes_delivered: int = 0

    def as_dict(self):
        total_attempted = self.sent
        survival = (self.delivered / total_attempted) if total_attempted else 1.0
        return {
            "sent": self.sent,
            "dropped_loss": self.dropped_loss,
            "dropped_bandwidth": self.dropped_bandwidth,
            "delivered": self.delivered,
            "bytes_sent": self.bytes_sent,
            "bytes_delivered": self.bytes_delivered,
            "survival_rate": round(survival, 4),
        }


class _TokenBucket:
    """Simple bandwidth cap: bytes trickle in at bandwidth_bps, packets that
    would exceed the current budget are dropped (mirrors a saturated,
    severely bandwidth-constrained downlink -- there is no infinite buffer
    on a real rover radio either)."""

    def __init__(self, rate_bps: float, burst_bytes: float = 512):
        self.rate = rate_bps
        self.capacity = max(burst_bytes, rate_bps * 0.25)
        self.tokens = self.capacity
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def set_rate(self, rate_bps: float):
        with self._lock:
            self.rate = rate_bps
            self.capacity = max(512.0, rate_bps * 0.25)

    def consume(self, n_bytes: int) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= n_bytes:
                self.tokens -= n_bytes
                return True
            return False


class LinkEmulator:
    """Owns three UDP sockets bound to localhost and runs the forwarding
    loop in a background thread. Call start()/stop(). send() is how the
    pipeline hands a packet to the sender socket; register a callback with
    on_receive() to get decoded bytes out the other end."""

    def __init__(self, config: Optional[LinkConfig] = None):
        self.config = config or LinkConfig()
        self.stats = LinkStats()
        self._bucket = _TokenBucket(self.config.bandwidth_bps)
        self._lock = threading.Lock()

        self._sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sender_sock.bind(("127.0.0.1", 0))
        self._sender_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

        self._emulator_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._emulator_sock.bind(("127.0.0.1", 0))
        self._emulator_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._emulator_addr = self._emulator_sock.getsockname()

        self._receiver_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._receiver_sock.bind(("127.0.0.1", int(os.environ.get("CRATER_RECEIVER_PORT", "47001"))))
        self._receiver_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._receiver_addr = self._receiver_sock.getsockname()

        self._on_receive: Optional[Callable[[bytes], None]] = None
        self._on_receive_src: Optional[Callable[[bytes, tuple], None]] = None
        self._running = False
        self._threads: list[threading.Thread] = []

        # Single min-heap timer queue + one dispatcher thread, instead of a
        # threading.Timer per packet: under a burst of hundreds of packets,
        # spawning hundreds of OS threads causes real scheduling-induced
        # drops that would be misread as "protocol loss". One scheduler
        # thread scales cleanly and is also the architecture a real
        # deadline-scheduled downlink would use.
        self._pending: list[tuple[float, bytes]] = []
        self._pending_lock = threading.Lock()

    # ---- config (mutable live, for the "crank it up" demo control) -------
    def set_config(self, *, loss_rate=None, latency_ms=None, jitter_ms=None, bandwidth_bps=None):
        with self._lock:
            if loss_rate is not None:
                self.config.loss_rate = max(0.0, min(1.0, loss_rate))
            if latency_ms is not None:
                self.config.latency_ms = max(0.0, latency_ms)
            if jitter_ms is not None:
                self.config.jitter_ms = max(0.0, jitter_ms)
            if bandwidth_bps is not None:
                self.config.bandwidth_bps = max(1.0, bandwidth_bps)
                self._bucket.set_rate(self.config.bandwidth_bps)

    def get_config(self) -> dict:
        with self._lock:
            return {
                "loss_rate": self.config.loss_rate,
                "latency_ms": self.config.latency_ms,
                "jitter_ms": self.config.jitter_ms,
                "bandwidth_bps": self.config.bandwidth_bps,
            }

    def on_receive(self, callback: Callable[[bytes], None]):
        self._on_receive = callback

    def on_receive_src(self, callback: Callable[[bytes, tuple], None]):
        self._on_receive_src = callback

    @property
    def forward_addr(self) -> tuple:
        return self._emulator_sock.getsockname()

    # ---- lifecycle ---------------------------------------------------
    def start(self):
        self._running = True
        t_emu = threading.Thread(target=self._emulator_loop, daemon=True)
        t_disp = threading.Thread(target=self._dispatcher_loop, daemon=True)
        t_recv = threading.Thread(target=self._receiver_loop, daemon=True)
        t_emu.start()
        t_disp.start()
        t_recv.start()
        self._threads = [t_emu, t_disp, t_recv]

    def stop(self):
        self._running = False
        for s in (self._sender_sock, self._emulator_sock, self._receiver_sock):
            try:
                s.close()
            except OSError:
                pass

    # ---- sending -------------------------------------------------------
    def send(self, packet: bytes):
        self.stats.sent += 1
        self.stats.bytes_sent += len(packet)
        self._sender_sock.sendto(packet, self._emulator_addr)

    # ---- internal loops --------------------------------------------------
    def _emulator_loop(self):
        self._emulator_sock.settimeout(0.5)
        while self._running:
            try:
                data, _addr = self._emulator_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            with self._lock:
                target_loss_rate = self.config.loss_rate
                latency_ms = self.config.latency_ms
                jitter_ms = self.config.jitter_ms

            # Gilbert-Elliott Burst Loss Model
            # Link toggles between GOOD (0% loss) and BAD (100% loss) states.
            # We tune the transition probabilities so the *average* loss rate
            # matches the target, but losses happen in heavy bursts.
            if not hasattr(self, "_ge_bad_state"):
                self._ge_bad_state = False

            # Fixed expected burst length: say, 20 packets (heavy occlusion)
            p_b_to_g = 1.0 / 20.0
            
            # If target loss rate is 0, clamp cleanly
            if target_loss_rate <= 0.001:
                self._ge_bad_state = False
                p_drop = 0.0
            elif target_loss_rate >= 0.999:
                self._ge_bad_state = True
                p_drop = 1.0
            else:
                # Solve for P(g -> b) to maintain the average loss rate:
                # L = P(g->b) / (P(g->b) + P(b->g))
                p_g_to_b = (target_loss_rate * p_b_to_g) / (1.0 - target_loss_rate)
                
                # State transition
                if self._ge_bad_state:
                    if random.random() < p_b_to_g:
                        self._ge_bad_state = False
                else:
                    if random.random() < p_g_to_b:
                        self._ge_bad_state = True
                
                p_drop = 1.0 if self._ge_bad_state else 0.0

            if random.random() < p_drop:
                self.stats.dropped_loss += 1
                continue

            if not self._bucket.consume(len(data)):
                self.stats.dropped_bandwidth += 1
                continue

            delay = max(0.0, (latency_ms + random.uniform(-jitter_ms, jitter_ms))) / 1000.0
            send_at = time.monotonic() + delay
            with self._pending_lock:
                heapq.heappush(self._pending, (send_at, data))

    def _dispatcher_loop(self):
        while self._running:
            now = time.monotonic()
            ready = []
            with self._pending_lock:
                while self._pending and self._pending[0][0] <= now:
                    _, data = heapq.heappop(self._pending)
                    ready.append(data)
            for data in ready:
                self._forward(data)
            time.sleep(0.002)

    def _forward(self, data: bytes):
        try:
            self._emulator_sock.sendto(data, self._receiver_addr)
        except OSError:
            pass

    def _receiver_loop(self):
        self._receiver_sock.settimeout(0.5)
        while self._running:
            try:
                data, _addr = self._receiver_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            self.stats.delivered += 1
            self.stats.bytes_delivered += len(data)
            if self._on_receive_src is not None:
                self._on_receive_src(data, _addr)
            elif self._on_receive is not None:
                self._on_receive(data)
