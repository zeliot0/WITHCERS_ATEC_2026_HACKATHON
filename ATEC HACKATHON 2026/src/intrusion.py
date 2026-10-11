from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from typing import Callable, Optional

LAYER_ORIGIN = "L1-origin"
LAYER_AEAD = "L2-AEAD"
LAYER_CRC = "L3-CRC8"


class RLBlockPolicy:
    """Tabular Q-learning policy for choosing how long to block a source.

    State: strike count for that source, clipped to 1..4+.
    Action: a block duration in seconds, picked from a fixed menu.
    Reward: settled once the outcome of a decision is known --
      -1.0 if the source re-offends within a short grace window right
           after the block expires (the block was too short),
      +1.0 if the source stays quiet through the grace window (the
           block worked), scaled down for longer durations so the
           policy also learns not to over-block.

    Q-table persists to disk (policy_path) so the agent keeps what it
    learned across restarts instead of relearning every run.
    """

    ACTIONS = (8.0, 16.0, 32.0, 60.0)

    def __init__(self, path: Optional[str] = None, alpha: float = 0.3,
                 gamma: float = 0.5, epsilon: float = 0.15):
        self.path = path
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: dict[int, list] = {}
        self._pending: dict[tuple, list] = {}
        self.decisions = 0
        self.rewards_given = 0
        self._load()

    def _state(self, strikes: int) -> int:
        return min(strikes, 4)

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.q = {int(k): v for k, v in data.get("q", {}).items()}
            self.decisions = data.get("decisions", 0)
            self.rewards_given = data.get("rewards_given", 0)
        except (OSError, ValueError, json.JSONDecodeError):
            self.q = {}

    def _save(self):
        if not self.path:
            return
        try:
            with open(self.path, "w") as f:
                json.dump({"q": self.q, "decisions": self.decisions,
                           "rewards_given": self.rewards_given}, f)
        except OSError:
            pass

    def choose(self, source: tuple, strikes: int) -> float:
        s = self._state(strikes)
        row = self.q.setdefault(s, [0.0] * len(self.ACTIONS))
        if random.random() < self.epsilon or self.decisions < 2 * len(self.ACTIONS):
            a = random.randrange(len(self.ACTIONS))
        else:
            a = max(range(len(row)), key=lambda i: row[i])
        self.decisions += 1
        self._pending[source] = [s, a]
        self._save()
        return self.ACTIONS[a]

    def reward_reoffend(self, source: tuple) -> None:
        self._settle(source, -1.0)

    def reward_clean(self, source: tuple) -> None:
        pending = self._pending.get(source)
        if pending is None:
            return
        duration = self.ACTIONS[pending[1]]
        efficiency_bonus = 1.0 - 0.4 * (duration / max(self.ACTIONS))
        self._settle(source, 0.6 + efficiency_bonus)

    def _settle(self, source: tuple, reward: float) -> None:
        pending = self._pending.pop(source, None)
        if pending is None:
            return
        s, a = pending
        row = self.q.setdefault(s, [0.0] * len(self.ACTIONS))
        best_next = max(row)
        row[a] += self.alpha * (reward + self.gamma * best_next - row[a])
        self.rewards_given += 1
        self._save()

    def best_action(self, strikes: int) -> float:
        s = self._state(strikes)
        row = self.q.get(s)
        if not row:
            return self.ACTIONS[0]
        return self.ACTIONS[max(range(len(row)), key=lambda i: row[i])]

    def snapshot(self) -> dict:
        return {
            "decisions": self.decisions,
            "rewards_given": self.rewards_given,
            "epsilon": self.epsilon,
            "actions_s": list(self.ACTIONS),
            "q_table": {str(k): [round(x, 3) for x in v] for k, v in self.q.items()},
            "learned_next_block_s": self.best_action(1),
        }


class IntrusionGuard:
    def __init__(
        self,
        trusted_source: tuple,
        on_event: Optional[Callable[[str, tuple, str, str], None]] = None,
        fail_threshold: int = 3,
        window_s: float = 10.0,
        base_block_s: float = 8.0,
        max_block_s: float = 60.0,
        max_events: int = 50,
        policy_path: Optional[str] = None,
        reoffend_grace_s: float = 6.0,
    ):
        self.trusted_source = tuple(trusted_source)
        self.on_event = on_event
        self.fail_threshold = fail_threshold
        self.window_s = window_s
        self.base_block_s = base_block_s
        self.max_block_s = max_block_s
        self.reoffend_grace_s = reoffend_grace_s
        self._failures: dict[tuple, deque] = {}
        self._blocked: dict[tuple, dict] = {}
        self._strikes: dict[tuple, int] = {}
        self._recent_unblocks: dict[tuple, float] = {}
        self._events: deque = deque(maxlen=max_events)
        self.counts = {"blocked_drops": 0, "rejected": 0, "off_path": 0, "incidents": 0, "recoveries": 0}
        self.last_incident: Optional[dict] = None
        self.policy = RLBlockPolicy(path=policy_path)

    def origin(self, source: tuple) -> str:
        return "on-link" if tuple(source) == self.trusted_source else "off-path"

    def admit(self, source: tuple, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        self._settle_due_rewards(now)
        block = self._blocked.get(source)
        if block is None:
            return True
        if now < block["until"]:
            self.counts["blocked_drops"] += 1
            return False
        del self._blocked[source]
        self._failures.pop(source, None)
        self._recent_unblocks[source] = now
        self.counts["recoveries"] += 1
        self._emit("recovered", source, LAYER_ORIGIN, "block expired, source readmitted -- self-healed")
        return True

    def record_failure(self, source: tuple, layer: str, reason: str, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        source = tuple(source)
        origin = self.origin(source)
        self.counts["rejected"] += 1
        if origin == "off-path":
            self.counts["off_path"] += 1
        self._emit("rejected", source, layer, f"{origin}: {reason}")

        if origin != "off-path":
            return

        if source in self._recent_unblocks:
            if now - self._recent_unblocks[source] <= self.reoffend_grace_s:
                self.policy.reward_reoffend(source)
            del self._recent_unblocks[source]

        failures = self._failures.setdefault(source, deque())
        failures.append(now)
        while failures and now - failures[0] > self.window_s:
            failures.popleft()
        if len(failures) < self.fail_threshold or source in self._blocked:
            return

        strikes = self._strikes.get(source, 0) + 1
        self._strikes[source] = strikes
        duration = min(self.max_block_s, self.policy.choose(source, strikes))
        self._blocked[source] = {"until": now + duration, "strikes": strikes}
        failures.clear()
        self.counts["incidents"] += 1
        self.last_incident = {"source": _fmt(source), "layer": layer, "origin": origin, "strike": strikes}
        self._emit("blocked", source, layer,
                   f"source blocked for {duration:.0f}s (strike {strikes}, RL-chosen) after {self.fail_threshold} rejected packets")

    def _settle_due_rewards(self, now: float) -> None:
        due = [src for src, ts in self._recent_unblocks.items() if now - ts > self.reoffend_grace_s]
        for src in due:
            self.policy.reward_clean(src)
            del self._recent_unblocks[src]

    def state(self, now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        if any(now < b["until"] for b in self._blocked.values()):
            return "UNDER ATTACK"
        if self.counts["recoveries"] > 0:
            return "RECOVERED"
        return "NOMINAL"

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = time.monotonic() if now is None else now
        self._settle_due_rewards(now)
        blocked = [
            {"source": _fmt(src), "resumes_in_s": round(max(0.0, b["until"] - now), 1), "strike": b["strikes"]}
            for src, b in self._blocked.items() if now < b["until"]
        ]
        return {
            "state": self.state(now),
            "trusted_link": _fmt(self.trusted_source),
            "blocked": blocked,
            "counts": dict(self.counts),
            "last_incident": self.last_incident,
            "policy": self.policy.snapshot(),
        }

    def _emit(self, kind: str, source: tuple, layer: str, detail: str) -> None:
        event = {"kind": kind, "source": _fmt(source), "layer": layer, "detail": detail, "ts": time.time()}
        self._events.append(event)
        if self.on_event is not None:
            self.on_event(kind, source, layer, detail)

    def events(self) -> list:
        return list(self._events)


def _fmt(source: tuple) -> str:
    return f"{source[0]}:{source[1]}" if source else "unknown"
