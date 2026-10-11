#!/usr/bin/env python3
"""
Project C.R.A.T.E.R — single-command end-to-end pipeline.

    synthetic camera frame
        -> perception.detect_hazards()           [Perceive]
        -> localization.pixel_to_landing_frame()  [Localize]
        -> telemetry.TelemetryEncoder.encode()     [Transmit: encode]
        -> link_sim.LinkEmulator (lossy UDP)        [Transmit: lossy link]
        -> telemetry decode + FEC reconstruct       [Transmit: decode]
        -> dashboard broadcast over WebSocket        [Visualize]

Every hazard detection gets a stable id (quantized grid cell) and every hop
is timestamped into trace_log.jsonl, so one detection's whole journey can
be pulled up live: "hazard #7, detected at T+..., encoded in N bytes,
survived a lossy link, rendered on the dashboard with an uncertainty of
... meters."

Run:  python run_pipeline.py
Then open http://127.0.0.1:8000 in a browser.
Stop with Ctrl+C.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn

# On Windows, asyncio's default "Proactor" event loop logs a noisy (but
# harmless) ConnectionResetError whenever a browser tab closes or reloads
# its WebSocket mid-connection -- it's tearing down an already-dead socket,
# not an actual failure. The "Selector" loop doesn't have that teardown
# quirk, so we switch to it before uvicorn ever creates a loop. No effect
# on Linux/Mac, where this isn't needed.
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src import telemetry as T
from src.link_sim import LinkEmulator, LinkConfig
from src.localization import CameraModel, RoverPose, pixel_to_landing_frame, LocalizationError
from src.perception import generate_world_hazard_field, render_world_view, detect_hazards
from src import yolo_detector
from src.security import MissionCrypto, SecurityError, derive_mission_key
from src.intrusion import IntrusionGuard, LAYER_AEAD, LAYER_CRC
from src.dashboard import server as dashboard_server

TRACE_PATH = Path(__file__).parent / "trace_log.jsonl"
IMAGE_W, IMAGE_H = 480, 360
CAMERA = CameraModel(fx=420, fy=420, cx=IMAGE_W / 2, cy=IMAGE_H / 2, height_m=1.2, pitch_deg=22)
HAZARD_GRID_M = 0.6  # detections within this distance of each other are treated as the same hazard
WORLD_AREA_M = 26.0  # extent of the fixed hazard field the rover roams
N_SECTORS = 6  # 360-degree sweep: 6 sectors x 60 deg each == full coverage, matched to CAMERA's own ~60 deg FOV
SECTOR_STEP_DEG = 360.0 / N_SECTORS


class TraceLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.path.write_text("")  # fresh log each run

    def log(self, hazard_id: int, stage: str, detail: str = ""):
        entry = {"ts_ms": int(time.time() * 1000), "hazard_id": hazard_id, "stage": stage, "detail": detail}
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")


from scipy.optimize import linear_sum_assignment

class HazardTracker:
    """True mathematical Data Association tracker. Uses the Hungarian algorithm 
    (Bipartite matching) to pair new frame detections with existing tracked 
    hazards based on minimum Euclidean distance. Far superior to grid snapping."""

    def __init__(self, max_dist_m: float = HAZARD_GRID_M * 1.5):
        self.max_dist = max_dist_m
        self._tracked: dict[int, tuple[float, float]] = {}  # id -> (x, y)
        self._next_id = 1

    def assign_ids(self, detections: list[tuple[float, float]]) -> list[int]:
        if not detections:
            return []
            
        if not self._tracked:
            out = []
            for d in detections:
                self._tracked[self._next_id] = d
                out.append(self._next_id)
                self._next_id += 1
            return out

        # Build cost matrix
        tracked_ids = list(self._tracked.keys())
        cost_matrix = np.zeros((len(detections), len(tracked_ids)))
        for i, (dx, dy) in enumerate(detections):
            for j, tid in enumerate(tracked_ids):
                tx, ty = self._tracked[tid]
                dist = math.hypot(dx - tx, dy - ty)
                cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_ids = [-1] * len(detections)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_dist:
                tid = tracked_ids[c]
                assigned_ids[r] = tid
                self._tracked[tid] = detections[r]  # Update position
                
        # Handle unmatched detections
        for i in range(len(detections)):
            if assigned_ids[i] == -1:
                assigned_ids[i] = self._next_id
                self._tracked[self._next_id] = detections[i]
                self._next_id += 1
                
        return assigned_ids


def _on_guard_event(trace, kind, source, layer, detail):
    if kind in ("blocked", "recovered"):
        trace.log(0, kind, f"{source[0]}:{source[1]} via {layer}: {detail}")


class ReceiverState:
    """Receiver-side authoritative hazard state: merges FULL/DELTA packets
    and performs FEC reconstruction before handing clean hazard updates to
    the dashboard broadcast queue."""

    def __init__(self, trace: TraceLogger, kind_lookup: dict, crypto: MissionCrypto, guard: IntrusionGuard):
        self.trace = trace
        self.guard = guard
        self.kind_lookup = kind_lookup  # hazard_id -> "crater"/"rock" (sent out-of-band for this demo)
        self.crypto = crypto
        self.positions: dict[int, dict] = {}
        self.fec = T.FECReconstructor(k=T.FEC_K)
        self._block_seq_counter = 0
        self._fec_slot_of = {}  # seq -> (block_id, slot) bookkeeping for FULL packets only
        self._next_slot = 0
        self._current_block = 0
        self.fec_recovery_count = 0  # aggregate stat for dashboard

    def handle_packet(self, wire_bytes: bytes, src: tuple | None = None):
        # Security layer first: every wire packet is AEAD-encrypted
        # (ChaCha20-Poly1305). A tampered or corrupted ciphertext fails
        # authentication here and is dropped before it ever reaches the
        # telemetry decoder -- a stronger guarantee than CRC8 alone, which
        # only detects accidental bit errors, not deliberate tampering.
        if src is not None and not self.guard.admit(src):
            return
        try:
            raw = self.crypto.decrypt(wire_bytes)
        except SecurityError:
            where = "unknown" if src is None else f"{src[0]}:{src[1]}"
            self.trace.log(0, "auth_failed", f"rejected at {LAYER_AEAD} from {where} -- tampered or corrupted ciphertext")
            if src is not None:
                self.guard.record_failure(src, LAYER_AEAD, "ChaCha20-Poly1305 tag mismatch")
            return
        self.trace.log(0, "authenticated", f"{len(wire_bytes)}B AEAD verified")

        try:
            msg = T.decode_any(raw)
        except T.TelemetryError:
            if src is not None:
                self.guard.record_failure(src, LAYER_CRC, "telemetry CRC8 mismatch after decryption")
            return

        if msg["type"] == "FULL":
            slot = self._next_slot % T.FEC_K
            if slot == 0 and self._next_slot > 0:
                self._current_block += 1
            self._next_slot += 1
            self.fec.observe_full(self._current_block, slot, raw)
            self._apply_full(msg)
        elif msg["type"] == "DELTA":
            self._apply_delta(msg)
        elif msg["type"] == "PARITY":
            rebuilt = self.fec.observe_parity(msg["block_id"], msg["seq_mask"], msg["xor_payload"])
            if rebuilt is not None:
                try:
                    full_msg = T.decode_full(rebuilt)
                    self.fec_recovery_count += 1
                    self.trace.log(full_msg["hazard_id"], "fec_reconstructed",
                                    "packet recovered from parity after loss")
                    self._apply_full(full_msg)
                except T.TelemetryError:
                    pass

    def _apply_full(self, msg: dict):
        hid = msg["hazard_id"]
        self.positions[hid] = {"x": msg["x"], "y": msg["y"], "confidence": msg["confidence"],
                                 "timestamp_ms": msg["timestamp_ms"]}
        self._emit(hid)
        self.trace.log(hid, "decoded_full", f"x={msg['x']:.2f} y={msg['y']:.2f}")

    def _apply_delta(self, msg: dict):
        hid = msg["hazard_id"]
        prev = self.positions.get(hid)
        if prev is None:
            return  # delta with no prior FULL -- can happen right at startup, safely ignored
        prev["x"] += msg["dx"]
        prev["y"] += msg["dy"]
        prev["confidence"] = msg["confidence"]
        prev["timestamp_ms"] += msg["dt_ms"]
        self._emit(hid)
        self.trace.log(hid, "decoded_delta", f"dx={msg['dx']:.2f} dy={msg['dy']:.2f}")

    def _emit(self, hid: int):
        p = self.positions[hid]
        queue = dashboard_server.get_broadcast_queue()
        if queue is None:
            return
        payload = {
            "type": "hazard", "hazard_id": hid, "x": p["x"], "y": p["y"],
            "confidence": p["confidence"], "kind": self.kind_lookup.get(hid, "crater"),
            "sx": p.get("sx", 0.1),
            "sy": p.get("sy", 0.1),
            "angle": p.get("angle", 0.0)
        }
        # queue is bound to the uvicorn event loop; this call runs from a
        # non-asyncio thread (the link receiver), so hop over safely.
        loop = dashboard_server_loop_holder.get("loop")
        if loop is not None:
            loop.call_soon_threadsafe(queue.put_nowait, payload)


dashboard_server_loop_holder: dict = {}


def _quiet_socket_teardown_errors(loop, context):
    """Windows' Proactor event loop logs a noisy-but-harmless
    ConnectionResetError/ConnectionAbortedError whenever a browser tab
    closes or reloads its WebSocket mid-connection -- it's tearing down an
    already-dead socket, not an actual pipeline failure. Swallow exactly
    that case and forward everything else to the default handler so real
    errors still surface."""
    exc = context.get("exception")
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
        return
    loop.default_exception_handler(context)


def run_dashboard(emulator: LinkEmulator, host="127.0.0.1", port=8000):
    dashboard_server.bind_link_emulator(emulator)
    config = uvicorn.Config(dashboard_server.app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    async def _capture_loop():
        import asyncio
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_quiet_socket_teardown_errors)
        dashboard_server_loop_holder["loop"] = loop
        await server.serve()

    import asyncio
    asyncio.run(_capture_loop())


def main():
    trace = TraceLogger(TRACE_PATH)
    tracker = HazardTracker()
    kind_lookup: dict[int, str] = {}

    # Sender and receiver share one mission key (see src/security.py for
    # why this is derived locally for the demo rather than provisioned).
    # Two separate MissionCrypto instances deliberately -- each side only
    # ever needs encrypt() or decrypt(), never both, which is how a real
    # sender/receiver pair would be split across two separate machines.
    crypto_key = derive_mission_key()
    sender_crypto = MissionCrypto(crypto_key)
    receiver_crypto = MissionCrypto(crypto_key)

    emulator = LinkEmulator(LinkConfig(loss_rate=0.15, latency_ms=150, jitter_ms=60, bandwidth_bps=2000))
    guard = IntrusionGuard(
        trusted_source=emulator.forward_addr,
        on_event=lambda kind, source, layer, detail: _on_guard_event(trace, kind, source, layer, detail),
        policy_path=str(Path(__file__).parent / "rl_policy.json"),
    )
    receiver = ReceiverState(trace, kind_lookup, receiver_crypto, guard)
    emulator.on_receive_src(receiver.handle_packet)
    emulator.start()

    dash_thread = threading.Thread(target=run_dashboard, args=(emulator,), daemon=True)
    dash_thread.start()
    time.sleep(1.0)  # give uvicorn a moment to bind and capture its event loop
    print("Mission Control dashboard: http://127.0.0.1:8000")

    # A FIXED set of hazards in world coordinates, generated once for the
    # whole run -- not regenerated per frame. This is what makes the same
    # physical rock/crater reappear at the same world position the next
    # time the rover's camera sweeps past it, instead of the perception
    # pipeline effectively hallucinating a brand new hazard every frame.
    world_hazards = generate_world_hazard_field(n_hazards=14, area_m=WORLD_AREA_M, seed=0)
    trace.log(0, "world_init", f"{len(world_hazards)} fixed hazards seeded for this run")

    encoder = T.TelemetryEncoder()
    rover = RoverPose(x_m=0.0, y_m=0.0, heading_deg=0.0)
    frame_i = 0
    total_packets_sent = 0
    total_bytes_sent = 0
    total_detections = 0
    fec_skipped = 0
    start_time_ms = int(time.time() * 1000)

    dashboard_server.update_pipeline_stats({"start_time_ms": start_time_ms})

    print("Pipeline running. Ctrl+C to stop.")
    try:
        while True:
            # ---- Dashboard runtime controls ----
            ctrl = dashboard_server.get_pipeline_control()

            if ctrl.get("paused"):
                time.sleep(0.2)
                continue

            if ctrl.get("reset_requested"):
                rover.x_m, rover.y_m, rover.heading_deg = 0.0, 0.0, 0.0
                frame_i = 0
                total_packets_sent = 0
                total_bytes_sent = 0
                total_detections = 0
                fec_skipped = 0
                receiver.fec_recovery_count = 0
                start_time_ms = int(time.time() * 1000)
                ctrl["reset_requested"] = False
                trace.log(0, "reset", "pipeline reset from dashboard")
                continue

            path_speed = ctrl.get("speed", 0.15)
            fec_enabled = ctrl.get("fec_enabled", True)
            flare_active = ctrl.get("flare", False)
            fov_360 = ctrl.get("fov_360", True)

            # 360-degree sensing: the camera's own horizontal FOV is fixed
            # (~60 deg, set by CAMERA.fx and IMAGE_W), but the rover is not
            # limited to looking only in its direction of travel. We sweep
            # N_SECTORS virtual camera headings every frame -- same camera
            # model, same detector, just re-posed -- and localize each
            # sector's detections against ITS OWN heading before merging
            # everything into one point list for the tracker. This is a
            # real 360-degree hazard scan, not a cosmetic wide-angle lens:
            # a hazard directly behind the rover is found exactly as
            # reliably as one ahead of it.
            n_sectors = N_SECTORS if fov_360 else 1
            now_ms = int(time.time() * 1000)

            all_valid_locs = []
            all_valid_hz = []
            primary_frame = primary_disparity = primary_hazards = None

            for s in range(n_sectors):
                sector_heading = (rover.heading_deg + s * SECTOR_STEP_DEG) % 360.0
                sector_pose = RoverPose(rover.x_m, rover.y_m, sector_heading)
                frame_s, disparity_s, _visible = render_world_view(
                    world_hazards, CAMERA, sector_pose, IMAGE_W, IMAGE_H, flare=flare_active)

                mode = ctrl.get("detector", "depth")
                if mode != "depth" and not yolo_detector.available():
                    mode = "depth"  # no weights / no torch -> graceful fallback
                if s == 0:
                    ctrl["detector_active"] = mode

                if mode == "yolo":
                    hazards_s = yolo_detector.detect(frame_s)
                elif mode == "both":
                    hazards_s = yolo_detector.fuse(yolo_detector.detect(frame_s), detect_hazards(frame_s, disparity_s))
                else:
                    hazards_s = detect_hazards(frame_s, disparity_s)

                total_detections += len(hazards_s)
                if s == 0:  # forward sector is what the live camera feed shows
                    primary_frame, primary_disparity, primary_hazards = frame_s, disparity_s, hazards_s

                for hz in hazards_s:
                    try:
                        loc = pixel_to_landing_frame(hz.px, hz.py, IMAGE_W, IMAGE_H, CAMERA, sector_pose)
                        all_valid_locs.append(loc)
                        all_valid_hz.append(hz)
                    except LocalizationError:
                        continue

            # Send the forward-sector camera frame + detections to the
            # dashboard's live feed (showing all 6 sectors at once would be
            # unreadable at this resolution; the hazard map below already
            # shows the full 360-degree picture once localized).
            dashboard_server.set_frame_data(primary_frame, primary_hazards, {
                "x": rover.x_m, "y": rover.y_m,
                "heading": rover.heading_deg, "frame": frame_i,
                "fov_360": fov_360, "sectors_scanned": n_sectors,
            }, disparity=primary_disparity)

            valid_locs, valid_hz = all_valid_locs, all_valid_hz

            # Pass all detected points (from every sector) to the Hungarian tracker
            point_list = [(loc["x"], loc["y"]) for loc in valid_locs]
            assigned_ids = tracker.assign_ids(point_list)

            for i in range(len(valid_hz)):
                hz = valid_hz[i]
                loc = valid_locs[i]
                hid = assigned_ids[i]
                
                kind_lookup[hid] = hz.kind
                trace.log(hid, "detected", f"px=({hz.px:.0f},{hz.py:.0f}) conf={hz.confidence:.2f}")
                trace.log(hid, "localized", f"x={loc['x']:.2f} y={loc['y']:.2f} sx={loc['sx']:.2f} sy={loc['sy']:.2f}")

                for priority, pkt, redundant in encoder.encode(
                        hid, loc["x"], loc["y"], loc["sx"], loc["sy"], loc["angle"], hz.confidence, loc["distance_m"], now_ms):
                    if T.peek_type(pkt) == T.MSG_PARITY and not fec_enabled:
                        fec_skipped += 1
                        continue
                    # Encrypt at the wire boundary: FEC/CRC above this line
                    # operate on plaintext packet bytes (see security.py for
                    # why), each send gets its own fresh nonce, and the
                    # ~28-byte AEAD overhead (12B nonce + 16B tag) is counted
                    # honestly in total_bytes_sent rather than hidden.
                    wire_pkt = sender_crypto.encrypt(pkt)

                    emulator.send(wire_pkt)
                    total_packets_sent += 1
                    total_bytes_sent += len(wire_pkt)
                    if redundant:
                        wire_pkt2 = sender_crypto.encrypt(pkt)  # fresh nonce per send
                        emulator.send(wire_pkt2)  # high-priority redundancy: duplicate send
                        total_packets_sent += 1
                        total_bytes_sent += len(wire_pkt2)
                    trace.log(hid, "transmitted", f"{len(pkt)}B priority={priority:.2f}")

                # stash uncertainty on the receiver-visible dict once it exists (best-effort)
                st = receiver.positions.get(hid)
                if st is not None:
                    st["sx"] = loc["sx"]
                    st["sy"] = loc["sy"]
                    st["angle"] = loc["angle"]

            # ---- Push aggregate stats to dashboard every frame ----
            dashboard_server.update_pipeline_stats({
                "total_frames": frame_i + 1,
                "total_detections": total_detections,
                "total_packets_sent": total_packets_sent,
                "total_bytes_sent": total_bytes_sent,
                "fec_recoveries": receiver.fec_recovery_count,
                "fec_skipped": fec_skipped,
                "unique_hazards": len(tracker._tracked),
                "start_time_ms": start_time_ms,
                "packets_encrypted": sender_crypto.encrypt_count,
                "packets_auth_ok": receiver_crypto.decrypt_ok_count,
                "packets_auth_failed": receiver_crypto.decrypt_fail_count,
                "intrusion": guard.snapshot(),
            })

            # Compass convention (matches localization.py): heading 0 = +y
            # ("north"), increasing clockwise toward +x ("east") -> dx=sin, dy=cos.
            heading_rad = math.radians(rover.heading_deg)
            rover.x_m += path_speed * math.sin(heading_rad)
            rover.y_m += path_speed * math.cos(heading_rad)

            field_center = (WORLD_AREA_M / 2, WORLD_AREA_M / 2)
            out_of_bounds = not (-3 <= rover.x_m <= WORLD_AREA_M + 3 and -3 <= rover.y_m <= WORLD_AREA_M + 3)
            if out_of_bounds:
                # steer back toward the hazard field instead of wandering off
                # it forever -- keeps a long-running demo continually
                # re-visiting known hazards (the point of this whole fix).
                dx, dy = field_center[0] - rover.x_m, field_center[1] - rover.y_m
                rover.heading_deg = math.degrees(math.atan2(dx, dy)) % 360
            elif frame_i % 50 == 0 and frame_i > 0:
                rover.heading_deg = (rover.heading_deg + 35) % 360

            frame_i += 1
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        emulator.stop()


if __name__ == "__main__":
    main()
