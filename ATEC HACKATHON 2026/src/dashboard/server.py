"""
Mission Control dashboard backend for Project C.R.A.T.E.R.

FastAPI + WebSocket, real-time push (not polling): the moment a decoded
hazard comes off the link, it is broadcast to every connected browser.

Five things this dashboard is built to show honestly:
  - data *age* per hazard (how long since it was last refreshed)
  - the link's actual current health (loss / latency / bandwidth)
  - live camera feed with detection overlays (what the rover actually sees)
  - pipeline-wide aggregate stats (frames, detections, FEC recoveries, bytes)
  - full runtime control (pause/resume, rover speed, link parameters)

/api/link_config is a live control: POST new loss/latency/bandwidth values
and the running LinkEmulator picks them up immediately.

/api/pipeline_control is a live control: POST pause/resume, speed changes,
and position resets, and the running pipeline picks them up on its next frame.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .local_advisor import get_advice as get_local_advice
from .local_advisor import get_advice_with_source

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Mission Control - Project C.R.A.T.E.R")


@app.middleware("http")
async def _no_cache(request, call_next):
    """The dashboard HTML/JS changes between dev iterations and gets
    re-served from the same URL every time -- without this, browsers (and
    especially Chrome/Edge on a plain http:// localhost URL) are prone to
    silently keep serving a stale cached copy after a restart, which looks
    exactly like "my changes didn't take effect". Force a fresh fetch on
    every load instead of fighting browser cache heuristics."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


_clients: set[WebSocket] = set()
_broadcast_queue: "asyncio.Queue" = None
_link_emulator = None
_latest_frame_data: dict | None = None

# ---------------------------------------------------------------------------
# Pipeline control — mutable from the dashboard, read by the pipeline thread
# ---------------------------------------------------------------------------
_pipeline_control = {
    "paused": False,
    "speed": 0.15,
    "reset_requested": False,
    "fec_enabled": True,
    "flare": False,
    "detector": "depth",
    "detector_active": "depth",
    "fov_360": True,
}

# ---------------------------------------------------------------------------
# Pipeline stats — written by the pipeline thread, read by the dashboard
# ---------------------------------------------------------------------------
_pipeline_stats = {
    "total_frames": 0,
    "total_detections": 0,
    "total_packets_sent": 0,
    "total_bytes_sent": 0,
    "fec_recoveries": 0,
    "fec_skipped": 0,
    "unique_hazards": 0,
    "start_time_ms": 0,
    "packets_encrypted": 0,
    "packets_auth_ok": 0,
    "packets_auth_failed": 0,
}


def bind_link_emulator(emulator):
    global _link_emulator
    _link_emulator = emulator


def get_broadcast_queue() -> "asyncio.Queue":
    return _broadcast_queue


def get_pipeline_control() -> dict:
    """Called by the pipeline thread to read the current control state."""
    return _pipeline_control


def update_pipeline_stats(stats: dict):
    """Called by the pipeline thread to push aggregate stats."""
    _pipeline_stats.update(stats)


def set_frame_data(frame: np.ndarray, hazards: list, rover_info: dict, disparity: np.ndarray = None):
    """Called by the pipeline thread to provide the latest camera frame
    with detection overlays for the dashboard."""
    global _latest_frame_data
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()

    for hz in hazards:
        color = (255, 157, 77) if hz.kind == "rock" else (79, 168, 255)  # BGR
        cx, cy, r = int(hz.px), int(hz.py), int(hz.radius_px)
        cv2.circle(vis, (cx, cy), r + 4, color, 2)
        cv2.line(vis, (cx - r - 6, cy), (cx - r - 2, cy), color, 1)
        cv2.line(vis, (cx + r + 2, cy), (cx + r + 6, cy), color, 1)
        cv2.line(vis, (cx, cy - r - 6), (cx, cy - r - 2), color, 1)
        cv2.line(vis, (cx, cy + r + 2), (cx, cy + r + 6), color, 1)
        label = f"{hz.kind.upper()} {hz.confidence * 100:.0f}%"
        cv2.putText(vis, label, (cx + r + 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    cv2.putText(vis, f"ROVER  x={rover_info['x']:.1f} y={rover_info['y']:.1f}  hdg={rover_info['heading']:.0f}deg",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (58, 214, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"FRAME {rover_info.get('frame', 0)}  {len(hazards)} detections",
                (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (58, 214, 255), 1, cv2.LINE_AA)

    # If disparity is provided, colorize it and place it side-by-side
    if disparity is not None:
        # Normalize disparity for visualization
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)
        cv2.putText(disp_color, "STEREO DISPARITY MAP", (8, 16), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        vis = cv2.hconcat([vis, disp_color])

    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    _latest_frame_data = {"type": "camera_frame", "image": b64,
                          "width": vis.shape[1], "height": vis.shape[0],
                          "detections": len(hazards),
                          "rover_x": rover_info.get("x", 0),
                          "rover_y": rover_info.get("y", 0),
                          "rover_heading": rover_info.get("heading", 0),
                          "frame_num": rover_info.get("frame", 0)}


@app.on_event("startup")
async def _startup():
    global _broadcast_queue
    _broadcast_queue = asyncio.Queue()
    asyncio.create_task(_broadcast_loop())
    asyncio.create_task(_link_stats_loop())
    asyncio.create_task(_camera_frame_loop())


async def _broadcast_loop():
    while True:
        msg = await _broadcast_queue.get()
        await _broadcast(msg)


async def _link_stats_loop():
    """Broadcasts link health, pipeline stats, and control state every 0.5s."""
    while True:
        if _link_emulator is not None:
            stats = _link_emulator.stats.as_dict()
            cfg = _link_emulator.get_config()
            await _broadcast({
                "type": "link_stats",
                "stats": stats,
                "config": cfg,
                "server_time_ms": int(time.time() * 1000),
                "pipeline_stats": dict(_pipeline_stats),
                "pipeline_control": dict(_pipeline_control),
            })
        await asyncio.sleep(0.5)


async def _camera_frame_loop():
    """Broadcasts the latest camera frame at ~2.5 FPS to all connected clients."""
    while True:
        if _latest_frame_data is not None:
            await _broadcast(_latest_frame_data)
        await asyncio.sleep(0.4)


async def _broadcast(msg: dict):
    if not _clients:
        return
    data = json.dumps(msg)
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


# ---- Pydantic models for API inputs ----

class LinkConfigUpdate(BaseModel):
    loss_rate: Optional[float] = None
    latency_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    bandwidth_bps: Optional[float] = None


class PipelineControlUpdate(BaseModel):
    paused: Optional[bool] = None
    speed: Optional[float] = None
    reset_requested: Optional[bool] = None
    fec_enabled: Optional[bool] = None
    flare: Optional[bool] = None
    detector: Optional[str] = None
    fov_360: Optional[bool] = None


# ---- REST endpoints ----

@app.get("/api/link_config")
async def get_link_config():
    if _link_emulator is None:
        return {}
    return _link_emulator.get_config()


@app.post("/api/link_config")
async def set_link_config(update: LinkConfigUpdate):
    if _link_emulator is not None:
        _link_emulator.set_config(**update.dict(exclude_none=True))
    return {"ok": True, "config": _link_emulator.get_config() if _link_emulator else {}}


@app.get("/api/pipeline_control")
async def get_pipeline_control_endpoint():
    return _pipeline_control


@app.post("/api/pipeline_control")
async def set_pipeline_control_endpoint(update: PipelineControlUpdate):
    if update.paused is not None:
        _pipeline_control["paused"] = update.paused
    if update.speed is not None:
        _pipeline_control["speed"] = max(0.02, min(0.50, update.speed))
    if update.reset_requested is not None:
        _pipeline_control["reset_requested"] = update.reset_requested
    if update.fec_enabled is not None:
        _pipeline_control["fec_enabled"] = update.fec_enabled
    if update.flare is not None:
        _pipeline_control["flare"] = update.flare
    if update.detector in ("depth", "yolo", "both"):
        _pipeline_control["detector"] = update.detector
    if update.fov_360 is not None:
        _pipeline_control["fov_360"] = update.fov_360
    return {"ok": True, "control": _pipeline_control}


@app.get("/api/pipeline_stats")
async def get_pipeline_stats_endpoint():
    return _pipeline_stats


@app.get("/api/trace")
async def get_trace(limit: int = 50):
    trace_path = Path(__file__).resolve().parents[2] / "trace_log.jsonl"
    if not trace_path.exists():
        return []
    lines = trace_path.read_text().strip().splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@app.get("/api/ai_advice")
async def get_ai_advice():
    # Fully local, offline advisor -- no network call, no API key, nothing
    # leaves this process. See local_advisor.py for the retrieval + rule
    # engine. We hand it a flat dict rather than a free-text summary so the
    # rule engine can reason over real numbers, not re-parse prose.
    combined = {}
    if _link_emulator is not None:
        l_stats = _link_emulator.stats.as_dict()
        l_cfg = _link_emulator.get_config()
        combined["loss_rate"] = l_cfg.get("loss_rate", 0.0)
        combined["latency_ms"] = l_cfg.get("latency_ms", 0.0)
        combined["survival_rate"] = l_stats.get("survival_rate", 1.0)
    combined["fec_recoveries"] = _pipeline_stats.get("fec_recoveries", 0)
    combined["fec_enabled"] = _pipeline_control.get("fec_enabled", True)
    combined["unique_hazards"] = _pipeline_stats.get("unique_hazards", 0)
    combined["total_packets_sent"] = _pipeline_stats.get("total_packets_sent", 0)
    combined["flare_active"] = _pipeline_control.get("flare", False)
    advice, source = await asyncio.to_thread(get_advice_with_source, combined)
    return {"advice": advice, "source": source}

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/security")
async def security_dashboard():
    return FileResponse(str(STATIC_DIR / "security.html"))
