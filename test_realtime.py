"""
Test Realtime VGGT Pose Server (scaffold)
----------------------------------------

Goal
====
Accept frames from HoloLens/Unity via HTTP POST (multipart-form), keep a
small sliding window in memory, run VGGT on that window, and print the
latest frame's camera pose + timing to the terminal. No GLB export.

How to run
==========
# 1) Install deps (examples)
#    pip install fastapi uvicorn[standard] pillow numpy opencv-python torch
#    (VGGT repo deps must already be installed; run in your vggt env)
#
# 2) Start server
#    python test_realtime.py --host 0.0.0.0 --port 8000 --window 8 --cadence-ms 250
#
# 3) Unity/HoloLens client POSTs to:
#    POST http://<server>:8000/frame
#    Content-Type: multipart/form-data
#    Fields:
#      - image: binary (PNG/JPEG)
#      - metadata: JSON string with keys below
#
#    metadata schema:
#    {
#      "frame_id": 123,
#      "timestamp_ms": 1690000000000,
#      "intrinsics": {"fx":..., "fy":..., "cx":..., "cy":..., "width":..., "height":...,
#                      "distortion": [k1,k2,p1,p2,k3] or null },
#      "session_id": "optional-session-key"  // start new to reset origin/buffer
#    }
#
# Integration notes
# =================
# Replace the placeholder `vggt_infer_poses(...)` to call your actual VGGT
# code path that returns camera-to-world (R, t) for every image in the window.
# Keep the server in right-handed coordinates. If/when you stream to Unity,
# do LH/RH conversion on the Unity side.
"""

from __future__ import annotations
import argparse
import io
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# --- VGGT imports (adjust paths to your repo layout as needed) ---
import os
import tempfile
import shutil
import torch
import sys
sys.path.append("vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# --------------------------- Data Models ---------------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[List[float]] = None  # [k1,k2,p1,p2,k3] or None


@dataclass
class Frame:
    frame_id: int
    timestamp_ms: int
    img_rgb: np.ndarray  # HxWx3 uint8 (undistorted preferred)
    K: np.ndarray        # 3x3
    intrinsics: Intrinsics


# ------------------------- Utility Functions -----------------------

def parse_metadata(meta_json: str) -> Tuple[int, int, Intrinsics, Optional[str]]:
    meta = json.loads(meta_json)
    frame_id = int(meta.get("frame_id", 0))
    timestamp_ms = int(meta.get("timestamp_ms", int(time.time() * 1000)))
    intr = meta.get("intrinsics", {})
    intrinsics = Intrinsics(
        fx=float(intr["fx"]), fy=float(intr["fy"]), cx=float(intr["cx"]), cy=float(intr["cy"]),
        width=int(intr["width"]), height=int(intr["height"]),
        distortion=intr.get("distortion") if intr.get("distortion") is not None else None,
    )
    session_id = meta.get("session_id")
    return frame_id, timestamp_ms, intrinsics, session_id


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def make_K(intr: Intrinsics) -> np.ndarray:
    K = np.array([[intr.fx, 0.0, intr.cx],
                  [0.0, intr.fy, intr.cy],
                  [0.0,  0.0,   1.0 ]], dtype=np.float32)
    return K


# ----------------------- VGGT Pose Inference -----------------------

def vggt_infer_poses(
    frames: List[Frame],
    device: torch.device,
    model: VGGT,
    autocast_dtype: torch.dtype,
    sid: str,
    debug_save: bool,
    debug_root: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Run VGGT over the provided window and return (R,t) for each frame.
    Uses the repo's existing preprocessing/utilities. To keep things simple
    and consistent with your Gradio pipeline, we temporarily write the window
    images to a temp dir (or to sessions/<sid>/images when debug_save=True),
    then call load_and_preprocess_images(image_paths).
    """
    # 1) Materialize images to disk (temp or debug)
    if debug_save:
        target_dir = os.path.join(debug_root, sid)
        images_dir = os.path.join(target_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        # clean and rewrite the current window for clarity
        for f in os.listdir(images_dir):
            try:
                os.remove(os.path.join(images_dir, f))
            except Exception:
                pass
    else:
        target_dir = tempfile.mkdtemp(prefix=f"vggt_rt_{sid}_")
        images_dir = os.path.join(target_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    image_paths: List[str] = []
    for i, fr in enumerate(frames):
        # Keep the original frame_id in filename for traceability
        fn = os.path.join(images_dir, f"{fr.frame_id:06d}.png")
        Image.fromarray(fr.img_rgb).save(fn)
        image_paths.append(fn)

    try:
        # 2) Preprocess & model forward
        t0 = time.time()
        imgs = load_and_preprocess_images(image_paths).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda"), dtype=autocast_dtype):
                predictions = model(imgs)
        # 3) Decode poses
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], imgs.shape[-2:])
        # Move to CPU numpy
        extrinsic = extrinsic.detach().cpu().numpy().squeeze(0)
        intrinsic = intrinsic.detach().cpu().numpy().squeeze(0)

        # 4) Return camera-to-world (R,t) per frame
        outs: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(extrinsic.shape[0]):
            Ei = extrinsic[i]
            if Ei.shape == (3,4):
                H = np.eye(4, dtype=np.float32)
                H[:3,:4] = Ei
            elif Ei.shape == (4,4):
                H = Ei.astype(np.float32)
            else:
                raise ValueError(f"Bad extrinsic shape: {Ei.shape}")
            Twc = np.linalg.inv(H)  # world-from-camera
            Rwc = Twc[:3,:3].astype(np.float32)
            twc = Twc[:3,3].astype(np.float32)
            outs.append((Rwc, twc))
        return outs
    finally:
        # 5) Cleanup temp dir if not debugging
        if not debug_save:
            shutil.rmtree(target_dir, ignore_errors=True)


# ---------------------- Server + Sliding Window --------------------
class PoseServer:
    def __init__(self, window_size: int = 8, cadence_ms: int = 250, debug_save: bool = False, debug_root: str = "sessions"):
        self.window_size = window_size
        self.cadence_ms = cadence_ms
        self.buffers: Dict[str, Deque[Frame]] = {}
        self.last_run_ms: Dict[str, int] = {}
        self.inference_busy: Dict[str, bool] = {}
        self.debug_save = debug_save
        self.debug_root = debug_root

        # Lazy: one session if none provided
        self.default_session = "default"

        # Load VGGT once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(_URL, map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval().to(self.device)
        cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        self.autocast_dtype = torch.bfloat16 if cc >= 8 else torch.float16

    # ------------- Session helpers -------------
    def _get_session(self, session_id: Optional[str]) -> str:
        sid = session_id or self.default_session
        if sid not in self.buffers:
            self.buffers[sid] = deque(maxlen=self.window_size)
            self.last_run_ms[sid] = 0
            self.inference_busy[sid] = False
        return sid

    def reset_session(self, session_id: Optional[str]) -> None:
        sid = session_id or self.default_session
        self.buffers[sid] = deque(maxlen=self.window_size)
        self.last_run_ms[sid] = 0
        self.inference_busy[sid] = False
        print(f"[INFO] Reset session '{sid}'.")

    # ------------- Frame ingest -------------
    def add_frame(self, sid: str, frame: Frame) -> None:
        self.buffers[sid].append(frame)

    # ------------- Throttle / coalesce -------------
    def _should_run(self, sid: str, now_ms: int) -> bool:
        if self.inference_busy[sid]:
            return False
        return (now_ms - self.last_run_ms[sid]) >= self.cadence_ms

    # ------------- Main inference trigger -------------
    def maybe_run_inference(self, sid: str) -> Optional[Dict]:
        now_ms = int(time.time() * 1000)
        if not self._should_run(sid, now_ms):
            return None
        if len(self.buffers[sid]) < 2:
            # Need at least 2 frames for a stable relative pose in many methods
            return None

        self.inference_busy[sid] = True
        t0 = time.time()
        frames = list(self.buffers[sid])  # oldest -> newest

        # Call VGGT (now wired to real model)
        poses = vggt_infer_poses(
            frames=frames,
            device=self.device,
            model=self.model,
            autocast_dtype=self.autocast_dtype,
            sid=sid,
            debug_save=self.debug_save,
            debug_root=self.debug_root,
        )
        assert len(poses) == len(frames)
        R, t = poses[-1]

        t1 = time.time()
        self.last_run_ms[sid] = int(now_ms)
        self.inference_busy[sid] = False

        # Print a clean, human-readable block
        latest = frames[-1]
        dt_infer = (t1 - t0) * 1000.0
        print("\n" + "-" * 64)
        print(f"[{time.strftime('%H:%M:%S')}] session={sid} frame={latest.frame_id} buffer={len(frames)}")
        print(f"VGGT inference: {dt_infer:.1f} ms")
        print("pose(R|t), camera-to-world, RH coords:")
        np.set_printoptions(precision=3, suppress=True)
        print("R=\n" + str(R))
        print("t= " + str(t) + "  (m, up-to-scale unless metric anchored)")
        # If you compute reprojection error, print here too
        print("-" * 64)

        # Return a minimal JSON (handy for client debugging)
        return {
            "frame_id": latest.frame_id,
            "timestamp_ms": latest.timestamp_ms,
            "buffer": len(frames),
            "R": R.tolist(),
            "t": t.tolist(),
            "inference_ms": dt_infer,
        }


# ------------------------------ API -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER: PoseServer  # type hint alias
_server: Optional[PoseServer] = None


@app.on_event("startup")
async def _on_startup():
    global _server
    if _server is None:
        # Defaults get overridden via CLI main()
        _server = PoseServer(window_size=8, cadence_ms=250)
        print("[READY] Pose server initialized with window=8, cadence=250ms")


@app.post("/frame")
async def post_frame(
    image: UploadFile = File(...),
    metadata: str = Form(...),
):
    """Receive one frame + intrinsics, enqueue, maybe run inference.

    Returns last-pose JSON if an inference just ran; otherwise returns a
    lightweight ack. This keeps network traffic low while allowing the sender
    to get occasional pose snapshots.
    """
    assert _server is not None

    # Parse metadata
    try:
        frame_id, timestamp_ms, intrinsics, session_id = parse_metadata(metadata)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad metadata: {e}"})

    sid = _server._get_session(session_id)

    # Decode image into RGB np array
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw))
        img_rgb = pil_to_numpy_rgb(pil)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad image: {e}"})

    # Build camera matrix
    K = make_K(intrinsics)

    # Construct frame and push to buffer
    frame = Frame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        img_rgb=img_rgb,
        K=K,
        intrinsics=intrinsics,
    )
    _server.add_frame(sid, frame)

    # Attempt a run (throttled)
    out = _server.maybe_run_inference(sid)
    if out is not None:
        return JSONResponse(status_code=200, content={"status": "ok", "pose": out})
    else:
        # Ack only; no inference happened this call
        return JSONResponse(status_code=202, content={"status": "queued"})


@app.post("/reset")
async def reset_session(payload: Dict[str, str]):
    assert _server is not None
    _server.reset_session(payload.get("session_id"))
    return {"status": "reset"}


# ------------------------------ Main ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--cadence-ms", type=int, default=250)
    parser.add_argument("--debug-save", action="store_true", help="Keep per-session images under sessions/<sid>/images")
    parser.add_argument("--debug-root", type=str, default="sessions")
    args = parser.parse_args()

    global _server
    _server = PoseServer(window_size=args.window, cadence_ms=args.cadence_ms, debug_save=args.debug_save, debug_root=args.debug_root)
    print(f"[READY] Pose server initialized with window={args.window}, cadence={args.cadence_ms}ms, debug_save={args.debug_save}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
