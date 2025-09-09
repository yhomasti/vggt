"""
Test Realtime VGGT Pose Server (optimized)
------------------------------------------

Key speed-ups vs previous version:
 - In-memory preprocessing (no disk PNG round-trip)
 - Warmup pass on startup (CUDA/cuDNN JIT, allocator priming)
 - Gradio-like resize (width=518), white padding to common HxW
 - AMP autocast + TF32 (when available), cuDNN benchmark on
 - Smaller window/cadence defaults for dev

API is unchanged.
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

import os
import torch
import torch.nn.functional as F
from torchvision import transforms as TF

# --- VGGT imports (repo local) ---
import sys
sys.path.append("vggt/")
from vggt.models.vggt import VGGT
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


def pil_from_numpy_rgb(arr: np.ndarray) -> Image.Image:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got {arr.shape}")
    return Image.fromarray(arr, mode="RGB")


def make_K(intr: Intrinsics) -> np.ndarray:
    K = np.array([[intr.fx, 0.0, intr.cx],
                  [0.0, intr.fy, intr.cy],
                  [0.0,  0.0,   1.0 ]], dtype=np.float32)
    return K


# ------------------- Gradio-like In-Memory Preproc -----------------
# Matches semantics of vggt.utils.load_fn.load_and_preprocess_images(mode="crop"):
# - Set WIDTH=518, keep aspect (BICUBIC)
# - If resulting height > 518, center-crop to 518
# - If images in the batch have different (H, W), white-pad to the max H/W
# - Pad to be divisible by 14 (safe for the model), using white value
TO_TENSOR = TF.ToTensor()  # [0,1]
TARGET_W = 518             # Gradio uses 518 width
TARGET_H = 518             # crop height if exceeding this
DIVIS = 14                 # make dims divisible by 14

def _resize_crop_tensor(img_pil: Image.Image) -> torch.Tensor:
    w, h = img_pil.size
    if w <= 0 or h <= 0:
        raise ValueError("Bad image size")

    scale = TARGET_W / float(w)
    new_w = TARGET_W
    new_h = int(round(h * scale))
    img = img_pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
    t = TO_TENSOR(img)  # shape (3, new_h, new_w), values [0,1]

    if new_h > TARGET_H:
        # center-crop to 518 high
        start = (new_h - TARGET_H) // 2
        t = t[:, start:start + TARGET_H, :]
    # else: keep new_h (<518), no pad yet (we batch-pad later)
    return t  # (3, H', 518)

def preprocess_batch_from_numpy(frames: List[Frame], device: torch.device) -> torch.Tensor:
    """
    frames: oldest -> newest
    returns: (S, 3, H, W) float32 on 'device', [0,1] white-padded.
    """
    ts: List[torch.Tensor] = []
    max_h, max_w = 0, 0
    for fr in frames:
        t = _resize_crop_tensor(pil_from_numpy_rgb(fr.img_rgb))  # CPU tensor
        ts.append(t)
        max_h = max(max_h, t.shape[1])
        max_w = max(max_w, t.shape[2])

    # ensure common shape + divisible by 14 (pad with white = 1.0)
    target_h = ((max_h + DIVIS - 1) // DIVIS) * DIVIS
    target_w = ((max_w + DIVIS - 1) // DIVIS) * DIVIS

    padded: List[torch.Tensor] = []
    for t in ts:
        h_pad = target_h - t.shape[1]
        w_pad = target_w - t.shape[2]
        # center vertically, already exact width=518 so w_pad mostly 0 unless DIVIS forces it
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        if h_pad > 0 or w_pad > 0:
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)  # white
        padded.append(t)

    batch = torch.stack(padded, dim=0)  # (S, 3, H, W) CPU
    return batch.to(device, non_blocking=True)


# ----------------------- VGGT Pose Inference -----------------------

def vggt_infer_poses(
    frames: List[Frame],
    device: torch.device,
    model: VGGT,
    autocast_dtype: torch.dtype,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """
    Run VGGT over the provided window and return (R,t) for each frame.
    Returns poses and a timing dict.
    """
    t0 = time.time()
    imgs = preprocess_batch_from_numpy(frames, device)  # (S,3,H,W)
    t1 = time.time()

    # Forward
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=autocast_dtype):
            preds = model(imgs)
    t2 = time.time()

    # Decode extrinsics/intrinsics (need H,W)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], imgs.shape[-2:])
    extrinsic = extrinsic.detach().cpu().numpy().squeeze(0)  # (S, 4,4) or (S,3,4)
    # intrinsic not used here, but could be returned if needed

    outs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(extrinsic.shape[0]):
        Ei = extrinsic[i]
        if Ei.shape == (3, 4):
            H = np.eye(4, dtype=np.float32)
            H[:3, :4] = Ei
        elif Ei.shape == (4, 4):
            H = Ei.astype(np.float32)
        else:
            raise ValueError(f"Bad extrinsic shape: {Ei.shape}")
        Twc = np.linalg.inv(H)  # world-from-camera
        Rwc = Twc[:3, :3].astype(np.float32)
        twc = Twc[:3, 3].astype(np.float32)
        outs.append((Rwc, twc))

    t3 = time.time()
    timing = {
        "preproc_ms": (t1 - t0) * 1000.0,
        "forward_ms": (t2 - t1) * 1000.0,
        "decode_ms": (t3 - t2) * 1000.0,
        "total_ms":  (t3 - t0) * 1000.0,
    }
    return outs, timing


# ---------------------- Server + Sliding Window --------------------
class PoseServer:
    def __init__(self, window_size: int = 3, cadence_ms: int = 2000,
                 debug_save: bool = False, debug_root: str = "sessions"):
        self.window_size = window_size
        self.cadence_ms = cadence_ms
        self.buffers: Dict[str, Deque[Frame]] = {}
        self.last_run_ms: Dict[str, int] = {}
        self.inference_busy: Dict[str, bool] = {}
        self.debug_save = debug_save
        self.debug_root = debug_root

        # Lazy: one session if none provided
        self.default_session = "default"

        # ---- Model + CUDA opts ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        if self.device.type == "cuda":
            # Allow TF32 on Ampere+; big speedup with negligible quality hit
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        # Load VGGT once
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(_URL, map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval().to(self.device)
        cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        self.autocast_dtype = torch.bfloat16 if cc >= 8 else torch.float16

        # ---- Warmup ----
        self._warmup()

    def _warmup(self):
        # One dummy forward at typical size (S=2, 392x518) to JIT & pick algos
        h = 392   # divisible by 14
        w = 518   # divisible by 14
        dummy = torch.zeros((2, 3, h, w), device=self.device, dtype=torch.float32)
        t0 = time.time()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda"),
                                         dtype=self.autocast_dtype):
                _ = self.model(dummy)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        print(f"[WARMUP] First forward took {(time.time()-t0)*1000:.1f} ms")

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
            # VGGT multi-view is happier with >=2 frames
            return None

        self.inference_busy[sid] = True
        frames = list(self.buffers[sid])  # oldest -> newest

        poses, timing = vggt_infer_poses(
            frames=frames,
            device=self.device,
            model=self.model,
            autocast_dtype=self.autocast_dtype,
        )
        R, t = poses[-1]
        self.last_run_ms[sid] = int(now_ms)
        self.inference_busy[sid] = False

        latest = frames[-1]
        print("\n" + "-" * 64)
        print(f"[{time.strftime('%H:%M:%S')}] session={sid} frame={latest.frame_id} buffer={len(frames)}")
        print(f"VGGT inference total: {timing['total_ms']:.1f} ms "
              f"(preproc {timing['preproc_ms']:.1f} | forward {timing['forward_ms']:.1f} | decode {timing['decode_ms']:.1f})")
        print("pose(R|t), camera-to-world, RH coords:")
        np.set_printoptions(precision=3, suppress=True)
        print("R=\n" + str(R))
        print("t= " + str(t) + "  (m, up-to-scale unless metric anchored)")
        print("-" * 64)

        return {
            "frame_id": latest.frame_id,
            "timestamp_ms": latest.timestamp_ms,
            "buffer": len(frames),
            "R": R.tolist(),
            "t": t.tolist(),
            "inference_ms": timing["total_ms"],
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

SERVER: PoseServer
_server: Optional[PoseServer] = None


@app.on_event("startup")
async def _on_startup():
    global _server
    if _server is None:
        _server = PoseServer(window_size=3, cadence_ms=2000)
        print("[READY] Pose server initialized with window=3, cadence=2000ms")


@app.post("/frame")
async def post_frame(
    image: UploadFile = File(...),
    metadata: str = Form(...),
):
    """Receive one frame + intrinsics, enqueue, maybe run inference."""
    assert _server is not None

    # Parse metadata
    try:
        frame_id, timestamp_ms, intrinsics, session_id = parse_metadata(metadata)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad metadata: {e}"})

    sid = _server._get_session(session_id)

    # Decode image into RGB np array (No disk!)
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.asarray(pil)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad image: {e}"})

    # Build camera matrix (currently unused downstream, but kept for completeness)
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
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--cadence-ms", type=int, default=2000)
    parser.add_argument("--debug-save", action="store_true",
                        help="(kept for parity; not used in fast path)")
    parser.add_argument("--debug-root", type=str, default="sessions")
    args = parser.parse_args()

    global _server
    _server = PoseServer(window_size=args.window,
                         cadence_ms=args.cadence_ms,
                         debug_save=args.debug_save,
                         debug_root=args.debug_root)
    print(f"[READY] Pose server initialized with window={args.window}, "
          f"cadence={args.cadence_ms}ms")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
