

from __future__ import annotations
import os
import io
import cv2
import json
import glob
import time
import gc
import shutil
import argparse
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional, Deque

import numpy as np
import torch

# Perf knobs
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# Prefer mem-efficient attention on Windows where flash SDP often isn't available
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
except Exception:
    pass

# --- vggt imports ---
import sys
sys.path.append("vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------- helpers -----------------------------

def _mat_to_euler_xyz_deg(R: np.ndarray):
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _cams_from_extri_intri(E: np.ndarray, K: np.ndarray, names: List[str]):
    # Normalize shapes to (S, 3, 4) and (S, 3, 3)
    if E.ndim == 4 and E.shape[0] == 1:
        E = E[0]
    if K.ndim == 4 and K.shape[0] == 1:
        K = K[0]
    if E.ndim != 3 or E.shape[-2:] != (3, 4):
        raise ValueError(f"Extrinsic must be (S,3,4); got {E.shape}")
    if K.ndim != 3 or K.shape[-2:] != (3, 3):
        raise ValueError(f"Intrinsic must be (S,3,3); got {K.shape}")

    cams = []
    S = E.shape[0]
    for i in range(S):
        Ei = E[i]  # (3,4)
        H = np.eye(4, dtype=Ei.dtype)
        H[:3, :4] = Ei

        # VGGT extrinsic is camera-from-world; invert to world-from-camera
        Twc = np.linalg.inv(H)
        Rwc, t = Twc[:3, :3], Twc[:3, 3]
        roll, pitch, yaw = _mat_to_euler_xyz_deg(Rwc)

        fx = float(K[i][0, 0]); fy = float(K[i][1, 1])
        cx = float(K[i][0, 2]); cy = float(K[i][1, 2])
        cams.append({
            "index": int(i),
            "image": os.path.basename(names[i]) if i < len(names) else f"{i:06d}.png",
            "position_m": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "euler_xyz_deg": {"roll": roll, "pitch": pitch, "yaw": yaw},
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        })
    return cams

# --------------------------- main class ----------------------------

class CameraOnlyVGGT:
    def __init__(self, size: int = 384, window: int = 4, dtype: Optional[torch.dtype] = None, compile_model: bool = False):
        """
            size: square short-side (pixels) for preprocessing (requested; patch-aligned internally)
            window: number of most-recent frames to feed per inference
            dtype: torch.bfloat16 (Ada+), torch.float16, or None to auto-pick
            compile_model: optional torch.compile; default False (Windows Dynamo can be flaky)
        """
        self.size = int(size)
        self.window = int(window)
        self.queue: Deque[str] = deque(maxlen=self.window)

        # camera head only
        self.model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected or missing:
            print(f"[VGGT] Loaded weights with strict=False → missing={len(missing)}, unexpected={len(unexpected)}")
        self.model.eval().to(DEVICE)

        if dtype is None:
            if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        else:
            self.dtype = dtype

        # optional compile (disabled by default)
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                print(f"[VGGT] torch.compile disabled (error: {e})")

        # warmup with patch-size–aligned dummy to avoid internal asserts
        if DEVICE == "cuda":
            patch_h, patch_w = self._get_patch_hw()
            H = self._align_to_patch(self.size, patch_h)
            W = self._align_to_patch(self.size, patch_w)
            dummy = torch.zeros(2, 3, H, W, device=DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda"), dtype=self.dtype):
                _ = self.model(dummy)
            torch.cuda.synchronize()

    # ---- internal: model patch + size alignment ----
    def _get_patch_hw(self) -> Tuple[int, int]:
        try:
            ps = getattr(self.model.aggregator.patch_embed, "patch_size", (16, 16))
        except Exception:
            ps = (16, 16)
        if isinstance(ps, (tuple, list)):
            return int(ps[0]), int(ps[1])
        return int(ps), int(ps)

    @staticmethod
    def _align_to_patch(size: int, patch: int) -> int:
        if patch <= 0:
            return int(size)
        return max(patch, int(round(size / patch) * patch))

    # ---------------- live-time helpers ----------------
    def push_path(self, img_path: str):
        self.queue.append(img_path)

    def _stack(self) -> Tuple[torch.Tensor, List[str]]:
        names = list(self.queue)
        if len(names) == 0:
            raise ValueError("No frames in queue")

        # align requested size to the model's patch size to avoid assert
        patch_h, patch_w = self._get_patch_hw()
        tgt_h = self._align_to_patch(self.size, patch_h)
        tgt_w = self._align_to_patch(self.size, patch_w)
        target_size = self._align_to_patch(int(round((tgt_h + tgt_w) * 0.5)), max(patch_h, patch_w))

        out = load_and_preprocess_images_square(names, target_size=target_size)
        imgs = out[0] if isinstance(out, (tuple, list)) else out
        imgs = imgs.to(DEVICE)
        return imgs, names

    @torch.no_grad()
    def infer_latest(self) -> dict:
        """Run VGGT on the current window; return latest frame's camera + timings."""
        if len(self.queue) == 0:
            raise ValueError("No frames queued")

        timings = {}
        t0 = time.time()

        t_load0 = time.time()
        images, names = self._stack()
        timings["load_preprocess_s"] = round(time.time() - t_load0, 4)

        t_inf0 = time.time()
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda"), dtype=self.dtype):
            preds = self.model(images)
        timings["inference_s"] = round(time.time() - t_inf0, 4)

        H, W = images.shape[-2:]
        t_pose0 = time.time()
        extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
        timings["pose_decode_s"] = round(time.time() - t_pose0, 4)

        # Squeeze a leading batch dimension if present (common: (1,S,3,4)/(1,S,3,3))
        extri_np = extri.detach().cpu().numpy()
        intri_np = intri.detach().cpu().numpy()
        if extri_np.ndim == 4 and extri_np.shape[0] == 1:
            extri_np = extri_np[0]
        if intri_np.ndim == 4 and intri_np.shape[0] == 1:
            intri_np = intri_np[0]

        cams = _cams_from_extri_intri(extri_np, intri_np, names)

        timings["camera_extract_s"] = 0.0
        timings["num_frames"] = len(cams)
        timings["total_s"] = round(time.time() - t0, 4)

        latest = cams[-1]
        return {"cameras": cams, "latest": latest, "timings": timings}

# --------------------------- CLI / UI ------------------------------

def run_folder(images_dir: str, size: int, window: int, cap: int = 0):
    all_imgs = sorted(glob.glob(os.path.join(images_dir, "*")))
    if cap and cap > 0:
        all_imgs = all_imgs[:cap]
    if not all_imgs:
        raise SystemExit(f"No images found in {images_dir}")

    actual_window = len(all_imgs) if window <= 0 or window > len(all_imgs) else window
    engine = CameraOnlyVGGT(size=size, window=window)

    for p in all_imgs:
        engine.push_path(p)
    out = engine.infer_latest()

    # Write cameras.json & timings.json in a timestamped folder next to the images
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tgt = os.path.join(os.path.dirname(images_dir.rstrip(os.sep)), f"camera_only_{ts}")
    os.makedirs(tgt, exist_ok=True)
    with open(os.path.join(tgt, "cameras.json"), "w") as f:
        json.dump(out["cameras"], f, indent=2)
    with open(os.path.join(tgt, "timings.json"), "w") as f:
        json.dump(out["timings"], f, indent=2)

    print("Latest frame camera:", json.dumps(out["latest"], indent=2))
    print("Timings:", json.dumps(out["timings"], indent=2))


def launch_ui(size: int, window: int):
    import gradio as gr

    def on_upload(files):
        nonlocal_engine.queue.clear()
        for f in files:
            nonlocal_engine.push_path(f.name if hasattr(f, 'name') else f)
        res = nonlocal_engine.infer_latest()
        log = (
            f"num_frames={res['timings']['num_frames']}, "
            f"load={res['timings']['load_preprocess_s']}s, "
            f"infer={res['timings']['inference_s']}s, "
            f"pose={res['timings']['pose_decode_s']}s, "
            f"total={res['timings']['total_s']}s "
            + json.dumps(res['latest'], indent=2)
        )
        cams_path = os.path.join(os.getcwd(), "cameras.json")
        with open(cams_path, "w") as f:
            json.dump(res["cameras"], f, indent=2)
        return log, cams_path

    nonlocal_engine = CameraOnlyVGGT(size=size, window=window)

    with gr.Blocks() as demo:
        gr.Markdown("# VGGT — Camera Only (no GLB)")
        files = gr.File(file_count="multiple", label="Upload images (ordered)")
        log = gr.Markdown("Upload a few frames; we'll run a single pass on the window.")
        cams_file = gr.File(label="Download cameras.json")
        files.change(on_upload, inputs=[files], outputs=[log, cams_file])
    demo.queue(max_size=20).launch(show_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default=None, help="Folder of images for a quick test")
    parser.add_argument("--size", type=int, default=384, help="Square preprocess size (requested; will be patch-aligned)")
    parser.add_argument("--window", type=int, default=4, help="Frames per inference (keep tiny for latency)")
    parser.add_argument("--cap", type=int, default=0, help="Use only first N images (0 = all)")
    parser.add_argument("--ui", action="store_true", help="Launch a minimal Gradio UI for manual timing")
    args = parser.parse_args()

    if args.ui:
        launch_ui(size=args.size, window=args.window)
    elif args.images_dir:
        run_folder(args.images_dir, size=args.size, window=args.window, cap=args.cap)
    else:
        print("Nothing to do. Pass --images_dir or --ui.")
