# server_vggt.py
# Minimal LAN server that accepts images, runs VGGT, and serves result.glb + cameras.json

import os, glob, shutil, time, threading, json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, abort

import torch
import numpy as np

# --- repo-local imports ---
import sys
sys.path.append("vggt/")  
try:
    from visual_util import predictions_to_glb
except ImportError:
    # some repos place visual_util at root
    from visual_util import predictions_to_glb

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# -----------------------------
#config
# -----------------------------
BASE_DIR   = Path(__file__).parent.resolve()
SESS_ROOT  = BASE_DIR / "sessions"      #all sessions will be saved here
SESS_ROOT.mkdir(exist_ok=True)
PORT       = int(os.environ.get("PORT", 7860))

locks = {}
def get_lock(session_id: str) -> threading.Lock:
    if session_id not in locks:
        locks[session_id] = threading.Lock()
    return locks[session_id]

# -----------------------------
#load VGGT once through GPU if available, otherwise CPU
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[VGGT] Loading model on {device}...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
state = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
model.load_state_dict(state)
model.eval().to(device)
print("[VGGT] Ready.")

# -----------------------------
# Core inference helper
# -----------------------------
def run_vggt_on_session(session_dir: Path, conf_thres: float = 50.0,
                        prediction_mode: str = "Depthmap and Camera Branch") -> dict:
    """
    session_dir/
        images/     <-- input jpg/png frames
        outputs/    <-- we write result.glb + cameras.json here
    """
    images_dir = session_dir / "images"
    outputs_dir = session_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    image_names = sorted(glob.glob(str(images_dir / "*")))
    if not image_names:
        raise ValueError("No images found for this session. Upload first.")

    # preprocess
    imgs = load_and_preprocess_images(image_names).to(device)

    # dtype heuristic
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device=="cuda"), dtype=dtype if device=="cuda" else None):
            preds = model(imgs)  # dict with "pose_enc", "depth", etc.

    # cameras (OpenCV convention: world->camera)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], imgs.shape[-2:])
    preds["extrinsic"] = extrinsic
    preds["intrinsic"] = intrinsic

    # tensors -> numpy (squeeze batch if present)
    out = {}
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().to("cpu").numpy()
            if v.ndim > 0 and v.shape[0] == 1:
                v = np.squeeze(v, axis=0)
        out[k] = v
    # world points from depth (more accurate points than point-map for many scenes)
    depth_map = out["depth"]        # (S,H,W,1) in numpy
    # convert back to torch for unprojection util
    _depth_t   = torch.from_numpy(depth_map)
    _extr_t    = torch.from_numpy(out["extrinsic"])
    _intr_t    = torch.from_numpy(out["intrinsic"])
    point_map_by_unproj = unproject_depth_map_to_point_map(_depth_t, _extr_t, _intr_t)  # (S,H,W,3)
    out["world_points_from_depth"] = point_map_by_unproj.numpy()

    # save predictions (npz) if you want to re-render without rerun
    np.savez(outputs_dir / "predictions.npz", **out, image_names=np.array(image_names, dtype=object))

    # export GLB (THIS MIGHT NOT BE NEEDED IF WE AREN'T DISPLAYING THE GLBS)
    glb_path = outputs_dir / "result.glb"
    glbscene = predictions_to_glb(
        out,
        conf_thres=conf_thres,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        target_dir=str(session_dir),
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=str(glb_path))

    #also export cameras.json 
    cams_json = outputs_dir / "cameras.json"
    write_cameras_json(image_names, out["extrinsic"], out["intrinsic"], cams_json)

    torch.cuda.empty_cache()
    return {
        "glb": str(glb_path),
        "cameras": str(cams_json),
        "num_images": len(image_names)
    }

def write_cameras_json(image_names, extrinsic_np, intrinsic_np, out_path: Path):
    """
    extrinsic_np: (S,4,4) world->camera
    intrinsic_np: (S,3,3)
    Writes cam centers + rotations in model/world frame (the frame defined by this run).
    """
    cams = []
    for i, img in enumerate(image_names):
        R_wc = extrinsic_np[i, :3, :3]    # world->camera (I think?)
        t_wc = extrinsic_np[i, :3,  3]    # world->camera
        # camera center in model/world coords:
        C_w = -R_wc.T @ t_wc
        R_cw = R_wc.T

        cams.append({
            "index": int(i),
            "image": os.path.basename(img),
            "position_m": {"x": float(C_w[0]), "y": float(C_w[1]), "z": float(C_w[2])},
            "R_cam_to_world_3x3": R_cw.tolist(),
            "R_world_to_cam_3x3": R_wc.tolist(),
            "t_world_to_cam_3x1": t_wc.tolist(),
            "K_3x3": intrinsic_np[i].tolist()
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cameras": cams}, f)

# -----------------------------
# Flask app code
# -----------------------------
app = Flask(__name__)

def _session_dirs(session_id: str) -> tuple[Path, Path, Path]:
    sdir = SESS_ROOT / session_id
    idir = sdir / "images"
    odir = sdir / "outputs"
    idir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    return sdir, idir, odir

@app.get("/health")
def health():
    return jsonify(ok=True, device=device)

@app.post("/api/upload/<session_id>/<device_id>")
def upload(session_id, device_id):
    """
    Accepts one or multiple files under 'file' field. Saves into sessions/<session>/images/.
    Filenames are timestamped to avoid collisions.
    """
    sdir, idir, _ = _session_dirs(session_id)
    files = request.files.getlist("file")
    if not files:
        return jsonify(ok=False, error="No files under field 'file'"), 400

    saved = []
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    for idx, f in enumerate(files):
        ext = Path(f.filename).suffix.lower() or ".jpg"
        name = f"{ts}_{device_id}_{idx}{ext}"
        path = idir / name
        f.save(path)
        saved.append(str(path))

    return jsonify(ok=True, saved=saved, count=len(saved))

@app.post("/api/reconstruct/<session_id>")
def reconstruct(session_id):
    """
    Runs VGGT on all images currently in the session. Blocking call; returns when done.
    """
    lock = get_lock(session_id)
    if not lock.acquire(blocking=False):
        return jsonify(ok=False, error="Reconstruction already running"), 409

    try:
        sdir, idir, _ = _session_dirs(session_id)
        if len(list(idir.glob("*"))) == 0:
            return jsonify(ok=False, error="No images uploaded"), 400

        result = run_vggt_on_session(sdir)
        # Build URLs for Unity
        base = request.host_url.rstrip("/")  # e.g., http://10.110.221.32:7860
        glb_url = f"{base}/api/scene/{session_id}/result.glb"
        cams_url = f"{base}/api/scene/{session_id}/cameras.json"
        return jsonify(ok=True, glb_url=glb_url, cameras_url=cams_url, **result)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    finally:
        lock.release()

@app.get("/api/scene/<session_id>/result.glb")
def serve_glb(session_id):
    sdir, _, odir = _session_dirs(session_id)
    path = odir / "result.glb"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="model/gltf-binary")

@app.get("/api/scene/<session_id>/cameras.json")
def serve_cameras(session_id):
    sdir, _, odir = _session_dirs(session_id)
    path = odir / "cameras.json"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="application/json")

@app.post("/api/reset/<session_id>")
def reset(session_id):
    sdir = SESS_ROOT / session_id
    if sdir.exists():
        shutil.rmtree(sdir)
    return jsonify(ok=True)

if __name__ == "__main__":
    # IMPORTANT: listen on all interfaces so HoloLens can reach you
    app.run(host="0.0.0.0", port=PORT, debug=False)
