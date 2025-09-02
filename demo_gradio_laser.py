# demo_gradio_laser.py
# Run: python demo_gradio_laser.py
# Requires: pip install -r requirements.txt; pip install -r requirements_demo.txt

import os
import io
import numpy as np
from PIL import Image, ImageDraw

import torch
import gradio as gr

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# =========================
# Geometry helpers (laser)
# =========================

def _decompose_extrinsic(extrinsic):
    """Extrinsic maps world->camera. Return R, t, and camera center C_w."""
    E = np.asarray(extrinsic)
    if E.shape == (4, 4):
        R, t = E[:3, :3], E[:3, 3]
    elif E.shape == (3, 4):
        R, t = E[:, :3], E[:, 3]
    else:
        raise ValueError("Extrinsic must be 3x4 or 4x4")
    C_w = -R.T @ t
    return R, t, C_w

def _project_world_point(K, R, t, Xw, im_w, im_h):
    """Project world-space point Xw to pixel coords in this camera."""
    Xc = R @ Xw + t
    Z = Xc[2]
    if Z <= 1e-6:
        return None, None, False
    uvw = K @ Xc
    u, v = uvw[0] / Z, uvw[1] / Z
    in_bounds = (0 <= u < im_w) and (0 <= v < im_h)
    return (float(u), float(v)), float(Z), in_bounds

def _backproject_pixel_to_world(K, R, t, depth_map, uv):
    """Backproject pixel (u,v) with Z-depth into world coords."""
    u, v = uv
    h, w = depth_map.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    d = float(depth_map[vi, ui])
    if d <= 0:
        return None
    Kinv = np.linalg.inv(K)
    x_cam = d * (Kinv @ np.array([u, v, 1.0], dtype=np.float64))
    # Xw = R^T (Xc - t)
    Xw = R.T @ (x_cam - t)
    return Xw

def _beams_for_click(source_cam, uv_src, cameras, depth_maps, depth_tol_ratio=0.05):
    """
    Given a click (u,v) on source_cam, compute world point Xw and per-camera 2D lines:
    for each camera j that sees Xw, returns line from projected source camera center to projected Xw.
    """
    K_s = source_cam["K"]
    R_s, t_s, Cw_s = _decompose_extrinsic(source_cam["extrinsic"])
    dm_s = depth_maps[source_cam["id"]]
    Xw = _backproject_pixel_to_world(K_s, R_s, t_s, dm_s, uv_src)
    if Xw is None:
        return None, {}

    overlays = {}
    for cam in cameras:
        cam_id = cam["id"]
        K = cam["K"]
        R, t, _Cw = _decompose_extrinsic(cam["extrinsic"])
        w, h = cam["image_size"]

        # project world point
        uv_pt, Zpt, in_bounds_pt = _project_world_point(K, R, t, Xw, w, h)
        if not in_bounds_pt:
            continue

        # project source camera center into this camera
        uv_srcCam, Zsc, in_bounds_sc = _project_world_point(K, R, t, Cw_s, w, h)
        if not in_bounds_sc:
            continue

        # occlusion check
        dm = depth_maps.get(cam_id)
        if dm is not None:
            ui, vi = int(round(uv_pt[0])), int(round(uv_pt[1]))
            if not (0 <= ui < w and 0 <= vi < h):
                continue
            d_here = float(dm[vi, ui])
            if d_here <= 0:
                continue
            tol = max(depth_tol_ratio * Zpt, 0.01)
            if abs(Zpt - d_here) > tol:
                continue

        overlays[cam_id] = {"p0": uv_srcCam, "p1": uv_pt}
    return Xw, overlays


# =========================
# VGGT / Gradio wiring
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

# lazy global model to avoid reloading on every call
_GLOBAL_MODEL = None

def _get_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)
    return _GLOBAL_MODEL

def _pil_from_file(fileobj):
    # Gradio passes tempfile paths as strings or file-like. Normalize to PIL
    if hasattr(fileobj, "read"):
        return Image.open(fileobj).convert("RGB")
    return Image.open(str(fileobj)).convert("RGB")

def reconstruct(images):
    """
    images: list of uploaded files from Gradio
    Returns:
      gallery_images: list of PIL for display
      state: dict with cameras, depth_maps, and copies of display images
      source_cam_choices: dropdown choices
      status: text
    """
    if not images or len(images) == 0:
        return [], {}, gr.update(choices=[], value=None), "No images provided."

    # Load original images (for display) and file paths (for preprocessing)
    pil_imgs = [_pil_from_file(f) for f in images]

    # Save display copies (we’ll draw on fresh copies per click)
    display_images = [img.copy() for img in pil_imgs]

    # Preprocess for VGGT
    # Write PILs to in-memory files so we can reuse load_and_preprocess_images
    tmp_paths = []
    for i, im in enumerate(pil_imgs):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        tmp_path = f"__tmp_vggt_{i}.png"
        with open(tmp_path, "wb") as f:
            f.write(buf.read())
        tmp_paths.append(tmp_path)

    model = _get_model()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=DTYPE):
            imgs = load_and_preprocess_images(tmp_paths).to(DEVICE)  # [N,3,H,W]
            # Add batch dimension for aggregator
            imgs_b1 = imgs[None]  # [B=1,N,3,H,W]
            aggregated_tokens_list, ps_idx = model.aggregator(imgs_b1)
            # Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # OpenCV convention, camera-from-world
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
            # Depth
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, imgs_b1, ps_idx)

    # Cleanup tmp files
    for p in tmp_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    # Convert tensors to numpy and build state
    # Shapes:
    #   intrinsic: [B=1, N, 3, 3]
    #   extrinsic: [B=1, N, 3, 4] (or 4x4 depending on utils)
    #   depth_map: [B=1, N, H, W]
    intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
    extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
    depth_np = depth_map.squeeze(0).detach().cpu().numpy()

    N = intrinsic_np.shape[0]
    cameras = []
    depth_maps = {}
    for i in range(N):
        H, W = depth_np[i].shape[-2], depth_np[i].shape[-1]
        # Note image_size passed as (W,H)
        cameras.append({
            "id": i,
            "K": intrinsic_np[i],
            "extrinsic": extrinsic_np[i],
            "image_size": (W, H),
        })
        depth_maps[i] = depth_np[i]

    state = {
        "cameras": cameras,
        "depth_maps": depth_maps,
        "display_images": display_images,  # PILs used by the gallery
    }

    source_cam_choices = [f"{i}" for i in range(N)]
    status = f"Reconstructed {N} view(s) — device={DEVICE}, dtype={DTYPE}"
    return display_images, state, gr.update(choices=source_cam_choices, value=source_cam_choices[0]), status


def on_click(source_cam_idx_str, click_event, state):
    """
    source_cam_idx_str: dropdown selected string index
    click_event: gr.Image returns dict with {"x": int, "y": int, "width": int, "height": int}
                 coords are in DISPLAY coordinates of the active image shown in 'source_view'
    state: dict from reconstruct()
    Returns: updated gallery images (with overlays), and a source image with a dot at click
    """
    if not state or "cameras" not in state:
        return gr.update(), gr.update()

    cams = state["cameras"]
    depths = state["depth_maps"]
    disp_imgs = state["display_images"]

    if source_cam_idx_str is None or source_cam_idx_str == "":
        return gr.update(), gr.update()

    cam_idx = int(source_cam_idx_str)
    if cam_idx < 0 or cam_idx >= len(cams):
        return gr.update(), gr.update()

    # The active source image is the cam_idx-th display image
    src_disp = disp_imgs[cam_idx]
    W_disp, H_disp = src_disp.size

    # Convert display click to original pixel coords (depth/intrinsics space)
    W_orig, H_orig = cams[cam_idx]["image_size"]
    u = click_event["x"] * (W_orig / W_disp)
    v = click_event["y"] * (H_orig / H_disp)

    # Compute beams (2D overlays for every camera that sees the 3D point)
    Xw, overlays = _beams_for_click(cams[cam_idx], (u, v), cams, depths)

    # Draw overlays on copies for gallery
    out_imgs = []
    for j, base_img in enumerate(disp_imgs):
        im2 = base_img.copy().convert("RGBA")
        draw = ImageDraw.Draw(im2)
        Wd, Hd = im2.size
        Wj, Hj = cams[j]["image_size"]
        if j in overlays:
            p0 = overlays[j]["p0"]  # original pixel coords in cam j
            p1 = overlays[j]["p1"]
            u0 = p0[0] * (Wd / Wj); v0 = p0[1] * (Hd / Hj)
            u1 = p1[0] * (Wd / Wj); v1 = p1[1] * (Hd / Hj)
            draw.line([(u0, v0), (u1, v1)], width=3, fill=(255, 0, 0, 255))
            r = 4
            draw.ellipse([u1 - r, v1 - r, u1 + r, v1 + r], outline=(255, 0, 0, 255), width=2)
        out_imgs.append(im2)

    # Also return a source preview with a small dot at the clicked pixel
    src_preview = disp_imgs[cam_idx].copy().convert("RGBA")
    draw_src = ImageDraw.Draw(src_preview)
    u_disp = click_event["x"]; v_disp = click_event["y"]
    r = 5
    draw_src.ellipse([u_disp - r, v_disp - r, u_disp + r, v_disp + r], fill=(0, 255, 0, 200), outline=(0, 0, 0, 200), width=2)

    return out_imgs, src_preview


with gr.Blocks() as demo:
    gr.Markdown("# VGGT — Multi-view ‘Laser Beam’ Click Demo")

    with gr.Row():
        with gr.Column(scale=1):
            imgs_in = gr.Files(file_types=["image"], file_count="multiple", label="Upload images (1+)")
            run_btn = gr.Button("Reconstruct", variant="primary")
            status = gr.Markdown("")

            source_cam_idx = gr.Dropdown(choices=[], label="Source camera (click target)")
            source_view = gr.Image(label="Click here to place beam target", interactive=True)

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Views with beams", allow_preview=False, height=600, columns=3)

    state = gr.State({})

    def _after_reconstruct(files):
        gallery_imgs, st, choices, stat = reconstruct(files)
        # default source image is the first
        src_img = gallery_imgs[0] if gallery_imgs else None
        # keep a copy of display images in state for later clicks (already in st)
        return gallery_imgs, st, choices, src_img, stat

    run_btn.click(
        _after_reconstruct,
        inputs=[imgs_in],
        outputs=[gallery, state, source_cam_idx, source_view, status]
    )

    # clicking on source_view returns dict with x,y,width,height
    source_view.select(
        on_click,
        inputs=[source_cam_idx, source_view, state],
        outputs=[gallery, source_view]
    )

if __name__ == "__main__":
    demo.launch()
