"""PolySplat side of correctness_vs_3dgs.

Dumps rendered images from PolySplat into
<out_dir>/<scene>__<img_name>.npy as uint8 (H, W, 3), one per selected camera.

Emits JSON lines on stdout; the driver matches them with the 3DGS side by
(scene, img_name).

PYTHONPATH must point at the PolySplat tree before invocation (driver handles this; default = the PolySplat repo root).
"""
import argparse
import copy
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def find_ply(model_path, iteration):
    for it in (iteration, 30000, 7000, 15000):
        p = os.path.join(model_path, "point_cloud", f"iteration_{it}", "point_cloud.ply")
        if os.path.exists(p):
            return p
    symlink = os.path.join(model_path, "point_cloud.ply")
    if os.path.exists(symlink):
        return symlink
    raise FileNotFoundError(f"no point_cloud.ply under {model_path}")


def safe_dump_name(name):
    return str(name).replace("/", "__").replace("\\", "__")


def select_cameras(cams, which):
    if which == "first":
        return cams[:1]
    if which == "all":
        return cams
    if isinstance(which, int):
        return cams[:which]
    if isinstance(which, str):
        if which.isdigit():
            return cams[: int(which)]
        if which.startswith("uniform-"):
            n = int(which.split("-", 1)[1])
        elif which.startswith("stride:"):
            n = int(which.split(":", 1)[1])
        else:
            raise ValueError(f"bad cameras spec: {which!r}")
        N = len(cams)
        if n <= 0 or n >= N:
            return cams
        step = (N - 1) / (n - 1) if n > 1 else 0
        idxs = sorted({int(round(i * step)) for i in range(n)})
        return [cams[i] for i in idxs]
    raise ValueError(f"bad cameras spec: {which!r}")


def apply_llffhold(cams, hold):
    if hold in (None, False, "", "none", "None", "all"):
        return cams
    hold = int(hold)
    if hold <= 1:
        return cams
    return [c for i, c in enumerate(cams) if i % hold == 0]


def is_oom_error(exc):
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def scaled_camera(cam, scale):
    out = copy.copy(cam)
    out.width = max(1, int(round(cam.width * scale)))
    out.height = max(1, int(round(cam.height * scale)))
    out.focal_x = cam.focal_x * (out.width / cam.width)
    out.focal_y = cam.focal_y * (out.height / cam.height)
    return out


def tensor_to_uint8_np(out):
    # Baseline-era int8 vs HEAD uint8 — same bytes.
    if out.dtype == torch.int8:
        out = out.view(torch.uint8)
    return out.cpu().numpy()


def resize_uint8(img, width, height):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    pil = Image.fromarray(img, mode="RGB")
    pil = pil.resize((width, height), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def render_with_oom_fallback(rast, scene, scene_name, cam, bg_color, variant):
    try:
        out = rast.forward(scene, cam, bg_color, render_variant=variant)
        torch.cuda.synchronize()
        return tensor_to_uint8_np(out), False
    except Exception as e:
        if not is_oom_error(e):
            raise
        torch.cuda.empty_cache()
        emit({"type": "warning", "side": "polysplat", "scene": scene_name,
              "img_name": cam.img_name, "warning": "oom_retry_half_res"})
        half_cam = scaled_camera(cam, 0.5)
        out = rast.forward(scene, half_cam, bg_color, render_variant=variant)
        torch.cuda.synchronize()
        arr = tensor_to_uint8_np(out)
        return resize_uint8(arr, cam.width, cam.height), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenes-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    scenes = json.loads(args.scenes_json)
    os.makedirs(args.out_dir, exist_ok=True)

    # Import AFTER sys.path is set up by driver
    from example import Scene, Camera, Rasterizer  # noqa: E402

    device = torch.device("cuda:0")
    bg_color = torch.zeros(3, dtype=torch.float32)
    cam_spec = cfg.get("cameras", "first")
    variant = cfg.get("polysplat_variant", "preranges_smem_persistent_lite_e2e")
    iteration = int(cfg.get("iteration", 30000))
    max_nr = int(cfg.get("max_num_rendered", 2 ** 27))
    max_nt = int(cfg.get("max_num_tiles", 2 ** 20))

    emit({"type": "meta", "side": "polysplat", "variant": variant,
          "scene_count": len(scenes)})

    for scene_path in scenes:
        name = os.path.basename(scene_path.rstrip("/"))
        try:
            ply = find_ply(scene_path, iteration)
            scene = Scene(device)
            with contextlib.redirect_stdout(io.StringIO()):
                scene.loadPly(ply)
            with open(os.path.join(scene_path, "cameras.json")) as f:
                cams_json = json.loads(f.read())
            cams = [Camera(c) for c in cams_json]
        except Exception as e:
            emit({"type": "error", "scene": name, "error": str(e)})
            continue

        holded = apply_llffhold(cams, cfg.get("llffhold", 8))
        selected = select_cameras(holded, cam_spec)
        rast = Rasterizer(scene, max_nr, max_nt)

        for cam in selected:
            try:
                arr, used_half_fallback = render_with_oom_fallback(
                    rast, scene, name, cam, bg_color, variant)
            except Exception as e:
                emit({"type": "error", "scene": name,
                      "img_name": cam.img_name, "error": str(e)})
                continue
            fname = f"{name}__{safe_dump_name(cam.img_name)}.npy"
            path = os.path.join(args.out_dir, fname)
            np.save(path, arr)
            emit({"type": "image", "side": "polysplat",
                  "scene": name, "img_name": cam.img_name,
                  "camera_id": getattr(cam, "id", ""),
                  "width": cam.width, "height": cam.height,
                  "num_vertex": scene.num_vertex, "path": path,
                  "oom_half_res_fallback": used_half_fallback})

        del rast, scene
        torch.cuda.empty_cache()

    emit({"type": "done", "side": "polysplat"})


if __name__ == "__main__":
    main()
