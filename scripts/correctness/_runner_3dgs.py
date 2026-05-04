"""Reference 3DGS side of correctness_vs_3dgs.

Loads each scene's trained `point_cloud.ply` into the reference
`GaussianModel`, builds minimal view objects from `cameras.json` (the
reference `Scene` loader requires a COLMAP source dir which we don't have
—  bypass it), calls `_render_default`, and dumps each output as uint8
(H, W, 3) `.npy`.

PYTHONPATH must point at the reference 3DGS tree before invocation so that
`from scene.gaussian_model import GaussianModel` and friends resolve.

Forces ``render_backend="default"`` so the reference INRIA 3DGS rasterizer
runs (not any alternative backend that the user's 3DGS checkout might ship
with).
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import types

import numpy as np
import torch
from PIL import Image


_SEPARATE_SH_UNSUPPORTED = False


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


def select_cameras(cams_json, which):
    if which == "first":
        return cams_json[:1]
    if which == "all":
        return cams_json
    if isinstance(which, int):
        return cams_json[:which]
    if isinstance(which, str):
        if which.isdigit():
            return cams_json[: int(which)]
        if which.startswith("uniform-"):
            n = int(which.split("-", 1)[1])
        elif which.startswith("stride:"):
            n = int(which.split(":", 1)[1])
        else:
            raise ValueError(f"bad cameras spec: {which!r}")
        N = len(cams_json)
        if n <= 0 or n >= N:
            return cams_json
        step = (N - 1) / (n - 1) if n > 1 else 0
        idxs = sorted({int(round(i * step)) for i in range(n)})
        return [cams_json[i] for i in idxs]
    raise ValueError(f"bad cameras spec: {which!r}")


def apply_llffhold(cams_json, hold):
    if hold in (None, False, "", "none", "None", "all"):
        return cams_json
    hold = int(hold)
    if hold <= 1:
        return cams_json
    return [c for i, c in enumerate(cams_json) if i % hold == 0]


def build_minicam(cam_json, device, scale=1.0):
    """Construct a reference-3DGS-compatible view object from one cameras.json
    record. Mirrors MiniCam's field set plus the extras that _render_default
    reads (image_name for use_trained_exp).

    cameras.json convention (from utils/camera_utils.py::camera_to_JSON):
        position = camera center in world-space (3,)
        rotation = R_c2w (3x3)              # camera-to-world rotation
        width / height / fx / fy = intrinsics
    """
    # Re-import within the reference 3DGS tree's sys.path.
    from utils.graphics_utils import getProjectionMatrix

    pos = np.asarray(cam_json["position"], dtype=np.float64)
    rot = np.asarray(cam_json["rotation"], dtype=np.float64)  # R_c2w, 3x3
    width0 = int(cam_json["width"])
    height0 = int(cam_json["height"])
    width = max(1, int(round(width0 * scale)))
    height = max(1, int(round(height0 * scale)))
    fx = float(cam_json["fx"]) * (width / width0)
    fy = float(cam_json["fy"]) * (height / height0)

    # Rebuild W2C: W2C[:3,:3] = R_w2c = R_c2w.T, W2C[:3, 3] = -R_w2c @ pos
    R_w2c = rot.T
    t_w2c = -R_w2c @ pos
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c

    # 3DGS stores world_view_transform as the TRANSPOSED W2C (row-major layout
    # with points as row vectors). See Camera.__init__:
    #     self.world_view_transform = torch.tensor(getWorld2View2(...)).transpose(0,1)
    world_view_transform = torch.from_numpy(w2c).transpose(0, 1).to(device=device)

    # FOV from focal length + sensor extent.
    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))

    znear, zfar = 0.01, 100.0
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y).transpose(0, 1).to(device=device)
    full_proj = world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    cam_center = torch.inverse(world_view_transform)[3, :3]

    view = types.SimpleNamespace(
        image_width=width, image_height=height,
        FoVx=fov_x, FoVy=fov_y,
        znear=znear, zfar=zfar,
        world_view_transform=world_view_transform,
        projection_matrix=proj,
        full_proj_transform=full_proj,
        camera_center=cam_center,
        image_name=cam_json.get("img_name", ""),
    )
    return view


def build_pipe():
    # Force the reference INRIA 3DGS kernel.
    return types.SimpleNamespace(
        debug=False,
        antialiasing=False,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        render_backend="default",
    )


def is_oom_error(exc):
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def resize_uint8(img, width, height):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    pil = Image.fromarray(img, mode="RGB")
    pil = pil.resize((width, height), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def identity_exposure(device):
    exposure = torch.zeros((3, 4), dtype=torch.float32, device=device)
    exposure[:3, :3] = torch.eye(3, dtype=torch.float32, device=device)
    return exposure


def ensure_exposures_for_selected_cameras(gaussians, cams_json, device):
    exposures = getattr(gaussians, "pretrained_exposures", None)
    if exposures is None:
        exposures = {}
    ident = identity_exposure(device)
    for cam in cams_json:
        name = cam.get("img_name", "")
        if name and name not in exposures:
            exposures[name] = ident.clone()
    gaussians.pretrained_exposures = exposures


def render_one(view, gaussians, pipe, bg_color, separate_sh, use_exp, _render_default):
    global _SEPARATE_SH_UNSUPPORTED
    use_separate_sh = separate_sh and not _SEPARATE_SH_UNSUPPORTED
    try:
        result = _render_default(
            view, gaussians, pipe, bg_color,
            scaling_modifier=1.0, separate_sh=use_separate_sh,
            override_color=None, use_trained_exp=use_exp,
        )
    except TypeError as e:
        if not use_separate_sh or "unexpected keyword argument 'dc'" not in str(e):
            raise
        _SEPARATE_SH_UNSUPPORTED = True
        emit({"type": "warning", "side": "3dgs",
              "warning": "separate_sh unsupported by this rasterizer; retrying unified SH"})
        result = _render_default(
            view, gaussians, pipe, bg_color,
            scaling_modifier=1.0, separate_sh=False,
            override_color=None, use_trained_exp=use_exp,
        )
    # rendered_image: float32 (3, H, W) in [0, 1].
    img = result["render"]
    img = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
    img = img.permute(1, 2, 0).contiguous().cpu().numpy()  # (H, W, 3)
    return img


def render_with_oom_fallback(cam_json, device, gaussians, pipe, bg_color,
                             separate_sh, use_exp, _render_default):
    try:
        view = build_minicam(cam_json, device)
        return render_one(view, gaussians, pipe, bg_color, separate_sh,
                          use_exp, _render_default), False
    except Exception as e:
        if not is_oom_error(e):
            raise
        torch.cuda.empty_cache()
        view = build_minicam(cam_json, device, scale=0.5)
        img = render_one(view, gaussians, pipe, bg_color, separate_sh,
                         use_exp, _render_default)
        return resize_uint8(img, int(cam_json["width"]), int(cam_json["height"])), True


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

    # sys.path was set by the driver to the reference tree. Import now.
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import _render_default

    device = torch.device("cuda:0")
    bg_color = torch.zeros(3, dtype=torch.float32, device=device)
    cam_spec = cfg.get("cameras", "first")
    iteration = int(cfg.get("iteration", 30000))
    separate_sh = bool(cfg.get("separate_sh", True))
    use_exp = bool(cfg.get("use_train_test_exp", True))
    pipe = build_pipe()

    emit({"type": "meta", "side": "3dgs", "scene_count": len(scenes)})

    for scene_path in scenes:
        name = os.path.basename(scene_path.rstrip("/"))
        try:
            ply = find_ply(scene_path, iteration)
            with open(os.path.join(scene_path, "cameras.json")) as f:
                cams_json = json.loads(f.read())
        except Exception as e:
            emit({"type": "error", "scene": name, "error": str(e)})
            continue

        # Load point cloud. sh_degree=3 matches 3DGS's standard training;
        # active_sh_degree is auto-detected from the ply by load_ply.
        gaussians = GaussianModel(sh_degree=3)
        with contextlib.redirect_stdout(io.StringIO()):
            gaussians.load_ply(ply, use_train_test_exp=use_exp)
        gaussians.active_sh_degree = gaussians.max_sh_degree

        holded = apply_llffhold(cams_json, cfg.get("llffhold", 8))
        selected = select_cameras(holded, cam_spec)
        if use_exp:
            ensure_exposures_for_selected_cameras(gaussians, selected, device)
        for cam_json in selected:
            try:
                img, used_half_fallback = render_with_oom_fallback(
                    cam_json, device, gaussians, pipe, bg_color,
                    separate_sh, use_exp, _render_default)
            except Exception as e:
                emit({"type": "error", "scene": name,
                      "img_name": cam_json.get("img_name", ""), "error": str(e)})
                continue

            fname = f"{name}__{safe_dump_name(cam_json.get('img_name', '?'))}.npy"
            path = os.path.join(args.out_dir, fname)
            np.save(path, img)
            emit({"type": "image", "side": "3dgs",
                  "scene": name, "img_name": cam_json.get("img_name", ""),
                  "camera_id": cam_json.get("id", ""),
                  "width": img.shape[1], "height": img.shape[0],
                  "num_vertex": int(gaussians.get_xyz.shape[0]),
                  "path": path,
                  "oom_half_res_fallback": used_half_fallback})

        del gaussians
        torch.cuda.empty_cache()

    emit({"type": "done", "side": "3dgs"})


if __name__ == "__main__":
    main()
