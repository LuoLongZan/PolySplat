"""Benchmark PolySplat on a trained 3D Gaussian Splatting scene.

Expected layout for ``<model_path>``::

    <model_path>/point_cloud.ply          (or point_cloud/iteration_*/point_cloud.ply)
    <model_path>/cameras.json

Two passes are run separately because instrumentation perturbs end-to-end
timing:

* **Pass 1 - E2E wall-clock.** Single fused C++ call with one CUDA event
  pair per frame and no per-phase events. Cleanest host+device timing.
* **Pass 2 - Per-kernel.** Non-fused path with
  ``measure_preprocess`` / ``measure_sort`` / ``measure_render``. Three
  event pairs per frame, reported separately.

Usage::

    CUDA_VISIBLE_DEVICES=0 python bench_polysplat.py /path/to/scene_dir
"""
import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time

# Resolve the repo root (two levels up) so `import example` works regardless
# of the caller's current working directory.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from example import Camera, Rasterizer, Scene


E2E_VARIANT = "preranges_smem_persistent_lite_e2e"
PERKERNEL_VARIANT = "preranges_smem_persistent_lite"


def find_ply(model_path):
    symlink = os.path.join(model_path, "point_cloud.ply")
    if os.path.exists(symlink):
        return symlink
    for it in (7000, 30000, 15000):
        p = os.path.join(model_path, "point_cloud", f"iteration_{it}", "point_cloud.ply")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no point_cloud.ply under {model_path}")


def load_scene(model_path, device):
    ply_path = find_ply(model_path)
    cam_path = os.path.join(model_path, "cameras.json")
    scene = Scene(device)
    with contextlib.redirect_stdout(io.StringIO()):
        scene.loadPly(ply_path)
    with open(cam_path) as f:
        cams_json = json.loads(f.read())
    cameras = [Camera(c) for c in cams_json]
    return scene, cameras, ply_path


def pick_cameras(cameras, which):
    if which == "first":
        return cameras[:1]
    if which == "all":
        return cameras
    if which.isdigit():
        n = int(which)
        return cameras[:n]
    raise ValueError(f"bad --cameras: {which}")


def pass_e2e(rast, scene, cameras, bg_color, warmup, runs):
    """Return (per_camera_mean_ms_list, overall_mean_ms)."""
    results = []
    for cam in cameras:
        for _ in range(warmup):
            rast.forward(scene, cam, bg_color, render_variant=E2E_VARIANT)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            rast.forward(scene, cam, bg_color, render_variant=E2E_VARIANT)
            t1.record()
            t1.synchronize()
            times.append(t0.elapsed_time(t1))
        results.append((cam, statistics.fmean(times), statistics.median(times)))
    return results


def pass_perkernel(rast, scene, cameras, bg_color, warmup, runs):
    """Return list of (camera, preprocess_mean, sort_mean, render_mean, num_rendered)."""
    results = []
    for cam in cameras:
        for _ in range(warmup):
            rast.forward(
                scene, cam, bg_color,
                render_variant=PERKERNEL_VARIANT,
                measure_preprocess=True, measure_sort=True, measure_render=True,
                return_stats=True,
            )
        torch.cuda.synchronize()
        pp, so, rd, nr = [], [], [], 0
        for _ in range(runs):
            _, stats = rast.forward(
                scene, cam, bg_color,
                render_variant=PERKERNEL_VARIANT,
                measure_preprocess=True, measure_sort=True, measure_render=True,
                return_stats=True,
            )
            pp.append(stats["preprocess_ms"])
            so.append(stats["sort_ms"])
            rd.append(stats["render_ms"])
            nr = stats["num_rendered"]
        results.append((
            cam,
            statistics.fmean(pp), statistics.fmean(so), statistics.fmean(rd),
            nr,
        ))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="path to a trained 3DGS scene dir")
    ap.add_argument("--cameras", default="first",
                    help="'first' | 'all' | integer N")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--skip-perkernel", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    bg_color = torch.zeros(3, dtype=torch.float32)

    scene, cameras, ply_path = load_scene(args.model_path, device)
    selected = pick_cameras(cameras, args.cameras)
    print(f"scene:       {args.model_path}")
    print(f"ply:         {ply_path}")
    print(f"gaussians:   {scene.num_vertex}")
    print(f"cameras:     {len(selected)} / {len(cameras)} total")
    print(f"warmup/runs: {args.warmup}/{args.runs}")
    print()

    rast = Rasterizer(scene, 2**27, 2**20)

    # Pass 1 — E2E wall-clock (no per-phase events).
    if not args.skip_e2e:
        print(f"[pass 1] E2E wall-clock  (variant = {E2E_VARIANT})")
        print(f"{'img_name':<20} {'WxH':>12} {'mean(ms)':>10} {'median(ms)':>12} {'fps':>8}")
        print("-" * 68)
        e2e_means = []
        t_start = time.time()
        e2e = pass_e2e(rast, scene, selected, bg_color, args.warmup, args.runs)
        t_wall = time.time() - t_start
        for cam, mean_ms, med_ms in e2e:
            fps = 1000.0 / mean_ms
            print(f"{cam.img_name:<20} {cam.width:>5}x{cam.height:<5} "
                  f"{mean_ms:>9.3f}  {med_ms:>11.3f}  {fps:>7.1f}")
            e2e_means.append(mean_ms)
        if len(e2e_means) > 1:
            print("-" * 68)
            avg = statistics.fmean(e2e_means)
            print(f"{'AVG':<20} {'':>12} {avg:>9.3f}  "
                  f"{'':>11}  {1000.0 / avg:>7.1f}")
        print(f"(pass 1 wall: {t_wall:.1f}s)")
        print()

    # Pass 2 — per-kernel (fresh warmup; events perturb E2E so we segregate).
    if not args.skip_perkernel:
        print(f"[pass 2] Per-kernel      (variant = {PERKERNEL_VARIANT})")
        print(f"{'img_name':<20} {'N_rendered':>11} "
              f"{'preproc(ms)':>12} {'sort(ms)':>10} {'render(ms)':>11} {'sum(ms)':>9}")
        print("-" * 80)
        t_start = time.time()
        pk = pass_perkernel(rast, scene, selected, bg_color, args.warmup, args.runs)
        t_wall = time.time() - t_start
        pp_all, so_all, rd_all = [], [], []
        for cam, pp, so, rd, nr in pk:
            print(f"{cam.img_name:<20} {nr:>11} "
                  f"{pp:>11.3f}  {so:>9.3f}  {rd:>10.3f}  {pp+so+rd:>8.3f}")
            pp_all.append(pp); so_all.append(so); rd_all.append(rd)
        if len(pp_all) > 1:
            print("-" * 80)
            print(f"{'AVG':<20} {'':>11} "
                  f"{statistics.fmean(pp_all):>11.3f}  "
                  f"{statistics.fmean(so_all):>9.3f}  "
                  f"{statistics.fmean(rd_all):>10.3f}  "
                  f"{statistics.fmean(pp_all)+statistics.fmean(so_all)+statistics.fmean(rd_all):>8.3f}")
        print(f"(pass 2 wall: {t_wall:.1f}s)")


if __name__ == "__main__":
    main()
