"""Batch runner: sweeps every scene under ``--root``, measures E2E +
per-kernel timings in separate passes, and writes a CSV + markdown summary.

Two passes per scene (warmed fresh each pass so the event instrumentation
of pass 2 does not perturb pass 1):

* **Pass 1**: ``forward_fused_e2e``, one event pair per frame.
* **Pass 2**: non-fused path with ``measure_preprocess/sort/render=True``.

The script expects the standard 3D Gaussian Splatting trained-model layout::

    <root>/<dataset>/<scene>/point_cloud.ply       (or point_cloud/iteration_*)
    <root>/<dataset>/<scene>/cameras.json

Usage::

    CUDA_VISIBLE_DEVICES=0 python batch_bench.py \
        --root /path/to/3dgs_models \
        --out-csv bench_results/all_scenes.csv \
        --out-md  bench_results/all_scenes.md
"""
import argparse
import contextlib
import gc
import io
import json
import os
import statistics
import sys
import time
import traceback

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

DATASETS = [
    "blender", "llff", "seathru-nerf", "nerfstudio", "tanksandtemples",
    "zipnerf", "mill19", "worldengine_navtest",
    "h3dgs", "eyeful",
]


def find_ply(model_path):
    symlink = os.path.join(model_path, "point_cloud.ply")
    if os.path.exists(symlink):
        return symlink
    for it in (7000, 30000, 15000):
        p = os.path.join(model_path, "point_cloud", f"iteration_{it}", "point_cloud.ply")
        if os.path.exists(p):
            return p
    return None


def enumerate_scenes(root):
    scenes = []
    for ds in DATASETS:
        ds_dir = os.path.join(root, ds)
        if not os.path.isdir(ds_dir):
            continue
        for name in sorted(os.listdir(ds_dir)):
            p = os.path.join(ds_dir, name)
            if not os.path.isdir(p):
                continue
            if find_ply(p) and os.path.exists(os.path.join(p, "cameras.json")):
                scenes.append((ds, name, p))
    return scenes


def measure_one(model_path, warmup, runs, device, bg_color):
    ply_path = find_ply(model_path)
    cam_path = os.path.join(model_path, "cameras.json")

    scene = Scene(device)
    with contextlib.redirect_stdout(io.StringIO()):
        scene.loadPly(ply_path)
    with open(cam_path) as f:
        cams_json = json.loads(f.read())
    cam = Camera(cams_json[0])

    rast = Rasterizer(scene, 2**27, 2**20)

    # Pass 1 — E2E wall-clock
    for _ in range(warmup):
        rast.forward(scene, cam, bg_color, render_variant=E2E_VARIANT)
    torch.cuda.synchronize()
    e2e = []
    for _ in range(runs):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        rast.forward(scene, cam, bg_color, render_variant=E2E_VARIANT)
        t1.record()
        t1.synchronize()
        e2e.append(t0.elapsed_time(t1))

    # Pass 2 — per-kernel
    for _ in range(warmup):
        rast.forward(
            scene, cam, bg_color,
            render_variant=PERKERNEL_VARIANT,
            measure_preprocess=True, measure_sort=True, measure_render=True,
            return_stats=True,
        )
    torch.cuda.synchronize()
    pp, so, rd = [], [], []
    num_rendered = 0
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
        num_rendered = stats["num_rendered"]

    result = {
        "width": cam.width,
        "height": cam.height,
        "img_name": cam.img_name,
        "num_vertex": scene.num_vertex,
        "num_rendered": num_rendered,
        "e2e_mean_ms": statistics.fmean(e2e),
        "e2e_median_ms": statistics.median(e2e),
        "preproc_mean_ms": statistics.fmean(pp),
        "sort_mean_ms": statistics.fmean(so),
        "render_mean_ms": statistics.fmean(rd),
        "perkernel_sum_ms": statistics.fmean(pp) + statistics.fmean(so) + statistics.fmean(rd),
        "ply_path": ply_path,
    }

    del rast, scene
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fmt_row(ds, scene, r):
    return (
        f"| {ds} | {scene} | {r['num_vertex']:,} | {r['width']}x{r['height']} | "
        f"{r['num_rendered']:,} | {r['e2e_mean_ms']:.3f} | {1000.0/r['e2e_mean_ms']:.0f} | "
        f"{r['preproc_mean_ms']:.3f} | {r['sort_mean_ms']:.3f} | {r['render_mean_ms']:.3f} | "
        f"{r['perkernel_sum_ms']:.3f} | "
        f"{(r['perkernel_sum_ms']-r['e2e_mean_ms'])*1000:.0f} |"
    )


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "dataset,scene,num_vertex,width,height,num_rendered,"
            "e2e_mean_ms,e2e_median_ms,preproc_mean_ms,sort_mean_ms,render_mean_ms,"
            "perkernel_sum_ms,fps\n"
        )
        for ds, scene, r in rows:
            fps = 1000.0 / r["e2e_mean_ms"]
            f.write(
                f"{ds},{scene},{r['num_vertex']},{r['width']},{r['height']},{r['num_rendered']},"
                f"{r['e2e_mean_ms']:.4f},{r['e2e_median_ms']:.4f},"
                f"{r['preproc_mean_ms']:.4f},{r['sort_mean_ms']:.4f},{r['render_mean_ms']:.4f},"
                f"{r['perkernel_sum_ms']:.4f},{fps:.2f}\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="root directory containing per-dataset subdirs of trained 3DGS scenes")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out-csv", default="bench_results/trained_3dgs.csv")
    ap.add_argument("--out-md", default=None, help="optional markdown summary path")
    ap.add_argument("--skip-datasets", default="", help="comma-separated datasets to skip")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    bg_color = torch.zeros(3, dtype=torch.float32)

    skip = set(x.strip() for x in args.skip_datasets.split(",") if x.strip())
    all_scenes = [s for s in enumerate_scenes(args.root) if s[0] not in skip]
    print(f"[batch] {len(all_scenes)} scenes, warmup={args.warmup}, runs={args.runs}")
    print()

    rows = []
    failed = []
    t_all = time.time()
    for i, (ds, scene, path) in enumerate(all_scenes, 1):
        t0 = time.time()
        try:
            r = measure_one(path, args.warmup, args.runs, device, bg_color)
            dt = time.time() - t0
            rows.append((ds, scene, r))
            print(
                f"[{i:>2}/{len(all_scenes)}] {ds}/{scene:<22} "
                f"N={r['num_vertex']:>9,}  {r['width']}x{r['height']:<5}  "
                f"e2e={r['e2e_mean_ms']:>6.3f}ms  "
                f"pp={r['preproc_mean_ms']:>5.3f} so={r['sort_mean_ms']:>5.3f} rd={r['render_mean_ms']:>5.3f}  "
                f"({dt:.1f}s)"
            )
        except Exception as e:
            dt = time.time() - t0
            failed.append((ds, scene, str(e)))
            print(f"[{i:>2}/{len(all_scenes)}] {ds}/{scene}  FAILED  ({dt:.1f}s)  {e}")
            traceback.print_exc()

    t_all = time.time() - t_all
    print()
    print(f"[batch] done in {t_all:.1f}s, {len(rows)} ok, {len(failed)} failed")

    write_csv(args.out_csv, rows)
    print(f"[batch] wrote {args.out_csv}")

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write("# trained_3dgs PolySplat benchmark\n\n")
            f.write("| dataset | scene | gaussians | WxH | num_rendered | "
                    "e2e(ms) | fps | preproc(ms) | sort(ms) | render(ms) | "
                    "pk_sum(ms) | overhead(us) |\n")
            f.write("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for ds, scene, r in rows:
                f.write(fmt_row(ds, scene, r) + "\n")
        print(f"[batch] wrote {args.out_md}")

    if failed:
        print("[batch] failures:")
        for ds, scene, err in failed:
            print(f"  {ds}/{scene}: {err}")


if __name__ == "__main__":
    main()
