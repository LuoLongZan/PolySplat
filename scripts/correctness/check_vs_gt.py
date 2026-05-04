"""Paper-grade correctness check: compare PolySplat and reference 3DGS against
the ORIGINAL CAPTURED PHOTOS (ground truth), not against each other.

This is the right question for a paper claiming "no quality loss":

    PSNR(PolySplat_render,  GT_photo)   ≈   PSNR(3DGS_render, GT_photo)

If the two PSNRs are within ~0.1-0.3 dB, the PolySplat optimizations have
not degraded image quality relative to the reference rasterizer. Both
rasterizers are compared to the same GT, so any rasterizer-internal
differences (sort tie-breaks, FP order, etc.) that don't affect quality
vs. the captured image are correctly ignored.

Contrast with check.py, which compares PolySplat renders vs 3DGS renders
directly — that catches implementation drift but says nothing about
quality relative to what a reviewer considers "the truth".

Usage::

    python scripts/correctness/check_vs_gt.py --config scripts/correctness/config.smoke.json

Config additions vs check.py:
    "gt_dirs":  { "<scene_name>": "/path/to/images_or_images_2_or_4" }
        Maps the basename of each scene directory to where its GT photos
        live. Files are looked up as <gt_dir>/<img_name>.{jpg,png,JPG,PNG}.

Resolution handling:
    Rendered images come out at the cameras.json resolution, which is
    often 2x or 4x the GT resolution (because models are trained on
    downsampled images but cameras are exported at full intrinsics).
    Before metrics, we LANCZOS-downsample each rendered image to the
    GT resolution. Both PolySplat and 3DGS get the same treatment, so the
    relative comparison remains fair.
"""
import argparse
import csv
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[correctness-vs-gt] {msg}", flush=True)


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def dataset_name_from_config(config_path, cfg):
    if cfg.get("dataset"):
        return cfg["dataset"]
    return os.path.splitext(os.path.basename(config_path))[0]


# ------------------------------------------------------------- subprocess
def run_subprocess(label, runner_path, tree_dir, cfg, config_path, out_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("cuda_visible_devices", "0"))
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = tree_dir + (os.pathsep + prev if prev else "")

    cmd = [
        sys.executable, runner_path,
        "--config", config_path,
        "--scenes-json", json.dumps(cfg["scenes"]),
        "--out-dir", out_dir,
    ]
    log(f"run: {label} tree={tree_dir} -> {out_dir}")
    p = subprocess.Popen(cmd, cwd=tree_dir, env=env,
                         stdout=subprocess.PIPE, stderr=sys.stderr,
                         text=True, bufsize=1)
    records = []
    for line in p.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"[{label}] {line}\n")
            continue
        if ev.get("type") == "image":
            records.append(ev)
        elif ev.get("type") == "error":
            log(f"  [{label}] ERROR {ev.get('scene')}/{ev.get('img_name','')}: {ev.get('error')}")
        elif ev.get("type") == "warning":
            log(f"  [{label}] WARN {ev.get('scene')}/{ev.get('img_name','')}: {ev.get('warning')}")
    rc = p.wait()
    if rc != 0:
        sys.exit(f"{label} runner failed rc={rc}")
    return records


# ----------------------------------------------------------------- GT IO
_GT_EXTS = (".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG")


def find_gt_file(gt_dir, img_name, camera_id=None):
    # img_name may already include extension; also may not
    stem, ext = os.path.splitext(img_name)
    if ext:
        p = os.path.join(gt_dir, img_name)
        if os.path.exists(p):
            return p
        for cand_ext in _GT_EXTS:
            p = os.path.join(gt_dir, stem + cand_ext)
            if os.path.exists(p):
                return p
    for ext in _GT_EXTS:
        p = os.path.join(gt_dir, img_name + ext)
        if os.path.exists(p):
            return p
    if camera_id not in (None, ""):
        try:
            idx = int(camera_id)
        except (TypeError, ValueError):
            idx = None
        if idx is not None:
            for prefix in ("image", "img"):
                for ext in _GT_EXTS:
                    p = os.path.join(gt_dir, f"{prefix}{idx:03d}{ext}")
                    if os.path.exists(p):
                        return p
    return None


def load_gt_uint8(path):
    """GT as (H, W, 3) uint8, RGB."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def resize_to(img_uint8, target_w, target_h):
    """Downsample/upsample (H, W, 3) uint8 via LANCZOS to (target_h, target_w, 3)."""
    h, w = img_uint8.shape[:2]
    if w == target_w and h == target_h:
        return img_uint8
    pil = Image.fromarray(img_uint8, mode="RGB")
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


# --------------------------------------------------------------- metrics
def psnr(mse):
    if mse <= 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_uint8(a, b):
    if not _HAS_SSIM:
        return None
    return float(_ssim(a, b, data_range=255, channel_axis=-1))


def pair_metrics(render_u8, gt_u8):
    """render_u8 and gt_u8 must already be at the same resolution."""
    r32 = render_u8.astype(np.int32); g32 = gt_u8.astype(np.int32)
    d = np.abs(r32 - g32)
    mse = float((d.astype(np.float64) ** 2).mean())
    return {
        "psnr_db": psnr(mse),
        "ssim":    ssim_uint8(render_u8, gt_u8),
    }


# ------------------------------------------------------------- pair walk
def build_rows(dataset, polysplat_recs, ref_recs, gt_dirs):
    a_by_key = {(r["scene"], r["img_name"]): r for r in polysplat_recs}
    b_by_key = {(r["scene"], r["img_name"]): r for r in ref_recs}
    keys = sorted(set(a_by_key) & set(b_by_key))
    missing_polysplat = sorted(set(b_by_key) - set(a_by_key))
    missing_ref     = sorted(set(a_by_key) - set(b_by_key))
    for k in missing_polysplat:
        log(f"  MISSING on PolySplat side: {k}")
    for k in missing_ref:
        log(f"  MISSING on 3DGS side: {k}")

    rows = []
    for (scene, img_name) in keys:
        a = a_by_key[(scene, img_name)]
        b = b_by_key[(scene, img_name)]
        a_img = np.load(a["path"])      # (H, W, 3) uint8 — PolySplat render
        b_img = np.load(b["path"])      # (H, W, 3) uint8 — 3DGS render
        if a_img.shape != b_img.shape:
            log(f"  SHAPE MISMATCH: polysplat={a_img.shape} 3dgs={b_img.shape}")
            continue

        gt_dir = gt_dirs.get(scene)
        if gt_dir is None:
            log(f"  no gt_dir for scene '{scene}', skipping")
            continue
        gt_path = find_gt_file(gt_dir, img_name, a.get("camera_id") or b.get("camera_id"))
        if gt_path is None:
            log(f"  GT file not found for {scene}/{img_name} under {gt_dir}")
            continue

        gt_u8 = load_gt_uint8(gt_path)
        gh, gw = gt_u8.shape[:2]

        # Downsample rendered images to GT resolution (both renderers same treatment).
        a_resized = resize_to(a_img, gw, gh)
        b_resized = resize_to(b_img, gw, gh)

        m_poly = pair_metrics(a_resized, gt_u8)
        m_3dgs  = pair_metrics(b_resized, gt_u8)
        d_psnr  = m_poly["psnr_db"] - m_3dgs["psnr_db"]
        d_ssim  = None
        if m_poly["ssim"] is not None and m_3dgs["ssim"] is not None:
            d_ssim = m_poly["ssim"] - m_3dgs["ssim"]

        rows.append({
            "dataset": dataset,
            "scene": scene, "img_name": img_name,
            "camera_id": a.get("camera_id") or b.get("camera_id"),
            "gt_w": gw, "gt_h": gh,
            "render_w": a_img.shape[1], "render_h": a_img.shape[0],
            "num_vertex": a.get("num_vertex") or b.get("num_vertex"),
            "psnr_polysplat_vs_gt": m_poly["psnr_db"],
            "psnr_3dgs_vs_gt":  m_3dgs["psnr_db"],
            "delta_psnr":       d_psnr,
            "ssim_polysplat_vs_gt": m_poly["ssim"],
            "ssim_3dgs_vs_gt":  m_3dgs["ssim"],
            "delta_ssim":       d_ssim,
            "gt_path":          gt_path,
        })
    return rows


# --------------------------------------------------------------- output
def _fmt_psnr(v):
    if v is None:          return "   n/a "
    if math.isinf(v):      return "   inf "
    return f"{v:7.3f}"


def _fmt_ssim(v):
    if v is None:          return "  n/a "
    return f"{v:6.4f}"


def _fmt_delta(v, width=7, digits=3):
    if v is None:          return " n/a".rjust(width)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{digits}f}".rjust(width)


def verdict_for(delta_psnr, delta_ssim=None):
    """Asymmetric: PolySplat being BETTER than 3DGS is never a problem.
    Only regressions (PolySplat < 3DGS beyond noise) are flagged."""
    if delta_psnr is None:
        return "n/a"
    if delta_psnr <= -1.0:
        return "FAIL"  # significant regression
    if delta_psnr < -0.30:
        return "WARN"  # mild quality regression
    return "OK"       # equivalent or better


def print_table(rows):
    if not rows:
        print("(no rows to report)")
        return
    print()
    print("=" * 124)
    print("Correctness vs ground-truth photos:  PolySplat render → GT    AND    3DGS render → GT")
    print("=" * 124)
    hdr = (f"{'scene':<14} {'img_name':<12} {'GTwxh':>11}  "
           f"{'PSNR_poly':>10}  {'PSNR_3dgs':>9}  {'ΔPSNR':>7}  "
           f"{'SSIM_poly':>10}  {'SSIM_3dgs':>9}  {'ΔSSIM':>7}  {'verdict':>7}")
    print(hdr)
    print("-" * len(hdr))

    psnrs_f, psnrs_d, deltas_p, ssims_f, ssims_d, deltas_s = [], [], [], [], [], []
    n_ok = n_warn = n_fail = 0
    for r in rows:
        v = verdict_for(r["delta_psnr"], r["delta_ssim"])
        if v == "OK":      n_ok += 1
        elif v == "WARN":  n_warn += 1
        else:              n_fail += 1
        print(f"{r['scene']:<14} {r['img_name']:<12} "
              f"{r['gt_w']:>4}x{r['gt_h']:<4}  "
              f"{_fmt_psnr(r['psnr_polysplat_vs_gt'])}   {_fmt_psnr(r['psnr_3dgs_vs_gt'])}  "
              f"{_fmt_delta(r['delta_psnr']):>7}  "
              f"{_fmt_ssim(r['ssim_polysplat_vs_gt'])}   {_fmt_ssim(r['ssim_3dgs_vs_gt'])}  "
              f"{_fmt_delta(r['delta_ssim'], digits=4):>7}  {v:>7}")
        if r["psnr_polysplat_vs_gt"] is not None and not math.isinf(r["psnr_polysplat_vs_gt"]):
            psnrs_f.append(r["psnr_polysplat_vs_gt"])
        if r["psnr_3dgs_vs_gt"] is not None and not math.isinf(r["psnr_3dgs_vs_gt"]):
            psnrs_d.append(r["psnr_3dgs_vs_gt"])
        if r["delta_psnr"] is not None:
            deltas_p.append(r["delta_psnr"])
        if r["ssim_polysplat_vs_gt"] is not None:
            ssims_f.append(r["ssim_polysplat_vs_gt"])
        if r["ssim_3dgs_vs_gt"] is not None:
            ssims_d.append(r["ssim_3dgs_vs_gt"])
        if r["delta_ssim"] is not None:
            deltas_s.append(r["delta_ssim"])

    if len(rows) > 1:
        print("-" * len(hdr))
        pf = sum(psnrs_f) / len(psnrs_f) if psnrs_f else None
        pd = sum(psnrs_d) / len(psnrs_d) if psnrs_d else None
        dp = sum(deltas_p) / len(deltas_p) if deltas_p else None
        sf = sum(ssims_f) / len(ssims_f) if ssims_f else None
        sd = sum(ssims_d) / len(ssims_d) if ssims_d else None
        ds = sum(deltas_s) / len(deltas_s) if deltas_s else None
        print(f"{'AVG':<14} {'':<12} {'':>11}  "
              f"{_fmt_psnr(pf)}   {_fmt_psnr(pd)}  "
              f"{_fmt_delta(dp):>7}  "
              f"{_fmt_ssim(sf)}   {_fmt_ssim(sd)}  "
              f"{_fmt_delta(ds, digits=4):>7}")
    print("-" * len(hdr))
    print(f"Summary: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL  (out of {len(rows)})")
    print("  ΔPSNR = PSNR(PolySplat, GT) − PSNR(3DGS, GT);  positive = PolySplat better.")
    print("  OK    = ΔPSNR ≥ −0.30 dB")
    print("  WARN  = −1.00 dB < ΔPSNR < −0.30 dB")
    print("  FAIL  = ΔPSNR ≤ −1.00 dB  (significant quality regression — likely a bug)")
    if n_fail:
        print("\n!!! FAIL detected — PolySplat quality regresses against reference.")


def save_csv(out_dir, dataset, stamp, rows):
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"correctness_vs_gt_{dataset}_{stamp}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"wrote {path}")


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.smoke_gt.json"))
    ap.add_argument("--keep-images", action="store_true",
                    help="keep dumped .npy images (default: cleaned up)")
    args = ap.parse_args()

    args.config = os.path.abspath(args.config)
    cfg = load_cfg(args.config)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = dataset_name_from_config(args.config, cfg)

    if cfg.get("gt_unavailable"):
        log(f"{dataset}: GT unavailable; skipping render/metric pass")
        return

    polysplat_dir = cfg["polysplat_dir"]
    ref_dir = cfg["reference_3dgs_dir"]
    if not os.path.isdir(ref_dir):
        sys.exit(f"reference_3dgs_dir not found: {ref_dir}")

    gt_dirs = cfg.get("gt_dirs", {})
    if not gt_dirs:
        sys.exit("config missing 'gt_dirs' — this check requires ground-truth photos")
    for name, d in gt_dirs.items():
        if not os.path.isdir(d):
            sys.exit(f"gt_dir for '{name}' not found: {d}")

    tmp_root = tempfile.mkdtemp(prefix=f"correctgt_{stamp}_", dir="/tmp")
    polysplat_out = os.path.join(tmp_root, "polysplat")
    ref_out = os.path.join(tmp_root, "3dgs")
    log(f"dump dir: {tmp_root}")

    try:
        polysplat_runner = os.path.join(HERE, "_runner_polysplat.py")
        ref_runner     = os.path.join(HERE, "_runner_3dgs.py")

        polysplat_recs = run_subprocess(
            "polysplat", polysplat_runner, polysplat_dir, cfg, args.config, polysplat_out)
        ref_recs = run_subprocess(
            "3dgs", ref_runner, ref_dir, cfg, args.config, ref_out)

        rows = build_rows(dataset, polysplat_recs, ref_recs, gt_dirs)
        print_table(rows)
        save_csv(cfg.get("output_dir", "bench_results"), dataset, stamp, rows)
    finally:
        if args.keep_images:
            log(f"kept images in {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
