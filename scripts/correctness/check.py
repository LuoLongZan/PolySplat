"""Implementation-drift check: PolySplat vs reference 3DGS, pixel-to-pixel.

This is a regression test — it catches "did I write an implementation bug?"
by comparing the two rasterizers' rendered pixels directly. It does NOT
answer "is the rendered image quality equivalent?"; for that, use the
sibling tool ``check_vs_gt.py`` (each renderer vs. the original photograph).

Short version:

* paper-grade "no quality loss" claim   →  ``check_vs_gt.py``
* quick regression smoke test           →  ``check.py``  (this file)

Usage::

    python scripts/correctness/check.py --config <path>
    python scripts/correctness/check.py --config <path> --keep-images

What it does:

1. Launches a subprocess with ``PYTHONPATH`` pointing at the PolySplat
   repo. The subprocess renders each selected camera with PolySplat and
   dumps uint8 ``(H, W, 3)`` ``.npy`` images.

2. Launches a second subprocess with ``PYTHONPATH`` pointing at
   ``cfg["reference_3dgs_dir"]`` (the user's clone of the official
   3D Gaussian Splatting reference repo). That subprocess loads the same
   ``.ply`` into the reference ``GaussianModel``, builds viewpoint objects
   from ``cameras.json``, renders via ``_render_default``, and dumps
   matching ``.npy`` images.

3. Loads both sets of ``.npy``, computes PSNR / SSIM / max_diff /
   mean_diff / diff_pixels% BETWEEN the two rendered images (not vs. a
   ground-truth photo — use ``check_vs_gt.py`` for that), prints a table
   with a per-image verdict, and writes a CSV under ``cfg["output_dir"]``.

Verdict thresholds:

* ``OK``   PSNR ≥ 30 AND SSIM ≥ 0.95  — PolySplat matches reference well.
* ``WARN`` PSNR ≥ 25 AND SSIM ≥ 0.90  — mild drift, inspect before trusting.
* ``FAIL`` otherwise                  — likely a bug.

Passing this check is necessary but NOT sufficient for a "no quality loss"
paper claim. You can pass this check while silently degrading vs-GT quality,
and you can fail this check while still matching vs-GT quality (if the two
rasterizers drift in opposite directions that both happen to stay close to
GT). That's why ``check_vs_gt.py`` exists.
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

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[correctness-vs-3dgs] {msg}", flush=True)


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


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
    rc = p.wait()
    if rc != 0:
        sys.exit(f"{label} runner failed rc={rc}")
    return records


# --------------------------------------------------------------- metrics
def psnr(mse):
    if mse <= 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_uint8(a, b):
    if not _HAS_SSIM:
        return None
    return float(_ssim(a, b, data_range=255, channel_axis=-1))


def diff_metrics(a, b):
    a32 = a.astype(np.int32); b32 = b.astype(np.int32)
    d = np.abs(a32 - b32)
    mse = float((d.astype(np.float64) ** 2).mean())
    max_diff = int(d.max())
    mean_diff = float(d.mean())
    pix_diff = (d.sum(axis=-1) > 0).sum()
    return {
        "psnr_db": psnr(mse),
        "ssim": ssim_uint8(a, b),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "diff_pixels": int(pix_diff),
        "total_pixels": a.shape[0] * a.shape[1],
        "diff_pct": float(pix_diff) / (a.shape[0] * a.shape[1]) * 100.0,
    }


def verdict(m):
    s = m["ssim"]
    ok_ssim   = True if s is None else s >= 0.95
    warn_ssim = True if s is None else s >= 0.90
    if m["psnr_db"] >= 30 and ok_ssim:
        return "OK"
    if m["psnr_db"] >= 25 and warn_ssim:
        return "WARN"
    return "FAIL"


def fmt_psnr(v):
    if v == math.inf:
        return "  inf"
    return f"{v:6.2f}"


def fmt_ssim(v):
    if v is None:
        return "   n/a"
    return f"{v:6.4f}"


# --------------------------------------------------------------- pairing
def compare_pairs(polysplat_recs, ref_recs):
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
    for key in keys:
        a = a_by_key[key]; b = b_by_key[key]
        a_img = np.load(a["path"])
        b_img = np.load(b["path"])
        if a_img.shape != b_img.shape:
            log(f"  SHAPE MISMATCH: polysplat={a_img.shape} 3dgs={b_img.shape}")
            continue
        m = diff_metrics(a_img, b_img)
        rows.append({
            "scene": a["scene"], "img_name": a["img_name"],
            "width": a["width"], "height": a["height"],
            "num_vertex": a.get("num_vertex") or b.get("num_vertex"),
            **m, "verdict": verdict(m),
        })
    return rows


def print_table(rows):
    if not rows:
        print("(no rows to report)")
        return
    print()
    print("=" * 116)
    print("Correctness: PolySplat vs reference 3DGS implementation")
    print("=" * 116)
    hdr = (f"{'scene':<14} {'img_name':<20} {'WxH':>12}  "
           f"{'PSNR(dB)':>9}  {'SSIM':>7}  {'max':>4}  {'mean':>6}  {'diff%':>8}  {'verdict':>7}")
    print(hdr)
    print("-" * len(hdr))

    psnrs, ssims, maxes, means, pcts = [], [], [], [], []
    n_ok = n_warn = n_fail = 0
    for r in rows:
        print(f"{r['scene']:<14} {r['img_name']:<20} "
              f"{r['width']:>5}x{r['height']:<5}  "
              f"{fmt_psnr(r['psnr_db'])}  {fmt_ssim(r['ssim'])}  "
              f"{r['max_abs_diff']:>4}  {r['mean_abs_diff']:>6.3f}  "
              f"{r['diff_pct']:>7.4f}%  {r['verdict']:>7}")
        if r['psnr_db'] != math.inf:
            psnrs.append(r['psnr_db'])
        if r['ssim'] is not None:
            ssims.append(r['ssim'])
        maxes.append(r['max_abs_diff'])
        means.append(r['mean_abs_diff'])
        pcts.append(r['diff_pct'])
        if r['verdict'] == 'OK':     n_ok += 1
        elif r['verdict'] == 'WARN': n_warn += 1
        else:                        n_fail += 1

    if len(rows) > 1:
        print("-" * len(hdr))
        pa = sum(psnrs) / len(psnrs) if psnrs else math.inf
        sa = sum(ssims) / len(ssims) if ssims else None
        print(f"{'AVG':<14} {'':<20} {'':>12}  "
              f"{fmt_psnr(pa)}  {fmt_ssim(sa)}  "
              f"{max(maxes):>4}  {sum(means)/len(means):>6.3f}  "
              f"{sum(pcts)/len(pcts):>7.4f}%")
    print("-" * len(hdr))
    print(f"Summary: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL  (out of {len(rows)})")
    if n_fail:
        print("\n!!! FAIL detected — inspect the flagged rows for likely rendering bugs.")
    elif n_warn:
        print("\n!!! WARN detected — inspect flagged rows; may still be acceptable drift.")


def save_csv(out_dir, stamp, rows):
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"correctness_vs_3dgs_{stamp}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--keep-images", action="store_true",
                    help="keep dumped .npy images (default: cleaned up)")
    args = ap.parse_args()

    args.config = os.path.abspath(args.config)
    cfg = load_cfg(args.config)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    polysplat_dir = cfg["polysplat_dir"]
    ref_dir = cfg["reference_3dgs_dir"]
    if not os.path.isdir(ref_dir):
        sys.exit(f"reference_3dgs_dir not found: {ref_dir}")

    tmp_root = tempfile.mkdtemp(prefix=f"correct3dgs_{stamp}_", dir="/tmp")
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

        rows = compare_pairs(polysplat_recs, ref_recs)
        print_table(rows)
        save_csv(cfg.get("output_dir", "bench_results"), stamp, rows)
    finally:
        if args.keep_images:
            log(f"kept images in {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
