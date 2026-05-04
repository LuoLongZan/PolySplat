#!/usr/bin/env python3
"""Aggregate correctness-vs-GT CSVs into a paper-ready dataset summary."""
import csv
import glob
import math
import os
import sys
from collections import defaultdict


RESULTS_DIR = os.environ.get("POLYSPLAT_QUALITY_DIR", "./bench_results/quality")
FAIL_DIR = os.environ.get("POLYSPLAT_QUALITY_FAIL_DIR",
                          os.path.join(RESULTS_DIR, "fail"))
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.csv")

# Dataset bucket names match the per-dataset rows of the paper's main results
# table. The bucket name "flashgs_data" denotes the canonical 13-scene 3DGS
# benchmark (Mip-NeRF 360 + Tanks & Temples train/truck + Deep Blending
# drjohnson/playroom), following the literature convention.
EXPECTED_DATASETS = [
    "flashgs_data",
    "tanksandtemples",
    "nerfstudio",
    "llff",
    "seathru-nerf",
    "zipnerf",
    "h3dgs",
    "mill19",
    "eyeful",
    "dl3dv",
    "urbanscene3d",
    "worldengine_navtest",
]
EXCLUDED_OVERALL = {"urbanscene3d", "worldengine_navtest"}
METRICS = [
    "psnr_polysplat_vs_gt",
    "psnr_3dgs_vs_gt",
    "delta_psnr",
    "ssim_polysplat_vs_gt",
    "ssim_3dgs_vs_gt",
    "delta_ssim",
]


def dataset_from_filename(path):
    name = os.path.basename(path)
    prefix = "correctness_vs_gt_"
    stem = name[:-4] if name.endswith(".csv") else name
    if not stem.startswith(prefix):
        return ""
    tail = stem[len(prefix):]
    parts = tail.rsplit("_", 2)
    if len(parts) == 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return parts[0]
    return tail


def as_float(value):
    if value in (None, ""):
        return None
    value = str(value)
    if value.lower() == "inf":
        return math.inf
    return float(value)


def mean(values):
    vals = [v for v in values if v is not None and not math.isinf(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(value, digits=6):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def verdict(delta_psnr, delta_ssim):
    if delta_psnr is None:
        return "n/a"
    if delta_psnr <= -1.0:
        return "FAIL"
    if delta_psnr < -0.30:
        return "WARN"
    return "OK"


def frame_verdict(row):
    return verdict(as_float(row.get("delta_psnr")), as_float(row.get("delta_ssim")))


def write_failed_rows(rows):
    os.makedirs(FAIL_DIR, exist_ok=True)
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    fields = [
        "dataset", "scene", "img_name",
        "psnr_polysplat_vs_gt", "psnr_3dgs_vs_gt", "delta_psnr",
        "ssim_polysplat_vs_gt", "ssim_3dgs_vs_gt", "delta_ssim",
        "gt_path", "_source_csv", "row_verdict",
    ]
    for path in glob.glob(os.path.join(FAIL_DIR, "failed_frames*.csv")):
        if os.path.exists(path):
            os.remove(path)

    if not rows:
        with open(os.path.join(FAIL_DIR, "failed_frames.csv"), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
        return

    def write(path, subset):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in subset:
                writer.writerow({k: row.get(k, "") for k in fields})

    write(os.path.join(FAIL_DIR, "failed_frames.csv"), rows)
    for dataset, subset in by_dataset.items():
        write(os.path.join(FAIL_DIR, f"failed_frames_{dataset}.csv"), subset)


def read_rows(paths):
    rows = []
    failed_rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["dataset"] = row.get("dataset") or dataset_from_filename(path)
                row["_source_csv"] = path
                row["row_verdict"] = frame_verdict(row)
                if row["row_verdict"] == "FAIL":
                    failed_rows.append(row)
                else:
                    rows.append(row)
    write_failed_rows(failed_rows)
    return rows, failed_rows


def aggregate(rows, failed_rows=None):
    failed_counts = defaultdict(int)
    for row in failed_rows or []:
        failed_counts[row["dataset"]] += 1

    by_target = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in EXPECTED_DATASETS:
            continue
        by_target[(dataset, row["scene"])].append(row)

    targets_by_dataset = defaultdict(list)
    for (dataset, scene), target_rows in by_target.items():
        rec = {"dataset": dataset, "scene": scene, "cams": len(target_rows)}
        for metric in METRICS:
            rec[metric] = mean(as_float(r.get(metric)) for r in target_rows)
        targets_by_dataset[dataset].append(rec)

    summary = []
    for dataset in EXPECTED_DATASETS:
        target_recs = targets_by_dataset.get(dataset, [])
        if not target_recs:
            note = "GT unavailable" if dataset == "urbanscene3d" else "missing CSV"
            summary.append({
                "dataset": dataset, "targets": 0, "cams": 0,
                "cams_per_target": "", "overall_included": "no",
                "verdict": "n/a", "note": note,
            })
            continue

        rec = {
            "dataset": dataset,
            "targets": len(target_recs),
            "cams": sum(t["cams"] for t in target_recs),
        }
        rec["cams_per_target"] = rec["cams"] / rec["targets"]
        for metric in METRICS:
            rec[metric] = mean(t[metric] for t in target_recs)
        rec["verdict"] = verdict(rec["delta_psnr"], rec["delta_ssim"])
        rec["overall_included"] = "no" if dataset in EXCLUDED_OVERALL else "yes"
        if dataset == "urbanscene3d":
            note = "1 target; excluded from overall mean"
        elif dataset == "worldengine_navtest":
            note = "sparse GT subset; excluded from overall mean"
        else:
            note = ""
        if failed_counts.get(dataset):
            suffix = f"{failed_counts[dataset]} failed frames excluded"
            note = f"{note}; {suffix}" if note else suffix
        rec["note"] = note
        summary.append(rec)

    included = [r for r in summary if r.get("overall_included") == "yes" and r.get("targets", 0)]
    overall = {
        "dataset": "overall_mean",
        "targets": sum(int(r["targets"]) for r in included),
        "cams": sum(int(r["cams"]) for r in included),
        "cams_per_target": "",
        "overall_included": "",
        "note": f"dataset-balanced mean over {len(included)} datasets",
    }
    for metric in METRICS:
        overall[metric] = mean(r.get(metric) for r in included)
    overall["verdict"] = verdict(overall.get("delta_psnr"), overall.get("delta_ssim"))
    summary.append(overall)
    return summary


def write_summary(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset", "targets", "cams", "cams_per_target",
        "psnr_polysplat_vs_gt", "psnr_3dgs_vs_gt", "delta_psnr",
        "ssim_polysplat_vs_gt", "ssim_3dgs_vs_gt", "delta_ssim",
        "verdict", "overall_included", "note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for metric in METRICS:
                out[metric] = fmt(out.get(metric))
            if isinstance(out.get("cams_per_target"), float):
                out["cams_per_target"] = fmt(out["cams_per_target"], digits=2)
            writer.writerow({k: out.get(k, "") for k in fields})


def main(argv):
    paths = argv[1:] or [os.path.join(RESULTS_DIR, f)
                         for f in os.listdir(RESULTS_DIR)
                         if f.startswith("correctness_vs_gt_") and f.endswith(".csv")]
    clean_rows, failed_rows = read_rows(paths)
    rows = aggregate(clean_rows, failed_rows)
    write_summary(rows, SUMMARY_PATH)
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {os.path.join(FAIL_DIR, 'failed_frames.csv')}")
    for row in rows:
        if row["dataset"] in ("overall_mean", "urbanscene3d", "worldengine_navtest"):
            print(row)
    if any(row.get("verdict") == "FAIL" for row in rows):
        sys.exit("FAIL verdict detected in quality summary")


if __name__ == "__main__":
    main(sys.argv)
