# PolySplat: Workload-Regime-Aware Rasterization for 3D Gaussian Splatting

> **Anonymized code release accompanying the NeurIPS&nbsp;2026 submission
> *PolySplat: Workload-Regime-Aware Rasterization for 3D Gaussian
> Splatting*. Author identities and any organization-affiliating
> artifacts have been removed for double-blind review.**

PolySplat is a CUDA forward-pass rasterizer for 3D Gaussian Splatting (3DGS).
It is designed to deliver robust acceleration across the entire workload
space — extreme Gaussian counts (up to 56.5&nbsp;M), high resolutions (up
to 9K), and diverse scene categories — rather than over-fitting to the
canonical 13-scene benchmark. On a stratified 76-target benchmark the
system attains dataset-balanced geometric-mean speedups of
**1.26×&nbsp;/&nbsp;2.27×&nbsp;/&nbsp;4.58×&nbsp;/&nbsp;6.48×** over
FlashGS, gsplat, Flash3DGS, and the reference INRIA 3DGS rasterizer at
lossless visual quality. Under sustained 60&nbsp;Hz interaction at
9216&nbsp;×&nbsp;6912 on a city-scale scene, PolySplat strictly meets
every 1-vsync deadline.

## Core ideas

Three CUDA-level mechanisms are introduced (see `csrc/cuda_rasterizer/`):

1. **Adaptive tile-key emission with three-way dispatch
   (`preprocess.cu`).** A warp-lane-saturating dispatcher routes
   single-tile, fragmented multi-tile, and massive multi-tile Gaussians
   into three specialized execution paths based on a hardware-grounded
   `max_t ≥ 32` predicate.
2. **Asynchronous shared-memory staging (`render.cu`).** Pixel
   throughput is decoupled from gather latency by overlapping
   per-Gaussian feature loads with shading execution, with a
   double-buffered batch size of 32 and one-warp thread blocks.
3. **Centralized persistent scheduling (`render.cu`).** A persistent
   kernel with an atomic global dispatch queue mathematically bounds
   terminal-phase tail latency to the cost of the single heaviest tile.

## Hardware and software requirements

* NVIDIA GPU with compute capability ≥ 8.0 (developed and tuned on
  H100; A100 is a supported fallback).
* CUDA Toolkit ≥ 12.1 with `nvcc` available; `CUDA_HOME` must be
  exported.
* Python ≥ 3.10 with PyTorch ≥ 2.1 (`torch` must be importable; CUDA
  build is detected via `torch.version.cuda`).
* Standard build chain (`g++ ≥ 9`, `setuptools`, `pybind11`).

The release ships its own copy of [GLM](https://github.com/g-truc/glm)
under `csrc/glm/`. CUB / CCCL is consumed from the system CUDA Toolkit.

## Build and install

```bash
# from the repository root
pip install -r requirements.txt
python setup.py build_ext --inplace      # in-place build of polysplat.*.so
# or
pip install .                            # install as the `polysplat` package
```

The build produces a single CPython extension named `polysplat` exposing
the `polysplat.ops` namespace.

## Quick start

A trained 3DGS model directory is expected to follow the standard
INRIA layout:

```
<scene_dir>/
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply
└── cameras.json
```

Render every camera in `cameras.json` and dump PPM frames into
`<scene_dir>/test_out/`:

```bash
python example.py /path/to/scene_dir
```

Programmatic usage:

```python
import json
import torch
from example import Scene, Camera, Rasterizer

device = torch.device("cuda:0")
bg_color = torch.zeros(3, dtype=torch.float32)

scene = Scene(device)
scene.loadPly("scene_dir/point_cloud/iteration_30000/point_cloud.ply")

with open("scene_dir/cameras.json") as f:
    cams = [Camera(c) for c in json.load(f)]

rast = Rasterizer(scene, MAX_NUM_RENDERED=2**27, MAX_NUM_TILES=2**20)

# Recommended render variant for end-to-end performance:
img = rast.forward(scene, cams[0], bg_color,
                   render_variant="preranges_smem_persistent_lite_e2e")
```

The `render_variant` argument selects among the kernel variants exposed
by the C++ extension. Two variants are intended for paper-grade timing:

| `render_variant`                              | Purpose                                                                                        |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `preranges_smem_persistent_lite_e2e`          | Fully fused single-call forward pass. Used for **end-to-end wall-clock** timing.               |
| `preranges_smem_persistent_lite`              | Per-stage variant; pair with `measure_preprocess`/`measure_sort`/`measure_render` for a stage breakdown. |

## Repository layout

```
.
├── csrc/                               # CUDA / C++ source
│   ├── ops.h                           # Public C++ API declarations
│   ├── pybind.cpp                      # Python bindings
│   ├── glm/                            # Vendored GLM math library (third-party)
│   └── cuda_rasterizer/
│       ├── preprocess.cu               # Contribution 1 — three-way tile-key dispatch
│       ├── render.cu                   # Contributions 2, 3 — async staging + persistent kernel
│       ├── sort.cu                     # CUB radix-sort wrapper for tile keys
│       └── gather.cu                   # Tile-sorted feature gather kernel
├── example.py                          # End-to-end usage example & high-level Rasterizer wrapper
├── setup.py                            # CUDA extension build configuration
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT
└── scripts/
    ├── benchmark/
    │   ├── bench_polysplat.py          # Single-scene wall-clock + per-kernel breakdown
    │   └── batch_bench.py              # Multi-scene driver, writes CSV + Markdown summary
    └── correctness/
        ├── check.py                    # Pixel-vs-pixel implementation drift vs reference 3DGS
        ├── check_vs_gt.py              # Paper-grade quality check vs ground-truth photographs
        ├── _runner_polysplat.py        # PolySplat-side image dumper (driven by the checkers)
        ├── _runner_3dgs.py             # Reference-3DGS-side image dumper
        ├── aggregate_quality.py        # Aggregates per-dataset quality CSVs into a summary
        └── config.smoke.json           # Editable config template
```

## Reproducing the paper

> **GPU contention matters.** All wall-clock measurements should be
> taken on an idle GPU; sharing the device adds 1–2&nbsp;ms of random
> jitter that silently corrupts comparisons. Verify with
> `nvidia-smi --query-gpu=utilization.gpu --format=csv` before each
> run.

### Single-scene wall-clock & per-kernel breakdown

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/bench_polysplat.py \
    /path/to/scene_dir --cameras 20 --warmup 10 --runs 50
```

This runs two passes per camera (E2E, then per-kernel) and prints
per-camera and aggregated mean / median / FPS.

### Sweeping a directory of scenes (Table 2, PolySplat side)

`scripts/benchmark/batch_bench.py` enumerates a directory tree
of trained 3DGS models grouped by dataset (e.g.
`<root>/tanksandtemples/<scene>/...`,
`<root>/nerfstudio/<scene>/...`) and writes a CSV of E2E +
per-kernel timings:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/batch_bench.py \
    --root /path/to/3dgs_models \
    --warmup 10 --runs 50 \
    --out-csv bench_results/all_scenes.csv \
    --out-md  bench_results/all_scenes.md
```

The recognized dataset subdirectories are listed in `DATASETS` at the
top of `batch_bench.py` and match the per-dataset rows of the paper's
main results table.

### Cross-rasterizer comparison (Table 2, full)

To reproduce the cross-rasterizer columns of the paper's main results
table, evaluate each comparator (FlashGS, gsplat, Flash3DGS, the
reference INRIA 3DGS rasterizer) on the same set of scenes under the
same protocol used by `bench_polysplat.py` (same PLY, same camera list,
same per-camera warmup/runs counts, same idle-GPU constraint). Each
comparator must be installed independently from its own upstream
repository.

## Notes on numerical reproducibility

* PolySplat preserves the canonical sort-then-blend rendering equation
  of 3DGS, so per-pixel outputs match the reference INRIA 3DGS
  rasterizer up to floating-point reordering effects in alpha
  compositing and SH evaluation. Empirically, this is well below
  one-pixel disagreement on the canonical 13-scene benchmark and
  contributes a Δ&nbsp;PSNR vs ground truth of ≤ 0.005&nbsp;dB at the
  dataset-balanced level.
* All wall-clock numbers reported in the paper were measured on an
  NVIDIA H100 PCIe&nbsp;80&nbsp;GB GPU with `--use_fast_math` and the
  build flags in `setup.py`. Different GPUs and CUDA versions will
  shift absolute timings.

## License

MIT — see [`LICENSE`](LICENSE).
