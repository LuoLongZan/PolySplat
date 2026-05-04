import torch
import polysplat

import os
import sys
import json
import time

SMEM_TOPK_GAUSSIANS = 1365


class Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = polysplat.ops.loadPly(
            scene_path)
        print("num_vertex = %d" % self.num_vertex)
        # 58*4byte
        self.position = self.position.to(self.device)  # 3
        self.shs = self.shs.to(self.device)  # 48
        self.shs_half = self.shs.half()  # FP16 SH for reduced memory bandwidth
        self.opacity = self.opacity.to(self.device)  # 1
        self.cov3d = self.cov3d.to(self.device)  # 6


class Camera:
    def __init__(self, camera_json):
        self.id = camera_json['id']
        self.img_name = camera_json['img_name']
        self.width = camera_json['width']
        self.height = camera_json['height']
        self.position = torch.tensor(camera_json['position'])
        self.rotation = torch.tensor(camera_json['rotation'])
        self.focal_x = camera_json['fx']
        self.focal_y = camera_json['fy']
        self.zFar = 100.0
        self.zNear = 0.01


# 静态分配内存光栅化器
class Rasterizer:
    AUTO_RENDER_CANDIDATES = (
        (16, 16, "default"),
        (32, 16, "split"),
    )
    ADAPTIVE_TRUCK_TILE_THRESHOLD = 16384

    # 构造函数中分配内存
    def __init__(self, scene, MAX_NUM_RENDERED, MAX_NUM_TILES, enable_es=True):
        # 24 bytes
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.SORT_BUFFER_SIZE = polysplat.ops.get_sort_buffer_size(MAX_NUM_RENDERED)
        self.list_sorting_space = torch.zeros(self.SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8)
        self.curr_offset = torch.zeros(1, device=scene.device, dtype=torch.int32)

        # --- Early Sorting (ES) adaptive state + buffers ---
        # When enable_es=True, allocate the metadata/scratch buffers used by the ES
        # 2-pass preprocess path. These are only touched when `forward(..., use_es=True)`
        # (or auto mode decides to use ES); when ES is not used, the baseline path
        # is bit-identical to before this integration.
        self.enable_es = enable_es
        self._es_last_M = None              # previous frame's num_rendered (for adaptive dispatch)
        # Thresholds are env-overridable so benching / tuning is painless.
        self._es_threshold_M = int(os.environ.get("POLYSPLAT_ES_THRESHOLD_M", "5000000"))
        self._es_threshold_N_firstcall = int(os.environ.get("POLYSPLAT_ES_THRESHOLD_N", "3000000"))
        # Two-pass overhead scales with P (Pass-A + depth sort + scan + Pass-B all
        # touch every Gaussian). Tile-key sort downgrade saves bandwidth on M.
        # So two-pass only wins when M is large in *both* absolute and ratio-to-P
        # terms. ratio_thresh = 0.7 was hand-set; cold-start P guard prevents
        # firstcall mis-fire on city-scale scenes (P > 30M with no M history yet).
        self._es_ratio_thresh = float(os.environ.get("POLYSPLAT_ES_RATIO_THRESH", "0.7"))
        self._es_firstcall_huge_P_guard = int(os.environ.get("POLYSPLAT_ES_FIRSTCALL_HUGE_P", "30000000"))
        # B3' ablation: force two-pass regardless of cost-model decision.
        self._es_force_two_pass = os.environ.get("POLYSPLAT_ES_FORCE_TWO_PASS", "0") not in ("0", "", "false", "False")
        if enable_es:
            P = scene.num_vertex
            dev = scene.device
            # Pass-A outputs (per-Gaussian metadata)
            self._es_tiles_per_gauss = torch.zeros(P, device=dev, dtype=torch.int32)
            self._es_conic_power_raw = torch.zeros((P, 4), device=dev, dtype=torch.float32)
            self._es_depth_natural = torch.zeros(P, device=dev, dtype=torch.float32)
            self._es_rect_bounds = torch.zeros((P, 2), device=dev, dtype=torch.int32)
            # Depth-sort + scan scratch
            self._es_identity = torch.zeros(P, device=dev, dtype=torch.int32)
            self._es_perm = torch.zeros(P, device=dev, dtype=torch.int32)
            self._es_depth_sorted = torch.zeros(P, device=dev, dtype=torch.float32)
            self._es_tiles_per_gauss_sorted = torch.zeros(P, device=dev, dtype=torch.int32)
            self._es_cum_offsets_sorted = torch.zeros(P, device=dev, dtype=torch.int32)
            self._es_total_num_rendered = torch.zeros(1, device=dev, dtype=torch.int32)
            # Pass-B output + tile-sort scratch (M-sized — share with key/value buffers)
            self._es_tile_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=dev, dtype=torch.int32)
            self._es_tile_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=dev, dtype=torch.int32)
            # CUB DeviceScan workspace (sort workspace is shared with baseline's list_sorting_space)
            scan_bytes = polysplat.ops.get_es_scan_buffer_size(P)
            self._es_scan_scratch = torch.zeros(scan_bytes, device=dev, dtype=torch.int8)

        # 40 bytes
        self.points_xy = torch.zeros((scene.num_vertex, 2), device=scene.device, dtype=torch.float32)
        self.rgb_depth = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)

        # Compact (gathered) buffers for tile-sorted feature access
        self.compact_xy = torch.zeros((MAX_NUM_RENDERED, 2), device=scene.device, dtype=torch.float32)
        self.compact_rgb_depth = torch.zeros((MAX_NUM_RENDERED, 4), device=scene.device, dtype=torch.float32)
        self.compact_conic_opacity = torch.zeros((MAX_NUM_RENDERED, 4), device=scene.device, dtype=torch.float32)
        self.identity_values = torch.arange(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.smem_topk_lookup = torch.full((scene.num_vertex,), -1, device=scene.device, dtype=torch.int16)
        self.smem_topk_xy = torch.zeros((SMEM_TOPK_GAUSSIANS, 2), device=scene.device, dtype=torch.float32)
        self.smem_topk_rgb_depth = torch.zeros((SMEM_TOPK_GAUSSIANS, 4), device=scene.device, dtype=torch.float32)
        self.smem_topk_conic = torch.zeros((SMEM_TOPK_GAUSSIANS, 4), device=scene.device, dtype=torch.float32)

        # Pre-allocated buffer for precomputed tile ranges (persistent_v3)
        self.tile_ranges_buf = torch.zeros((MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32)

        # Tile reordering buffers (Direction A)
        self.tile_order = torch.zeros(MAX_NUM_TILES, device=scene.device, dtype=torch.int32)
        self.tile_counts_buf = torch.zeros(MAX_NUM_TILES, device=scene.device, dtype=torch.int32)
        self.tile_ids_buf = torch.zeros(MAX_NUM_TILES, device=scene.device, dtype=torch.int32)
        sort_temp_size = polysplat.ops.get_tile_order_sort_temp_size(MAX_NUM_TILES)
        self.tile_order_sort_temp = torch.zeros(sort_temp_size, device=scene.device, dtype=torch.int8)

        # Packed int4 tile descriptor {col, row, range.x, range.y} for reordered_v2
        self.tile_desc_buf = torch.zeros((MAX_NUM_TILES, 4), device=scene.device, dtype=torch.int32)

        self._out_color = None  # lazily allocated per resolution
        self._auto_render_choice = None

    def _get_out_color(self, height, width, device):
        if self._out_color is None or self._out_color.shape[0] != height or self._out_color.shape[1] != width:
            self._out_color = torch.zeros((height, width, 3), device=device, dtype=torch.uint8)
        return self._out_color

    @staticmethod
    def _get_render_fn(block_x, block_y, render_variant):
        if block_x == 16 and block_y == 16:
            if render_variant == "split":
                return polysplat.ops.render_16x16_split
            if render_variant in ("unroll2", "gathered_unroll2"):
                return polysplat.ops.render_16x16_unroll2
            if render_variant in ("gathered",):
                return polysplat.ops.render_16x16
            return polysplat.ops.render_16x16
        if block_x == 24 and block_y == 16:
            if render_variant == "split":
                return polysplat.ops.render_24x16_split
            raise ValueError(f"Unsupported render kernel shape: {block_x}x{block_y} ({render_variant})")
        if block_x == 32 and block_y == 16:
            if render_variant == "split":
                return polysplat.ops.render_32x16_split
            return polysplat.ops.render_32x16
        if block_x == 32 and block_y == 32:
            return polysplat.ops.render_32x32
        raise ValueError(f"Unsupported render kernel shape: {block_x}x{block_y} ({render_variant})")

    def _prepare_topk_smem_cache(self, scene, num_rendered):
        num_topk = min(SMEM_TOPK_GAUSSIANS, scene.num_vertex)
        if num_topk <= 0:
            return 0

        rendered_ids = self.gaussian_values_sorted[:num_rendered].to(torch.int64)
        counts = torch.bincount(rendered_ids, minlength=scene.num_vertex)
        topk_ids = torch.topk(counts, k=num_topk, largest=True, sorted=False).indices

        # Build lookup: gaussian_id -> slot index (or -1)
        self.smem_topk_lookup.fill_(-1)
        self.smem_topk_lookup[topk_ids] = torch.arange(num_topk, device=scene.device, dtype=torch.int16)

        # Embed slot info into point_list values in-place:
        #   bit 31      = is_topk flag (1 = topk, 0 = normal gaussian_id)
        #   bits  0..30 = smem slot index (when flag=1)
        # When flag=0, the value is the plain gaussian_id unchanged.
        vals = self.gaussian_values_sorted[:num_rendered]
        slots = self.smem_topk_lookup[vals.to(torch.int64)].to(torch.int32)  # -1 or 0..num_topk-1
        hit_mask = slots >= 0
        # Pack: flag_bit | slot_index  (gaussian_id not needed — data comes from smem)
        packed = vals.clone()
        packed[hit_mask] = (1 << 31) | slots[hit_mask]
        vals.copy_(packed)

        self.smem_topk_xy[:num_topk].copy_(torch.index_select(self.points_xy, 0, topk_ids))
        selected_rgb_depth = torch.index_select(self.rgb_depth, 0, topk_ids)
        self.smem_topk_rgb_depth[:num_topk].copy_(selected_rgb_depth)
        self.smem_topk_conic[:num_topk].copy_(torch.index_select(self.conic_opacity, 0, topk_ids))
        return num_topk

    def _autotune_render_config(self, scene, camera, bg_color, measure_runs=2):
        best_choice = None
        best_total_ms = None
        for block_x, block_y, render_variant in self.AUTO_RENDER_CANDIDATES:
            self.forward(
                scene,
                camera,
                bg_color,
                block_x=block_x,
                block_y=block_y,
                render_variant=render_variant,
            )
            total_ms = 0.0
            for _ in range(measure_runs):
                _, stats = self.forward(
                    scene,
                    camera,
                    bg_color,
                    block_x=block_x,
                    block_y=block_y,
                    render_variant=render_variant,
                    measure_preprocess=True,
                    measure_sort=True,
                    measure_render=True,
                    return_stats=True,
                )
                total_ms += stats["preprocess_ms"] + stats["sort_ms"] + stats["render_ms"]
            avg_total_ms = total_ms / measure_runs
            if best_total_ms is None or avg_total_ms < best_total_ms:
                best_total_ms = avg_total_ms
                best_choice = (block_x, block_y, render_variant)
        self._auto_render_choice = best_choice
        return best_choice

    @classmethod
    def _resolve_reordered_persistent_adaptive(cls, width, height):
        x_blocks = (width + 15) // 16
        y_blocks = (height + 15) // 16
        total_tiles = x_blocks * y_blocks
        # Step 14-17 showed a clean split:
        # small tile grids (truck-like) favor Morton+zigzag; large grids
        # (bicycle-like) favor pure zigzag load balancing.
        if total_tiles <= cls.ADAPTIVE_TRUCK_TILE_THRESHOLD:
            return "reordered_persistent_morton1280_zigzag792"
        return "reordered_persistent_zigzag264"

    # 前向传播（应用层封装）
    def _decide_use_es(self, scene, use_es, render_variant):
        """Decide whether to use the ES (Early Sorting / two-pass) pipeline.

        ES is only wired into the **non-fused** rasterization path. Fused / E2E
        variants always fall back to baseline.

        Adaptive ('auto') mode: two-pass wins only when M is *both* absolutely
        large *and* large relative to P, because the two-pass overhead scales
        with P (Pass-A + depth sort + scan + Pass-B) while the benefit scales
        with M (32-bit stable tile sort vs 64-bit composite). So we route to
        two-pass iff `last_M > T_M` *and* `last_M > rho * P`. On scenes where
        P is large but M/P is low (e.g. h3dgs city-scale), the single-pass
        baseline wins, and adaptive routing must select it.

        First call (no last_M history): use scene.num_vertex N as a proxy, but
        bail to single-pass when N exceeds the cold-start huge-P guard — a
        first-call two-pass on a 56M-Gaussian scene wastes ~3-4 ms on
        Pass-A+depth-sort before any M-side benefit can be realized.
        """
        if not self.enable_es:
            return False
        # Fused / E2E variants don't have an ES counterpart yet.
        is_fused_variant = (
            render_variant == "preranges_smem_persistent_lite_e2e"
            or render_variant == "preranges_smem_persistent_lite_e2e_preranges"
        )
        if is_fused_variant:
            return False
        if use_es is True:
            return True
        if use_es is False:
            return False
        # use_es == "auto" — explicit force-on takes priority over cost model
        # (used by §6.3 B3' ablation to measure "always two-pass" baseline).
        if self._es_force_two_pass:
            return True
        P = scene.num_vertex
        if self._es_last_M is not None:
            M = self._es_last_M
            return (M > self._es_threshold_M) and (M > self._es_ratio_thresh * P)
        # First-call fallback: M unknown, lean on N. Cold-start huge-P guard
        # prevents campus-scale firstcall from paying full Pass-A overhead with
        # no chance to recoup until the second frame.
        if P > self._es_firstcall_huge_P_guard:
            return False
        return P > self._es_threshold_N_firstcall

    def forward(self, scene, camera, bg_color,
                block_x=16, block_y=16,
                render_variant="default",
                measure_preprocess=False, measure_sort=False, measure_render=False,
                return_stats=False,
                use_es="auto"):
        requested_render_variant = render_variant
        if render_variant == "auto":
            if self._auto_render_choice is None:
                self._autotune_render_config(scene, camera, bg_color)
            block_x, block_y, render_variant = self._auto_render_choice
        elif render_variant == "reordered_persistent_adaptive":
            render_variant = self._resolve_reordered_persistent_adaptive(camera.width, camera.height)

        use_topk_smem = render_variant in ("topk_smem", "topk_smem_persistent")
        use_persistent_v2 = render_variant == "topk_smem_persistent_v2"
        use_reordered = render_variant == "reordered"
        use_reordered_v2 = render_variant == "reordered_v2" or render_variant.startswith("reordered_zigzag")
        use_reordered_persistent = (
            render_variant == "reordered_persistent"
            or render_variant.startswith("reordered_persistent_morton")
            or render_variant.startswith("reordered_persistent_zigzag")
        )
        use_preranges_smem_persistent = render_variant.startswith("preranges_smem_persistent")
        use_preranges = render_variant in ("preranges", "preranges_scan", "preranges_gathered", "preranges_smem", "preranges_smem_v2", "preranges_naive")
        use_preranges_scan = render_variant == "preranges_scan"
        use_preranges_gathered = render_variant == "preranges_gathered"
        use_preranges_smem = render_variant in ("preranges_smem", "preranges_smem_v2")
        use_preranges_smem_v2 = render_variant == "preranges_smem_v2"
        use_preranges_naive = render_variant == "preranges_naive"
        use_reordered_zigzag = render_variant.startswith("reordered_zigzag") or render_variant.startswith("reordered_persistent_zigzag")
        use_reordered_morton = render_variant.startswith("reordered_persistent_morton")
        # preranges_smem_persistent uses Morton+zigzag tile ordering
        psp_morton_bucket_size = 0
        psp_zigzag_group_size = 0
        if use_preranges_smem_persistent:
            # Parse optional suffix: preranges_smem_persistent_morton1280_zigzag792
            suffix = render_variant[len("preranges_smem_persistent"):]
            if suffix.startswith("_morton"):
                suffix2 = suffix[len("_morton"):]
                if "_zigzag" in suffix2:
                    parts = suffix2.split("_zigzag")
                    psp_morton_bucket_size = int(parts[0]) if parts[0] else 1280
                    psp_zigzag_group_size = int(parts[1]) if parts[1] else 792
                else:
                    psp_morton_bucket_size = int(suffix2) if suffix2 else 1280
            elif suffix.startswith("_zigzag"):
                psp_zigzag_group_size = int(suffix[len("_zigzag"):] or "264")
            elif suffix == "_adaptive":
                # Same adaptive logic as reordered_persistent_adaptive
                x_blocks = (camera.width + block_x - 1) // block_x
                y_blocks = (camera.height + block_y - 1) // block_y
                total_tiles = x_blocks * y_blocks
                if total_tiles <= self.ADAPTIVE_TRUCK_TILE_THRESHOLD:
                    psp_morton_bucket_size = 1280
                    psp_zigzag_group_size = 792
                else:
                    psp_zigzag_group_size = 264
            elif suffix == "":
                # Default: adaptive
                x_blocks = (camera.width + block_x - 1) // block_x
                y_blocks = (camera.height + block_y - 1) // block_y
                total_tiles = x_blocks * y_blocks
                if total_tiles <= self.ADAPTIVE_TRUCK_TILE_THRESHOLD:
                    psp_morton_bucket_size = 1280
                    psp_zigzag_group_size = 792
                else:
                    psp_zigzag_group_size = 264

        use_e2e_fused = render_variant == "preranges_smem_persistent_lite_e2e"
        use_e2e_preranges = render_variant == "preranges_smem_persistent_lite_e2e_preranges"
        if (use_e2e_fused or use_e2e_preranges) and (measure_preprocess or measure_sort or measure_render):
            # E2E paths cannot expose per-phase events (everything is in one C++ call)
            use_e2e_fused = False
            use_e2e_preranges = False
            render_variant = "preranges_smem_persistent_lite"
        use_packed_order = use_reordered_v2 or use_reordered_persistent or use_reordered_zigzag or use_reordered_morton
        zigzag_group_size = 0
        morton_bucket_size = 0
        morton_zigzag_group_size = 0
        if use_reordered_morton:
            suffix = render_variant[len("reordered_persistent_morton"):]
            if "_zigzag" in suffix:
                parts = suffix.split("_zigzag")
                morton_bucket_size = int(parts[0]) if parts[0] else 1280
                morton_zigzag_group_size = int(parts[1]) if parts[1] else 792
            else:
                morton_bucket_size = int(suffix) if suffix else 1280
                morton_zigzag_group_size = 0
        elif use_reordered_zigzag:
            if render_variant.startswith("reordered_persistent_zigzag"):
                suffix = render_variant[len("reordered_persistent_zigzag"):]
                zigzag_group_size = int(suffix) if suffix else 264
            else:
                suffix = render_variant[len("reordered_zigzag"):]
                zigzag_group_size = int(suffix) if suffix else 132
        if use_topk_smem and (block_x != 16 or block_y != 16):
            raise ValueError("topk_smem currently only supports 16x16 blocks")
        use_gather = render_variant in ("gathered", "gathered_unroll2") or use_preranges_gathered
        use_fused = not (measure_preprocess or measure_sort or measure_render) and not use_gather and not use_topk_smem and not use_persistent_v2 and not use_reordered and not use_reordered_v2 and not use_reordered_persistent and not use_reordered_zigzag and not use_reordered_morton and not use_preranges and not use_preranges_smem_persistent

        if use_e2e_preranges:
            # E2E forward with separate precompute_tile_ranges kernel (no Python dispatch)
            out_color = self._get_out_color(camera.height, camera.width, scene.device)
            num_rendered = polysplat.ops.forward_fused_e2e_preranges(
                scene.position, scene.shs_half, scene.opacity, scene.cov3d,
                camera.width, camera.height, block_x, block_y,
                camera.position, camera.rotation,
                camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                self.curr_offset,
                self.list_sorting_space,
                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                self.tile_ranges_buf,
                bg_color, out_color)
            if return_stats:
                stats = {
                    "requested_render_variant": requested_render_variant,
                    "render_variant": "preranges_smem_persistent_lite_e2e_preranges",
                    "resolved_block_x": block_x,
                    "resolved_block_y": block_y,
                    "num_rendered": num_rendered,
                    "preprocess_ms": None,
                    "sort_ms": None,
                    "render_ms": None,
                }
                return out_color, stats
            return out_color

        if use_e2e_fused:
            # E2E fully-fused forward: preprocess_half_sh + sort + render_persistent_lite_fused
            # in one C++ call. Uses inline binary search (no precompute_tile_ranges kernel).
            out_color = self._get_out_color(camera.height, camera.width, scene.device)
            num_rendered = polysplat.ops.forward_fused_e2e(
                scene.position, scene.shs_half, scene.opacity, scene.cov3d,
                camera.width, camera.height, block_x, block_y,
                camera.position, camera.rotation,
                camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                self.curr_offset,
                self.list_sorting_space,
                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                bg_color, out_color)
            if return_stats:
                stats = {
                    "requested_render_variant": requested_render_variant,
                    "render_variant": "preranges_smem_persistent_lite_e2e",
                    "resolved_block_x": block_x,
                    "resolved_block_y": block_y,
                    "num_rendered": num_rendered,
                    "preprocess_ms": None,
                    "sort_ms": None,
                    "render_ms": None,
                }
                return out_color, stats
            return out_color

        if use_fused:
            # Fully fused forward: preprocess+sort+render in single C++ call
            out_color = self._get_out_color(camera.height, camera.width, scene.device)
            num_rendered = polysplat.ops.forward_fused(
                scene.position, scene.shs, scene.opacity, scene.cov3d,
                camera.width, camera.height, block_x, block_y,
                camera.position, camera.rotation,
                camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                render_variant,
                self.curr_offset,
                self.list_sorting_space,
                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                bg_color, out_color)
            if return_stats:
                stats = {
                    "requested_render_variant": requested_render_variant,
                    "render_variant": render_variant,
                    "resolved_block_x": block_x,
                    "resolved_block_y": block_y,
                    "num_rendered": num_rendered,
                    "preprocess_ms": None,
                    "sort_ms": None,
                    "render_ms": None,
                }
                return out_color, stats
            return out_color

        # Fallback: separate kernels for per-kernel timing or gather mode
        # 属性预处理 + 键值绑定
        self.curr_offset.fill_(0)
        # Decide whether to take the ES (Early Sorting) pipeline for this frame.
        # Only applies to the non-fused path; baseline otherwise.
        should_use_es = self._decide_use_es(scene, use_es, render_variant)

        preprocess_start = None
        preprocess_end = None
        if measure_preprocess:
            preprocess_start = torch.cuda.Event(enable_timing=True)
            preprocess_end = torch.cuda.Event(enable_timing=True)
            preprocess_start.record()
        if should_use_es:
            # ES Pass-A: same shape/culling as baseline preprocess but writes
            # per-Gaussian metadata (count / unscaled conic / depth / rect) for
            # Pass-B and skips the 64-bit key emission.
            if use_preranges_smem or use_preranges_smem_persistent:
                polysplat.ops.preprocess_es_pass_a_half_sh(
                    scene.position, scene.shs_half, scene.opacity, scene.cov3d,
                    camera.width, camera.height, block_x, block_y,
                    camera.position, camera.rotation,
                    camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self._es_tiles_per_gauss, self._es_conic_power_raw,
                    self._es_depth_natural, self._es_rect_bounds)
            else:
                polysplat.ops.preprocess_es_pass_a(
                    scene.position, scene.shs, scene.opacity, scene.cov3d,
                    camera.width, camera.height, block_x, block_y,
                    camera.position, camera.rotation,
                    camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self._es_tiles_per_gauss, self._es_conic_power_raw,
                    self._es_depth_natural, self._es_rect_bounds)
        else:
            if use_preranges_smem or use_preranges_smem_persistent:
                polysplat.ops.preprocess_half_sh(scene.position, scene.shs_half, scene.opacity, scene.cov3d,
                                                    camera.width, camera.height, block_x, block_y,
                                                    camera.position, camera.rotation,
                                                    camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                                                    self.points_xy, self.rgb_depth, self.conic_opacity,
                                                    self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                    self.curr_offset)
            else:
                polysplat.ops.preprocess(scene.position, scene.shs, scene.opacity, scene.cov3d,
                                                    camera.width, camera.height, block_x, block_y,
                                                    camera.position, camera.rotation,
                                                    camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                                                    self.points_xy, self.rgb_depth, self.conic_opacity,
                                                    self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                    self.curr_offset)
        if measure_preprocess:
            preprocess_end.record()

        preprocess_ms = None
        if not should_use_es:
            num_rendered = int(self.curr_offset.cpu()[0])
            if num_rendered >= self.MAX_NUM_RENDERED:
                raise "Too many k-v pairs!"
        # For ES, num_rendered is resolved mid-"sort" block (after depth_sort_and_scan)
        # because the total is a device-side scan result.

        sort_start = None
        sort_end = None
        if measure_sort:
            sort_start = torch.cuda.Event(enable_timing=True)
            sort_end = torch.cuda.Event(enable_timing=True)
            sort_start.record()
        if should_use_es:
            # ES sort pipeline: depth sort + gather + scan → Pass-B emit (32-bit tile keys)
            # → 32-bit stable tile sort → rebuild 64-bit sorted keys.
            polysplat.ops.es_depth_sort_and_scan(
                scene.num_vertex,
                self._es_tiles_per_gauss, self._es_depth_natural,
                self._es_identity, self._es_perm, self._es_depth_sorted,
                self._es_tiles_per_gauss_sorted, self._es_cum_offsets_sorted,
                self._es_total_num_rendered,
                self.list_sorting_space, self._es_scan_scratch)
            polysplat.ops.preprocess_es_pass_b(
                scene.num_vertex,
                camera.width, camera.height, block_x, block_y,
                self._es_perm, self._es_cum_offsets_sorted, self._es_tiles_per_gauss_sorted,
                self.points_xy, self._es_conic_power_raw, self._es_rect_bounds,
                self._es_tile_keys_unsorted, self.gaussian_values_unsorted)
            # Host-sync on total num_rendered (CUB SortPairs needs host int).
            num_rendered = int(self._es_total_num_rendered.cpu()[0])
            if num_rendered >= self.MAX_NUM_RENDERED:
                raise "Too many k-v pairs!"
            polysplat.ops.es_tile_sort(
                num_rendered,
                camera.width, camera.height, block_x, block_y,
                self._es_tile_keys_unsorted, self.gaussian_values_unsorted,
                self._es_tile_keys_sorted, self.gaussian_values_sorted,
                self.gaussian_keys_sorted,
                self.list_sorting_space)
        else:
            polysplat.ops.sort_gaussian(num_rendered, camera.width, camera.height, block_x, block_y,
                                                       self.list_sorting_space,
                                                       self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                       self.gaussian_keys_sorted, self.gaussian_values_sorted)
        if measure_sort:
            sort_end.record()
        # Gather step for tile-sorted feature access
        num_topk = 0
        if use_topk_smem:
            num_topk = self._prepare_topk_smem_cache(scene, num_rendered)
        if use_gather:
            polysplat.ops.gather_features(
                num_rendered,
                self.gaussian_values_sorted,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                self.compact_xy, self.compact_rgb_depth, self.compact_conic_opacity)
            render_xy = self.compact_xy
            render_rgb = self.compact_rgb_depth
            render_conic = self.compact_conic_opacity
            render_values = self.identity_values
        else:
            render_xy = self.points_xy
            render_rgb = self.rgb_depth
            render_conic = self.conic_opacity
            render_values = self.gaussian_values_sorted

        # 排序 + 像素着色 + 混色阶段
        out_color = self._get_out_color(camera.height, camera.width, scene.device)

        # Precompute tile ranges for persistent variants (before render timing)
        # preranges_smem_v2 does its own binary search, so skip precompute for it
        if use_preranges_smem_persistent or use_reordered or use_packed_order or (use_preranges and not use_preranges_scan and not use_preranges_smem_v2):
            polysplat.ops.precompute_tile_ranges(
                num_rendered, camera.width, camera.height, block_x, block_y,
                self.gaussian_keys_sorted, self.tile_ranges_buf)
        if use_preranges_scan:
            polysplat.ops.precompute_tile_ranges_scan(
                num_rendered, camera.width, camera.height, block_x, block_y,
                self.gaussian_keys_sorted, self.tile_ranges_buf)
        # Compute tile reordering for reordered variant
        if use_reordered:
            x_blocks = (camera.width + block_x - 1) // block_x
            y_blocks = (camera.height + block_y - 1) // block_y
            total_tiles = x_blocks * y_blocks
            polysplat.ops.compute_tile_order(
                self.tile_ranges_buf, total_tiles,
                self.tile_order, self.tile_counts_buf, self.tile_ids_buf,
                self.tile_order_sort_temp)
        if use_packed_order:
            x_blocks = (camera.width + block_x - 1) // block_x
            y_blocks = (camera.height + block_y - 1) // block_y
            total_tiles = x_blocks * y_blocks
            if use_reordered_morton:
                polysplat.ops.compute_tile_order_packed_morton(
                    self.tile_ranges_buf, total_tiles, x_blocks,
                    self.tile_desc_buf, self.tile_order,
                    self.tile_counts_buf, self.tile_ids_buf,
                    self.tile_order_sort_temp,
                    morton_zigzag_group_size,
                    morton_bucket_size)
            else:
                polysplat.ops.compute_tile_order_packed(
                    self.tile_ranges_buf, total_tiles, x_blocks,
                    self.tile_desc_buf, self.tile_order,
                    self.tile_counts_buf, self.tile_ids_buf,
                    self.tile_order_sort_temp,
                    zigzag_group_size)
        if use_preranges_smem_persistent and render_variant not in ("preranges_smem_persistent_lite", "preranges_smem_persistent_lite_dt"):
            x_blocks = (camera.width + block_x - 1) // block_x
            y_blocks = (camera.height + block_y - 1) // block_y
            total_tiles = x_blocks * y_blocks
            if psp_morton_bucket_size > 0:
                polysplat.ops.compute_tile_order_packed_morton(
                    self.tile_ranges_buf, total_tiles, x_blocks,
                    self.tile_desc_buf, self.tile_order,
                    self.tile_counts_buf, self.tile_ids_buf,
                    self.tile_order_sort_temp,
                    psp_zigzag_group_size,
                    psp_morton_bucket_size)
            elif psp_zigzag_group_size > 0:
                polysplat.ops.compute_tile_order_packed(
                    self.tile_ranges_buf, total_tiles, x_blocks,
                    self.tile_desc_buf, self.tile_order,
                    self.tile_counts_buf, self.tile_ids_buf,
                    self.tile_order_sort_temp,
                    psp_zigzag_group_size)
            else:
                polysplat.ops.compute_tile_order_packed(
                    self.tile_ranges_buf, total_tiles, x_blocks,
                    self.tile_desc_buf, self.tile_order,
                    self.tile_counts_buf, self.tile_ids_buf,
                    self.tile_order_sort_temp,
                    0)

        render_start = None
        render_end = None
        if measure_render:
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            render_start.record()
        if use_topk_smem:
            if render_variant == "topk_smem_persistent":
                topk_render_fn = polysplat.ops.render_16x16_topk_smem_persistent
            else:
                topk_render_fn = polysplat.ops.render_16x16_topk_smem
            topk_render_fn(
                num_rendered, camera.width, camera.height,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.smem_topk_xy, self.smem_topk_rgb_depth, self.smem_topk_conic,
                num_topk,
                bg_color, out_color)
        elif use_persistent_v2:
            polysplat.ops.render_16x16_topk_smem_persistent_v2(
                num_rendered, camera.width, camera.height,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.smem_topk_xy, self.smem_topk_rgb_depth, self.smem_topk_conic,
                0,
                bg_color, out_color)
        elif use_reordered:
            polysplat.ops.render_16x16_reordered(
                num_rendered, camera.width, camera.height,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.tile_ranges_buf, self.tile_order,
                bg_color, out_color)
        elif use_reordered_v2:
            polysplat.ops.render_16x16_reordered_v2(
                num_rendered, camera.width, camera.height,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.tile_desc_buf,
                bg_color, out_color)
        elif use_reordered_persistent:
            polysplat.ops.render_16x16_reordered_persistent(
                num_rendered, camera.width, camera.height,
                self.points_xy, self.rgb_depth, self.conic_opacity,
                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                self.tile_desc_buf,
                bg_color, out_color)
        elif use_preranges_smem_persistent:
            if render_variant == "preranges_smem_persistent_lite":
                polysplat.ops.render_16x16_preranges_smem_persistent_lite(
                    num_rendered, camera.width, camera.height,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self.gaussian_values_sorted,
                    self.tile_ranges_buf,
                    bg_color, out_color)
            elif render_variant == "preranges_smem_persistent_lite_dt":
                polysplat.ops.render_16x16_preranges_smem_persistent_lite_dt(
                    num_rendered, camera.width, camera.height,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self.gaussian_values_sorted,
                    self.tile_ranges_buf,
                    bg_color, out_color)
            else:
                polysplat.ops.render_16x16_preranges_smem_persistent(
                    num_rendered, camera.width, camera.height,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self.gaussian_values_sorted,
                    self.tile_desc_buf,
                    bg_color, out_color)
        elif use_preranges:
            if use_preranges_smem_v2:
                polysplat.ops.render_16x16_preranges_smem_v2(
                    num_rendered, camera.width, camera.height,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self.gaussian_keys_sorted, self.gaussian_values_sorted,
                    bg_color, out_color)
            elif use_preranges_smem:
                polysplat.ops.render_16x16_preranges_smem(
                    num_rendered, camera.width, camera.height,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self.gaussian_values_sorted,
                    self.tile_ranges_buf,
                    bg_color, out_color)
            elif use_preranges_naive:
                polysplat.ops.render_16x16_preranges_naive(
                    num_rendered, camera.width, camera.height,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self.gaussian_values_sorted,
                    self.tile_ranges_buf,
                    bg_color, out_color)
            elif use_preranges_gathered:
                polysplat.ops.render_16x16_preranges(
                    num_rendered, camera.width, camera.height,
                    self.compact_xy, self.compact_rgb_depth, self.compact_conic_opacity,
                    self.identity_values,
                    self.tile_ranges_buf,
                    bg_color, out_color)
            else:
                polysplat.ops.render_16x16_preranges(
                    num_rendered, camera.width, camera.height,
                    self.points_xy, self.rgb_depth, self.conic_opacity,
                    self.gaussian_values_sorted,
                    self.tile_ranges_buf,
                    bg_color, out_color)
        else:
            render_fn = self._get_render_fn(block_x, block_y, render_variant)
            render_fn(num_rendered, camera.width, camera.height,
                      render_xy, render_rgb, render_conic,
                      self.gaussian_keys_sorted, render_values,
                      bg_color, out_color)
        render_ms = None
        if measure_render:
            render_end.record()
            render_end.synchronize()
        elif measure_preprocess or measure_sort:
            torch.cuda.synchronize()
        if measure_preprocess:
            preprocess_ms = preprocess_start.elapsed_time(preprocess_end)
        sort_ms = None
        if measure_sort:
            sort_ms = sort_start.elapsed_time(sort_end)
        if measure_render:
            render_ms = render_start.elapsed_time(render_end)
        # Update adaptive ES state for next frame's dispatch.
        self._es_last_M = num_rendered
        if return_stats:
            stats = {
                "requested_render_variant": requested_render_variant,
                "render_variant": render_variant,
                "resolved_block_x": block_x,
                "resolved_block_y": block_y,
                "num_rendered": num_rendered,
                "preprocess_ms": preprocess_ms,
                "sort_ms": sort_ms,
                "render_ms": render_ms,
                "used_es": should_use_es,
            }
            return out_color, stats
        if measure_preprocess and measure_sort and measure_render:
            return out_color, preprocess_ms, sort_ms, render_ms
        if measure_preprocess and measure_render:
            return out_color, preprocess_ms, render_ms
        if measure_preprocess and measure_sort:
            return out_color, preprocess_ms, sort_ms
        if measure_sort and measure_render:
            return out_color, sort_ms, render_ms
        if measure_preprocess:
            return out_color, preprocess_ms
        if measure_sort:
            return out_color, sort_ms
        if measure_render:
            return out_color, render_ms
        return out_color


def savePpm(image, path):
    image = image.cpu()
    assert image.dim() >= 3
    assert image.size(2) == 3
    with open(path, 'wb') as f:
        f.write(b'P6\n' + f'{image.size(1)} {image.size(0)}\n255\n'.encode() + image.numpy().tobytes())


def render_scene(model_path, test_performance=False):
    scene_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    print(scene_path)
    camera_path = os.path.join(model_path, "cameras.json")
    print(camera_path)
    device = torch.device('cuda:0')
    bg_color = torch.zeros(3, dtype=torch.float32)  # black

    scene = Scene(device)
    scene.loadPly(scene_path)

    with open(camera_path, 'r') as camera_file:
        cameras_json = json.loads(camera_file.read())

    image_dir = os.path.join(model_path, "test_out")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    MAX_NUM_RENDERED = 2 ** 27
    MAX_NUM_TILES = 2 ** 20
    rasterizer = Rasterizer(scene, MAX_NUM_RENDERED, MAX_NUM_TILES)
    for camera_json in cameras_json:
        camera = Camera(camera_json)
        print("image name = %s" % camera.img_name)

        image = rasterizer.forward(scene, camera, bg_color)  # warm up

        if test_performance:
            n = 10
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                image = rasterizer.forward(scene, camera, bg_color)  # test performance
            torch.cuda.synchronize()
            t1 = time.time()
            print("elapsed time = %f ms" % ((t1 - t0) / n * 1000))
            print("fps = %f" % (n / (t1 - t0)))

        image_path = os.path.join(image_dir, "%s.ppm" % camera.img_name)
        savePpm(image, image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python example.py <model_path>", file=sys.stderr)
        print("       <model_path> must contain point_cloud/iteration_30000/point_cloud.ply", file=sys.stderr)
        print("       and a cameras.json describing the views to render.", file=sys.stderr)
        sys.exit(1)
    render_scene(sys.argv[1], test_performance=True)
