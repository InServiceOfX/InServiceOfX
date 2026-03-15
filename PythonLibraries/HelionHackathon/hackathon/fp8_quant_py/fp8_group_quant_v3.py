"""
FP8 Per-Group Quantization — Helion Submission v3
Team: luminous-kernels

All SHAPE_CONFIGS are real B200 autotuned results from autotune_fp8.py
run on helion-enormous-piranha, March 14 2026.
7 shapes from Helion cache (prior autotune run), 2 freshly autotuned.

Benchmark times from autotune_fp8.py:
  (256, 4096, 128) : 0.0297 ms   283.5 GB/s
  (256, 8192, 128) : 0.0299 ms   563.6 GB/s
  (4096,7168, 128) : 0.0842 ms  2801.0 GB/s
"""

from task import input_t, output_t
import torch
import helion
import helion.language as hl

SHAPE_CONFIGS: dict[tuple, helion.Config] = {

    # ── Test shapes ────────────────────────────────────────────────────────
    # (1, 256, 64): N=4, 0.0321 ms — cached from prior autotune run
    (1,    256,  64): helion.Config(
        block_sizes=[4],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        load_eviction_policies=['', '', ''],
        num_stages=1, num_warps=4, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[None],
    ),
    # (4, 512, 128): N=16, 0.0344 ms — cached
    (4,    512,  128): helion.Config(
        block_sizes=[16],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        load_eviction_policies=['', '', ''],
        num_stages=1, num_warps=4, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[None],
    ),
    # (16, 1024, 64): N=256, 0.0234 ms — cached
    (16,   1024, 64): helion.Config(
        block_sizes=[32],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        load_eviction_policies=['', '', ''],
        num_stages=1, num_warps=4, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[None],
    ),
    # (1, 4096, 128): N=32, 0.0282 ms — cached
    # Notable: persistent_interleaved scheduler + maxnreg=64 + num_sm_multiplier=4
    # The B200 has 160 SMs; with N=32 blocks this would leave most SMs idle with
    # 'flat' scheduling. persistent_interleaved keeps all SMs busy by cooperatively
    # dividing work. maxnreg=64 limits registers per thread to increase occupancy.
    (1,    4096, 128): helion.Config(
        block_sizes=[2],
        indexing=['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor',
                  'tensor_descriptor', 'pointer'],
        load_eviction_policies=['', '', ''],
        maxnreg=64, num_sm_multiplier=4,
        num_stages=1, num_warps=2, pid_type='persistent_interleaved',
        range_flattens=[True], range_multi_buffers=[True],
        range_unroll_factors=[0], range_warp_specializes=[None],
        reduction_loops=[None],
    ),
    # (8, 4096, 128): N=256, 0.0330 ms — cached
    (8,    4096, 128): helion.Config(
        block_sizes=[32],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        load_eviction_policies=['', '', ''],
        num_stages=1, num_warps=4, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[None],
    ),

    # ── Benchmark shapes ───────────────────────────────────────────────────
    # (16, 4096, 128): N=512, 0.0281 ms — cached
    # Notable: num_stages=7 (deep pipeline), num_warps=1, reduction_loops=64
    # Deep staging prefetches aggressively in the memory pipeline.
    (16,   4096, 128): helion.Config(
        block_sizes=[2],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer',
                  'tensor_descriptor', 'pointer', 'pointer'],
        load_eviction_policies=['', 'first', ''],
        num_stages=7, num_warps=1, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[64],
    ),
    # (256, 4096, 128): N=8192, 0.0297 ms — cached
    # Notable: num_warps=32 (maximum), tensor_descriptor on outputs [3,4,5]
    (256,  4096, 128): helion.Config(
        block_sizes=[32],
        indexing=['pointer', 'pointer', 'pointer', 'tensor_descriptor',
                  'tensor_descriptor', 'tensor_descriptor'],
        load_eviction_policies=['', '', 'first'],
        num_stages=1, num_warps=32, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[None],
    ),
    # (256, 8192, 128): N=16384, 0.0299 ms — freshly autotuned (15.8s, 47 configs)
    # Notable: tensor_descriptor on index[4] (qout write), load_eviction='last'
    (256,  8192, 128): helion.Config(
        block_sizes=[16],
        indexing=['pointer', 'pointer', 'pointer', 'pointer',
                  'tensor_descriptor', 'pointer'],
        load_eviction_policies=['', 'last', ''],
        num_stages=1, num_warps=4, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[None],
    ),
    # (4096, 7168, 128): N=229376, 0.0842 ms — freshly autotuned (51s, 30 configs)
    # Notable: tensor_descriptor on index[5] (scales write)
    (4096, 7168, 128): helion.Config(
        block_sizes=[8],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer',
                  'tensor_descriptor'],
        load_eviction_policies=['', '', ''],
        num_stages=1, num_warps=4, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[], range_unroll_factors=[0],
        range_warp_specializes=[None], reduction_loops=[None],
    ),
}

FALLBACK_CONFIG = helion.Config(
    block_sizes=[32],
    indexing=['pointer'] * 6,
    load_eviction_policies=['', '', ''],
    num_stages=1, num_warps=8, pid_type='flat',
    range_flattens=[None], range_multi_buffers=[None],
    range_num_stages=[], range_unroll_factors=[0],
    range_warp_specializes=[None], reduction_loops=[None],
)

# ── Kernel factory ────────────────────────────────────────────────────────────
_kernel_cache: dict = {}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def fp8_group_quant(
        data: torch.Tensor,        # [N, gsz]
        scales_out: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        nrows = data.size(0)
        ncols = hl.specialize(data.size(1))
        FP8_MAX = 448.0
        EPS     = 1e-10
        qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)
        for rr in hl.tile(nrows):
            row   = data[rr, :].to(torch.float32)
            amax  = torch.amax(torch.abs(row), -1)
            amax  = torch.clamp(amax, min=EPS)
            scale = amax / FP8_MAX
            qout[rr, :] = torch.clamp(row / scale[:, None], -FP8_MAX, FP8_MAX)
            scales_out[rr] = scale
        return qout
    return fp8_group_quant

def _get_kernel(config: helion.Config):
    # Use id(config) as cache key — each Config object is unique per shape
    key = id(config)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_kernel(config)
    return _kernel_cache[key]

# ── Entry point ───────────────────────────────────────────────────────────────
def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G    = x_s.shape[1]
    gsz  = H // G
    N    = T * G

    config = SHAPE_CONFIGS.get((T, H, gsz), FALLBACK_CONFIG)
    kernel = _get_kernel(config)

    flat_q = kernel(x.reshape(N, gsz), x_s.reshape(N))
    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = x_s.reshape(T, G)
    return x_q, x_s
