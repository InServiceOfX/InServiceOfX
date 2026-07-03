# CuLLM vs. JAX/XLA/cuDNN: Attention Forward Benchmark Report

Date: 2026-07-03. Hardware: NVIDIA GeForce RTX 3060 (Ampere, sm_86, 12 GB,
~12.7 FP32 TFLOP/s, ~51 FP16 tensor-core TFLOP/s dense). Software:
`propulsion-with-cuda:26.02-py3` (CUDA 13.1 container in minor-version
compatibility on a 13.0 driver), JAX 0.10.2 (XLA), cuDNN via
`jax.nn.dot_product_attention(implementation="cudnn")`.
Regenerate with `CUDALibraries/CuLLM/Python/benchmark_report.py` (header
comment has the exact commands).

## 1. What exactly is being compared

The single **attention forward core** — `O = softmax(QK^T/√d)·V` — at
multi-head scale: **batch·heads = 96 independent slices (B=8, H=12), head
dim d = 64**, GPT-2-medium-like. It is *not* full multi-head attention (no
QKV or output projections), not the backward pass, and not GQA. Contenders:

| Label | What it is | Precision |
|---|---|---|
| CuLLM warp-coop | Our hand-written CUDA FlashAttention (FA-2 math: one warp per query row, distributed (m, ℓ, õ) accumulator, delayed normalization, causal tile skipping + tail rebalancing, float4 K/V loads) | fp32 (and fp16 I/O with fp32 accumulation) |
| JAX/XLA standard | `einsum → softmax → einsum`, `@jax.jit` — XLA fuses the ops but **materializes the (B, H, N, N) score matrix**; the matmuls hit cuBLAS tensor-core paths | fp32 |
| JAX built-in (xla) | `jax.nn.dot_product_attention(implementation="xla")` | fp32 |
| JAX FA-2 tiled | The *same tiled FA-2 algorithm as CuLLM*, expressed as `lax.fori_loop`s whose per-tile einsums are cuBLAS matmuls | fp32 |
| cuDNN flash | NVIDIA's production fused flash attention (tensor cores, fp16-only by cuDNN requirement) | fp16 |

On "is the container set up for XLA": **JAX *is* an XLA frontend** — every
jitted function above already runs through XLA, and the cuDNN backend also
works out of the box. No Docker image change was needed.

## 2. Results (mean ms over 20 timed launches, 3 warmups)

CUDA timings via cudaEvent; JAX timings via wall clock around
`block_until_ready` (includes ~0.05–2 ms dispatch overhead — read the
N = 256 rows with that in mind).

### float32, non-causal

| N | CuLLM warp-coop | XLA standard | JAX built-in (xla) | JAX FA-2 tiled |
|---|---|---|---|---|
| 256 | 2.23 | **0.68** | 4.78 | 1.69 |
| 512 | 8.50 | **1.64** | 6.12 | 5.25 |
| 1024 | 33.99 | **5.49** | 19.84 | 19.25 |
| 2048 | 136.00 | **20.66** | 66.11 | 74.39 |
| 4096 | 549.02 | OOM-bound† | OOM-bound† | **292.48** |

† The standard paths materialize (8, 12, 4096, 4096) fp32 scores = **6.4 GB**
on a 12 GB card — skipped. This memory wall, not speed, is FlashAttention's
founding argument.

### float32, causal

| N | CuLLM warp-coop | XLA standard | JAX built-in (xla) | JAX FA-2 tiled |
|---|---|---|---|---|
| 256 | 1.28 | **0.60** | 8.52 | 1.60 |
| 512 | 4.56 | **1.69** | 8.68 | 5.31 |
| 1024 | 17.69 | **5.49** | 24.18 | 19.26 |
| 2048 | **69.82** | 20.65* | 80.92 | 74.25 |
| 4096 | **279.01** | OOM-bound† | OOM-bound† | 291.39 |

\* XLA's standard path computes the full N² and masks — it does no causal
work skipping, so its causal time equals non-causal. CuLLM's tile skipping
+ longest-block-first scheduling give a true ~1.96× causal speedup, which
is why CuLLM overtakes the FA-2 lax-loop reference here.

### float16 (production-backend context)

| N | causal | CuLLM (fp16 I/O, fp32 accum) | cuDNN flash (fp16, tensor cores) |
|---|---|---|---|
| 1024 | no | 37.83 | **4.19** |
| 2048 | no | 150.99 | **8.43** |
| 4096 | no | 606.23 | **21.81** |
| 2048 | yes | 79.43 | **4.58** |
| 4096 | yes | 312.84 | **13.47** |

### Accuracy (identical deterministic inputs, fp32)

CuLLM warp-cooperative vs JAX FA-2: max |Δ| between 5.4e-5 and 3.2e-4
across N ∈ {64, 100, 128, 150}, causal and not — pure
summation-order/association differences at fp32, both exact algorithms.

## 3. Interpretation: two separate axes, algorithm vs. engine

Achieved throughput at N = 2048, non-causal (4·B·H·N²·d = 103 GFLOP):

| Implementation | GFLOP/s | % of relevant peak | inner-product engine |
|---|---|---|---|
| CuLLM warp-coop fp32 | ~760 | ~6% of FP32 | scalar FFMA, one key/lane |
| JAX FA-2 tiled fp32 | ~1,390 | ~11% of FP32 | cuBLAS tile matmuls |
| XLA standard fp32 | ~4,990 | ~39% of FP32 | cuBLAS batched matmul |
| cuDNN flash fp16 | ~12,230 | ~24% of FP16-TC | tensor-core MMA, fused |

The matrix separates cleanly into **two orthogonal wins**:

1. **The algorithm win (IO-awareness).** Tiling + online softmax removes the
   N² memory footprint. That is what lets CuLLM and the FA-2 reference run
   N = 4096 where the standard paths exhaust a 12 GB card, and what
   produces the honest ~2× causal speedup that N²-materializing code cannot
   have. CuLLM fully delivers this win.
2. **The engine win (tensor cores).** Every implementation that beats CuLLM
   on wall-clock does its inner products on matmul hardware (cuBLAS or
   MMA). CuLLM's warp-cooperative kernel computes q·k with scalar FMAs —
   architecturally capped near ~1 TFLOP/s regardless of how good the
   algorithm is. cuDNN's flash attention is exactly "win 1 + win 2 in one
   kernel," and its ~16× lead over our fp16 path at N = 2048 is almost
   entirely the MMA engine.

Two corollaries worth stating plainly:

- **The single-slice numbers were misleading.** At B·H = 1 (earlier
  AGENTS.md table) our kernel looked strong because every contender was
  underutilizing the GPU; at B·H = 96 the kernel scales linearly (it was
  already occupancy-saturated), so per-slice cost is unchanged while
  cuBLAS-backed paths get *faster per slice* from batching.
- **Our fp16 path is currently slower than our fp32 path** (e.g. 151.0 vs
  136.0 ms at N = 2048): the kernel converts every element to float for
  arithmetic anyway, gains no math throughput, pays conversion cost, and
  the float4 vectorized load path is fp32-only. fp16 I/O only pays off with
  tensor-core math or half2 arithmetic.

## 4. What this means for CuLLM's roadmap

The measured gap decomposes the next performance milestone precisely:
replace the scalar q·k and p·V inner loops with **tensor-core tile matmuls**
(WMMA/`mma.sync`, 16×16×16 fp16/bf16 fragments with fp32 accumulators)
inside the existing FA-2 tile loop. The accumulator math, masking,
rebalancing, and logsumexp plumbing all stay; only the two inner products
change engines. The FA-2-lax-loop-beats-CuLLM-non-causal result is the
existence proof: same algorithm, tensor-core tiles, ~2× faster — and cuDNN
shows ~16× headroom at fp16.

## 5. Methodology fine print

- 20 timed launches after 3 warmups, mean reported. CUDA: cudaEvent around
  the launch loop. JAX: `time.perf_counter` around `block_until_ready`
  (jit compile excluded by warmups; dispatch overhead included).
- Identical deterministic inputs on both sides
  (`cullm_deterministic_values`), values in [-1, 1].
- fp32 rows are same-precision comparisons. The cuDNN column is fp16-only
  (cuDNN requirement) — compare it against CuLLM's fp16 row, and remember
  the accumulation precisions differ (cuDNN fp32 accumulators inside MMA;
  CuLLM fp32 accumulators around scalar math).
- Single GPU (RTX 3060, device-isolated in the container); no other GPU
  load during runs.
