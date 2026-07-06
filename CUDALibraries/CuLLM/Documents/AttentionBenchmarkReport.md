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

## 5. Follow-up (same day): the WMMA tensor-core kernel

Section 4's milestone was implemented as
`Transformer/Attention/flash_attention_tensor_core.h`: the same FA-2 loop
with the two inner products moved onto tensor cores via nvcuda::wmma
16×16×16 half fragments with float accumulators (one warp per 16-row query
tile; online softmax stays scalar; P rounded to half before P·V, the same
choice cuDNN makes). Correctness: 8 unit tests vs a double CPU reference,
fp16-scale tolerances, including ragged lengths, causal, multi-slice, and
head dims 32/48/64.

Measured at the same B·H = 96, d = 64 shapes (mean ms):

| N | causal | scalar fp16 | WMMA fp16 | speedup | cuDNN fp16 | remaining gap |
|---|---|---|---|---|---|---|
| 1024 | no | 37.8 | 13.7 | 2.8× | 4.19 | 3.3× |
| 2048 | no | 151.1 | 52.1 | 2.9× | 8.43 | 6.2× |
| 4096 | no | 606.3 | 229.3 | 2.6× | 21.8 | 10.5× |
| 2048 | yes | 79.8 | 27.0 | 3.0× | 4.58 | 5.9× |
| 4096 | yes | 312.4 | 105.2 | 3.0× | 13.5 | 7.8× |

Three consequences:

- The engine hypothesis is confirmed experimentally: changing only the
  inner-product engine (identical algorithm, masking, scheduling) bought
  2.6–3×, and the WMMA kernel now beats the JAX FA-2 lax-loop reference
  (52.1 vs 74.4 ms at N = 2048 non-causal).
- fp16 I/O finally pays for itself: the WMMA fp16 kernel is ~2.6× faster
  than our fp32 scalar kernel, where the scalar fp16 path had been *slower*
  than fp32.
- The remaining ~6–10× to cuDNN is the classic optimization ladder above
  naive WMMA: per-tile shared-memory round trips of S and P·V (our rescale
  merge goes through shared because WMMA fragments are opaque), no
  cp.async double buffering of K/V tiles, no swizzled layouts, one warp
  per 16 rows instead of warpgroup-wide tiles. Each is a named, measurable
  next step — CUTLASS/CuTe is the library that packages exactly these.

## 6. Second follow-up: the CuTe (CUTLASS) kernel

`Transformer/Attention/flash_attention_cute.h` (CUTLASS v4.5.2, vendored
header-only at `CUDALibraries/ThirdParty/cutlass`, gitignored). Three
engine upgrades over the WMMA kernel, same FA-2 math: (1) the output
accumulator stays in registers across all tiles — CuTe's coordinate
tensors expose each lane's fragment (row, col), so the per-row rescale
applies to the live fragment and P·V mma-accumulates into it, deleting the
WMMA kernel's per-tile shared round trip; (2) cp.async double-buffered K/V
tiles overlap the next copy with the current tile's math; (3) Q fragments
load once and persist. Correctness: 6 unit tests incl. the ragged
scalar-fallback path and long multi-buffer runs.

| N | causal | scalar fp16 | WMMA | CuTe | cuDNN | CuTe vs cuDNN |
|---|---|---|---|---|---|---|
| 1024 | no | 37.8 | 13.7 | 4.79 | 4.19 | 1.14× |
| 2048 | no | 151.1 | 52.1 | 19.0 | 8.43 | 2.3× |
| 4096 | no | 606.3 | 228.2 | 76.0 | 21.8 | 3.5× |
| 2048 | yes | 79.8 | 27.0 | 9.94 | 4.58 | 2.2× |
| 4096 | yes | 312.4 | 105.1 | 38.8 | 13.5 | 2.9× |

The ladder, cumulative at N = 2048 non-causal: scalar 151 → WMMA 52 →
CuTe 19.0 vs cuDNN 8.4 — from 18× behind to 2.3×. The CuTe kernel now
beats XLA's fused standard attention outright (19.0 vs 20.7 ms, and 9.9 vs
20.7 causal) while still never materializing N². At N = 1024 it is within
14% of cuDNN. The growing gap at N = 4096 points at the remaining rungs:
larger K/V tiles per stage (more math per sync), LDSM shared→register
copies, swizzled shared layouts, and splitting softmax work across all 32
lanes.

## 7. Methodology fine print

- **cuDNN's 8.43 ms figure (N=2048 non-causal, fp16) was independently
  reverified 2026-07-05**, after two screenshots dated 2026-07-04 turned
  up showing 6.505 / 7.097 ms instead — a 20%+ gap too large to be normal
  run-to-run noise. Four fresh readings the next day (3 back-to-back
  reruns plus a new screenshot) all landed at 8.4-8.7 ms, matching this
  section's original number and none reproducing the 07-04 anomaly. Take
  the 07-04 screenshots as an unreproduced one-off (likely a cuDNN
  algorithm-selection quirk in that container session), not evidence this
  figure needs revising.
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

## 8. Status and what's next

As of 2026-07-03, this benchmark and the ladder it documents (scalar →
WMMA → CuTe) are considered **feature-complete**. Further kernel
implementation is paused; active work has shifted to turning this report
into presentation material — see `AttentionBenchmarkShortForm.md` for the
video/talk treatment.

Two follow-on engineering items are identified but deliberately **not**
scheduled unless a specific need arises:

- **bf16 in the WMMA/CuTe kernels.** Both are `__half`-only today; the MHA
  pipeline supports bf16 end-to-end but only through the scalar
  warp-cooperative kernel, so bf16 never reaches tensor-core speed. The
  change is mechanical (swap the fragment/copy element type; CUTLASS ships
  an `F32BF16BF16F32` SM80 atom for exactly this), so the expected result is
  near-identical timings to the `__half` kernels — the interesting delta
  would be numerical (bf16's exponent range vs `__half`'s mantissa
  precision), not speed.
- **A tensor-core backward pass.** The backward kernels (single-head and
  the GQA/MQA group-reduction variant) are scalar throughout — the forward
  ladder's engine work was never applied there. This is substantially more
  involved than the forward changes (recomputing P tiles via MMA, then a
  second MMA pass for dS·K and dS^⊤·Q against a transposed operand layout)
  and would need its own scoping pass before starting.
