# CUDA C++ vs. JAX, measured honestly: what I learned writing FlashAttention three times

A project write-up for a general technical audience. I implemented
transformer attention at every level of the GPU software stack — from
scalar CUDA I derived from the math myself, up through tensor cores and
NVIDIA's CUTLASS library — and benchmarked every version against JAX/XLA
and NVIDIA's production cuDNN on the same hardware, same inputs, same
shapes. This document is the conclusions. The full data and methodology
live in `AttentionBenchmarkReport.md` (same directory); the kernels, the
JAX references, and the 2,900-line LaTeX derivation all live in this
repository.

## The project in one paragraph

I derived FlashAttention from first principles (the math, not the paper's
pseudocode), then implemented the forward pass three times: (1) scalar
CUDA — one multiply-add per thread per element, the algorithm exactly as
derived; (2) the same algorithm with its two inner matrix products moved
onto tensor cores via WMMA, CUDA's built-in fragment API; (3) rebuilt on
CUTLASS/CuTe, NVIDIA's kernel-building library, with cp.async
double-buffered tile loads and a register-resident output accumulator.
Each version is unit-tested against an independent CPU reference. Then I
benchmarked all three against four JAX-side implementations — XLA's fused
standard attention, JAX's built-in attention op, my same tiled algorithm
expressed in `lax` loops, and cuDNN's fused flash attention via
`jax.nn.dot_product_attention` — at realistic multi-head scale
(batch·heads = 96, head dim 64) on an RTX 3060.

## The result table (N = 2048 tokens, batch·heads = 96, d = 64, mean ms)

| Implementation | ms | inner-product engine |
|---|---|---|
| cuDNN flash attention (fp16) | 8.4 | fused tensor-core MMA — the production target |
| **mine: CUTLASS/CuTe (fp16)** | **19.0** | MMA + cp.async + register-resident accumulator |
| JAX/XLA fused standard (fp32/TF32) | 20.7 | cuBLAS batched matmul; materializes N² |
| **mine: WMMA tensor-core (fp16)** | **52.1** | wmma 16×16×16 fragments |
| JAX built-in attention (`implementation="xla"`) | 66.1 | XLA |
| JAX FA-2 — my algorithm in `lax` loops | 74.4 | cuBLAS tile matmuls |
| **mine: scalar warp-cooperative** | **151.1** | one FMA per lane per element |

Three headline numbers:

- **6.4 GB vs. never**: XLA's standard attention materializes the full
  N×N score matrix — at N = 4096 that buffer alone is 6.4 GB and cannot
  run on a 12 GB card. The flash kernels stream tiles and don't notice.
  This — memory, not speed — is FlashAttention's actual founding argument.
- **~2× causal**: my kernels skip the masked half of the work (and
  schedule the expensive row-blocks first). XLA's standard path computes
  all of it and throws half away — its causal time equals its non-causal
  time.
- **18× → 2.3×**: how far behind cuDNN my scalar kernel started, and where
  the CUTLASS version ended (1.14× — essentially tied — at N = 1024).

The memory claims above are **measured, not computed**
(`AttentionMemoryReport`, cudaMemGetInfo around real allocations): at
N = 4096 the flash working set measures 0.375 GiB while the standard
path's allocation fails live at 12.375 GiB requested on the 11.6 GiB
card. The same tool reports each kernel's on-chip budget — which is where
the ladder rungs *do* differ, since their HBM footprints are identical by
design: the scalar thread-per-row kernel's 254 registers/thread cap it at
192 resident threads/SM (the occupancy number behind its slowness); WMMA
spends 47.9 KB shared memory/block (2 blocks/SM) staging its opaque
fragments; CuTe halves that to 23.3 KB by keeping the accumulator in
registers (120 regs/thread, 4 blocks/SM) — the WMMA→CuTe upgrade,
expressed in hardware units instead of milliseconds. And a nuance worth
noticing: CuTe wins despite *lower* occupancy than the scalar
warp-cooperative kernel (512 vs. 640 threads/SM) — occupancy is a means,
not the metric.

## The central lesson

> GPU performance is two independent games: the **algorithm** decides how
> many bytes you move; the **hardware engine** decides how fast you crunch
> what's left. FlashAttention wins the first. Tensor cores win the second.
> cuDNN does both in one kernel — and this project measured exactly what
> each one is worth.

The table separates on precisely this line. Every implementation that
beat my scalar kernel on wall-clock runs its inner products on matmul
hardware (cuBLAS or MMA instructions) — a scalar-FMA kernel is
architecturally capped near ~6% of the card's peak regardless of how good
its algorithm is. And every implementation my kernels beat on *memory*
(or on causal masking) is paying for materializing the N² matrix that the
flash algorithm never builds.

## What this actually says about JAX

**JAX is not "slow Python." It's a compiler front-end, and a good one.**
`@jax.jit` traces your function once; XLA compiles it into fused CUDA
kernels and cuBLAS/cuDNN calls. Python never touches the hot loop. The
concrete evidence in the table: three lines of jitted `einsum` + `softmax`
(the "fused standard" row) hit ~5 TFLOP/s — ~39% of my card's FP32 peak —
and beat my first two hand-written kernels outright. If your problem fits
the operations XLA fuses well, you get tensor-core-class performance for
free, in Python, with autodiff included.

**Where hand-written CUDA is genuinely justified — because I measured the
boundary, not guessed it:**

1. **The memory wall.** XLA's standard attention cannot run N = 4096 on
   my card; the fused-kernel approach (never materializing N²) requires
   *owning the kernel* — no amount of XLA fusion of the standard formula
   gets you there. (JAX's answer is to call cuDNN's flash kernel — i.e.,
   someone else's hand-written CUDA.)
2. **Structured work-skipping.** Causal masking is ~2× real savings in a
   kernel that skips masked tiles; XLA's standard path can't skip — it
   masks after computing.
3. **The last multiples.** cuDNN is 2.3× ahead of my best kernel and the
   remaining gap decomposes into named, measurable techniques (larger
   pipeline tiles, LDSM shared-to-register copies, swizzled layouts).
   If those multiples matter for your product, that work is what buying
   or writing a production kernel gets you.
4. **Understanding.** After this project I know which specific hardware
   instruction bought each speedup. That transfers to every other GPU
   problem I touch; calling `jax.nn.dot_product_attention` teaches none
   of it.

**Rule of thumb the data supports**: prototype in JAX always; hand-write
CUDA only when you can name the specific thing XLA cannot do for your
problem — a memory wall, a fusion, a skip pattern — and verify the claim
with a measurement before committing to kernel work.

## Benchmarking pitfalls I hit (so you don't)

- **JAX's "fp32" matmuls are TF32 by default on Ampere+** — 10-bit
  mantissa on tensor cores. An "fp32 JAX vs. fp32 CUDA" comparison is
  quietly tensor-core vs. scalar unless you set
  `jax.default_matmul_precision('highest')`. Decide which comparison you
  mean, and say so.
- **JAX dispatch is asynchronous** — Python returns before the GPU
  finishes. Timing without `block_until_ready()` measures dispatch, not
  compute. (CUDA-side equivalent: cudaEvent timing around the launches.)
- **cuDNN's flash attention is fp16/bf16-only** (fp32 accumulation inside
  the MMA). Any comparison against it is a mixed-precision comparison;
  keep your own kernel's fp16 row next to it, not the fp32 one.
- **Single-slice benchmarks mislead.** At batch·heads = 1, my scalar
  kernel looked competitive — because every contender was underutilizing
  the GPU. At batch·heads = 96 the cuBLAS-backed paths got faster per
  slice from batching while my occupancy-saturated kernel scaled
  linearly. Benchmark at your real problem's parallelism, not a toy's.
- **A "why not cuBLAS inside my kernel" dead end, quantified**: cuBLAS is
  a host API — you can't call it from inside a kernel, so tile-level use
  means round-tripping intermediates through global memory. The table's
  `lax`-loop row (74 ms) *is* that ceiling, measured: same algorithm as my
  kernels, cuBLAS tile matmuls, no fusion.
- **WMMA's fragments are opaque** — a lane can't know which matrix rows
  its registers hold, which forced my per-row rescale through shared
  memory every tile. CUTLASS/CuTe's coordinate tensors expose the layout,
  letting the accumulator live in registers across all tiles. That single
  difference was worth ~2.7× — it's the concrete reason to graduate from
  WMMA to CUTLASS, not library fashion.

## Honest scope and what's next

This is a forward-pass, fp16, fixed-head-dim (64) inference-tier
comparison on one consumer GPU (RTX 3060, Ampere). It is not
production-ready software and doesn't claim to be. The named next steps,
deliberately deferred: bf16 in the tensor-core kernels (mechanical type
swap; the interesting outcome would be numerics — bf16 trades mantissa
for exponent range — rather than speed) and a tensor-core backward pass
(a genuinely larger job: MMA recomputation of score tiles plus a second
transposed-operand MMA pass for the weight gradients).

## Where everything lives

- Full benchmark report + methodology: `AttentionBenchmarkReport.md`
- Rerun everything yourself (exact commands, all six executables):
  `ReproducingTheBenchmarks.md`
- Kernels: `../Source/Transformer/Attention/` (scalar:
  `flash_attention_warp_cooperative.h`; WMMA:
  `flash_attention_tensor_core.h`; CUTLASS: `flash_attention_cute.h`)
- JAX reference implementations + comparison harness: `../Python/`
- The math the kernels were derived from:
  `../../../Documents/FlashAttention/FlashAttention.tex` (compiled PDF
  alongside it)
- Video treatments of this material: `AttentionBenchmarkPresentation.md`
  (economized, current cut: one ~8-10 min long-form talk + two ≤2 min
  shorts, timed and paired with actual benchmark screenshots),
  `AttentionBenchmarkShortForm.md` and `AttentionShortFormVisualStoryboard.md`
  (earlier drafts, superseded but kept for reference), and
  `AttentionSeries.md` (the from-first-principles series building up to it)
