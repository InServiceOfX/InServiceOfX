# AGENTS.md — CuLLM / FlashAttention-from-first-principles pickup guide

You are an AI coding agent (Claude Code, Codex, OpenClaw, Hermes, Grok, or
other) pointed at this workstream. This file is self-contained: read it,
skim the two source files it points at first, and you should have full
context to continue without re-deriving anything.

If the user says "continue the FlashAttention work" or "what's next for
CuLLM" or similar, start here.

## What this is, in one paragraph

An implementation of FlashAttention (forward + backward, causal masking,
multi-head) in `CUDALibraries/CuLLM`, built **from first principles** —
every kernel is derived from the math in a from-scratch LaTeX writeup
(`Documents/FlashAttention/FlashAttention.tex`, Papers I–III: Vaswani et al.
2017, Milakov & Gimelshein 2018, Dao et al. 2022 FlashAttention, Dao 2023
FlashAttention-2) rather than transcribed from any single paper's pseudocode
or ported from an existing implementation. Performance techniques are then
cross-checked against Karpathy's `llm.c` (`/media/ernest/Samsung980ProPCI/PropD/llm.c`)
and the project's own `MoreCUDA` utility library, but the *design* — module
boundaries, naming, what's compile-time vs. runtime — follows the patterns
already established in `CUDALibraries/CuLLM/Source/Transformer/`, not
Karpathy's C style. Code comments reference tex **section names**, never
section/equation numbers (numbers drift; names don't).

The end state as of this file's writing: complete multi-head attention
forward AND backward passes — dense QKV/output linear maps via cuBLASLt
(with weight gradients dW_qkv, dW^O through transposed GEMMs), IO-aware
FlashAttention core (two execution-mapping variants, float4-vectorized K/V
loads, causal tail rebalancing), causal masking, batch/head parallelism,
grouped-query/multi-query attention forward AND backward, and end-to-end
`__half` / `__nv_bfloat16` paths — all unit-tested against independent CPU
references, finite-difference gradient checks, and now JAX reference
implementations for standard attention, online softmax, FlashAttention, and
FlashAttention-2.

## Read in this order

1. **This file** — orientation, file map, build commands, design decisions.
2. **`Documents/FlashAttention/FlashAttention.tex`** — the math. Compile with
   `pdflatex -interaction=nonstopmode FlashAttention.tex` (run twice for
   cross-references) from `Documents/FlashAttention/`, or read the committed
   `FlashAttention.pdf` in the same directory. Section list is in "Math →
   code map" below.
3. **The code itself** — `CUDALibraries/CuLLM/Source/Transformer/`. Every
   header has a long doc comment deriving its kernel from the tex; read the
   comment before the code.
4. **JAX comparison path** — `CUDALibraries/CuLLM/Python/`, especially
   `jax_attention_reference.py`, `test_jax_attention_reference.py`, and
   `compare_cullm_jax_attention.py`. These are not production kernels; they
   are reference implementations and benchmark wrappers for validating CuLLM
   against JAX inside the `propulsion-with-cuda:26.02-py3` container.
5. **Git log** on `feature/scaled-dot-product-attention` (or `master`, which
   was fast-forwarded to include it as of this writing) — each commit is a
   complete, tested increment; commit messages explain *why*, not just
   *what*.

## Repo locations

```
InServiceOfX/
  Documents/FlashAttention/
    FlashAttention.tex        # the math, ~2900 lines, compiles standalone
    FlashAttention.pdf        # committed compiled output
  CUDALibraries/
    CuLLM/
      AGENTS.md                # THIS FILE
      Source/
        Transformer/
          Attention/           # the FlashAttention core (this file's focus)
          MultiHeadAttention/   # QKV/output linear maps + full MHA composition
          Softmax/              # standalone softmax kernels (design-space exploration)
        LLM/                   # legacy llm.c-style prototypes; currently empty after cleanup
        Benchmarks/            # AttentionIOBenchmark executable
        UnitTests/              # mirrors Source/ tree; gtest
      Python/
        jax_attention_reference.py        # standard/online/FA/FA-2 JAX refs
        test_jax_attention_reference.py   # pytest accuracy tests
        compare_cullm_jax_attention.py    # CuLLM-vs-JAX accuracy/perf harness
      BuildGcc/                 # out-of-tree cmake build dir (gitignored)
    MoreCUDA/
      Source/
        Numerics/MathFunctions.h            # exp/max/sqrt/log across float/double/__half
        Transformer/Softmax/AccumulationType.h  # NOT here -- lives in CuLLM; see below
        Utilities/Memory/streaming_load.h, streaming_store.h   # __ldcs/__stcs wrappers
        cuBLASWrappers/                     # cuBLASLt GEMM wrapper (Setup<T>, LtMatrixMultiplication<T>)
        StreamManagement/                   # CUDA stream RAII wrapper
```

Note: `AccumulationType.h` is under `CuLLM/Source/Transformer/Softmax/`, not
`MoreCUDA` — it's CuLLM-specific (selects float vs. double accumulation per
I/O type) even though `MathFunctions.h` (which it depends on) is in
`MoreCUDA`.

## Math → code map

The tex is organized in three "Papers" (Parts). Only Papers II and III have
direct code; Paper I is architecture background.

| Tex section | Code |
|---|---|
| Scaled Dot-Product Attention | `Attention/attention_scores.h`, `attention_weighted_values.h`, `scaled_dot_product_attention.h` (the *standard*, non-flash baseline — S and P materialized in HBM, used as the correctness/IO reference) |
| Multi-Head Attention | `MultiHeadAttention/*.h` (see below) |
| The Decoder Stack (causal mask def.) | `kCausal` template param threaded through every kernel below |
| The Safe-Softmax Statistics, The Merge Monoid, The Online Algorithm as a Left Fold | `Softmax/softmax_warp_fold_reduce.h`'s `SafeSoftmaxAccumulator` + `merge()` |
| The Attention Output Accumulator | `Attention/AttentionAccumulator.h` — the `(m, ℓ, õ)` triple + `merge()`, the core abstraction everything else is built from |
| The FlashAttention Algorithm | `Attention/flash_attention_forward.h` (one thread per query row) |
| What Becomes Linear, and What Does Not | `Python/jax_attention_reference.py` + `Benchmarks/Transformer/Attention/attention_reference_dump.cu` make the distinction executable: JAX references show exact dense attention remains quadratic in work while CuLLM avoids materializing S/P |
| FlashAttention-2: Reducing Non-Matmul FLOPs | `Attention/flash_attention_warp_cooperative.h` (one **warp** per query row — the higher-performance variant; also the one `MultiHeadAttention` uses) |
| Gradients of Scaled Dot-Product Attention, Recomputation and the Logsumexp Statistic, The FlashAttention Backward Pass | `Attention/flash_attention_backward.h` |
| Parallelism and Work Partitioning | `flash_attention_forward.h`'s / `flash_attention_warp_cooperative.h`'s `gridDim.y` batch/head dimension; split-Q is literally what "one warp per row" *is* |
| FlashAttention-2 from First Principles | `Python/jax_attention_reference.py` implements the same representative/schedule distinctions at the JAX level; `compare_cullm_jax_attention.py` cross-checks CuLLM output against the JAX FA-2 reference |

## Build & test

```bash
cd CUDALibraries/CuLLM/BuildGcc   # create with `mkdir BuildGcc` if absent
cmake ../Source
make Check -j4
./Check                             # full suite
./Check --gtest_filter='FlashAttention*'   # or any substring
```

As of this file's writing: **74 tests, 23 suites, all passing** (plus
MoreCUDA's 123), on RTX 30xx-class hardware (sm_86). `CMAKE_CUDA_ARCHITECTURES` is hardcoded to `75 86` in
`Source/CMakeLists.txt` — add your arch if different.

Benchmark: `make AttentionIOBenchmark -j4 && ./AttentionIOBenchmark` (from
`BuildGcc/`). Sweeps sequence length, compares standard vs. flash
(thread-per-row) vs. flash (warp-cooperative) vs. causal, prints a
`max |diff|` exactness column against the standard-attention baseline.
Measured on the RTX 3070 (n = sequence length, d = 64):

| n | standard ms | flash (thread) ms | flash (warp) ms | standard/warp speedup | causal (warp) ms |
|---|---|---|---|---|---|
| 256 | 0.11 | 0.34 | 0.036 | 3.1x | 0.036 |
| 1024 | 1.53 | 1.30 | 0.44 | 3.5x | 0.24 |
| 4096 | 18.79 | 4.12 | 4.4–4.6 | ~4.1–4.2x | 2.4 |

Causal tile-skipping gives ~1.8x over non-causal at the warp-cooperative
granularity (matches Dao 2023's reported 1.7–1.8x). At thread-per-row
granularity it only gave ~5% because a single tail row-block dominates wall
time with too few blocks in flight — this is exactly the "load imbalance"
remark in the tex's Causal Tile Skipping section.

JAX reference tests and comparison:

```bash
# Run inside the PropulsionWithCUDA Docker container.
cd /InServiceOfX
python3 -m pytest CUDALibraries/CuLLM/Python/test_jax_attention_reference.py -q

mkdir -p CUDALibraries/CuLLM/BuildDocker
cd CUDALibraries/CuLLM/BuildDocker
cmake ../Source
make AttentionReferenceDump AttentionIOBenchmark -j4

cd /InServiceOfX
python3 CUDALibraries/CuLLM/Python/compare_cullm_jax_attention.py \
  --build-dir CUDALibraries/CuLLM/BuildDocker
```

As of 2026-07-03, `propulsion-with-cuda:26.02-py3` already has JAX installed
(`jax 0.10.2`) and sees GPU 1 through the QuickDockerBuilder run wrapper, so
no Dockerfile change or image rebuild was needed (JAX *is* an XLA frontend;
the cuDNN flash-attention backend of `jax.nn.dot_product_attention` also
works out of the box, fp16/bf16 only). Verification results:
`test_jax_attention_reference.py` passed 5/5; direct CuLLM
warp-cooperative-vs-JAX-FA-2 max errors were ~`5e-5` to `3.25e-4` in
float32 (expected schedule/association differences); on RTX 3060, CuLLM
warp-cooperative attention at `d=64` measured about 0.043/0.121/0.386/1.44/5.45
ms for `n=256/512/1024/2048/4096`.

**Multi-head benchmark report** (2026-07-03): `Documents/AttentionBenchmarkReport.md`
(+ `AttentionBenchmarkShortForm.md` for presentation material), generated by
`Python/benchmark_report.py` + `Benchmarks/.../warp_attention_benchmark.cu`
at B·H = 96, d = 64. Headline findings — read before quoting any perf
numbers: (1) the single-slice table above is misleading at real multi-head
scale: the warp kernel is occupancy-saturated at one slice, so 96 slices
cost 96×, while cuBLAS/XLA paths get faster per slice from batching;
(2) CuLLM wins the *memory* axis (N = 4096 runs where XLA's standard path
needs a 6.4 GB score buffer) and the causal axis (~1.96× true tile
skipping), but loses raw throughput to every tensor-core-backed
implementation — XLA fused standard ~6.6× faster at fp32, cuDNN flash ~16×
at fp16; (3) CuLLM's fp16 path is *slower* than its fp32 path (float
conversion cost, no half2 math, float4 loads are fp32-only). The measured
next milestone: tensor-core (WMMA/mma.sync) tile matmuls inside the FA-2
loop — the FA-2-lax-loop reference beating our kernel non-causally is the
existence proof that the algorithm is fine and the inner-product engine is
the gap.

MoreCUDA has its own standalone build (`MoreCUDA/BuildGcc`, same
`cmake ../Source && make Check`); **re-run it after touching any file under
`MoreCUDA/Source/`** — CuLLM pulls MoreCUDA in via `ADD_SUBDIRECTORY`, and a
CMake fix that works for CuLLM's build can silently break MoreCUDA's
standalone build if done carelessly (see gotcha below).

## Key design decisions (the "why", not just the "what")

- **`kHeadDim` is always a compile-time template parameter**; batch size and
  number of heads are always runtime. This mirrors llm.c's own choice (head
  dim fixes register array sizes) while letting one binary serve any
  batch/head count via the `gridDim.y` trick below.
- **`gridDim.y` = flattened `(batch, head)` index.** Every attention kernel
  offsets all its pointers by `blockIdx.y * sequence_length * kHeadDim` at
  kernel entry and otherwise ignores batching entirely — heads and batch
  elements are *provably* independent (see
  `UnitTests/Transformer/Attention/multihead_flash_attention_tests.cu` and
  `.../MultiHeadAttention/multi_head_attention_tests.cu`: batched launches
  are bit-identical to per-slice launches). Tensors are laid out
  `(B·NH, T, d)` row-major throughout — never `(B, NH, T, d)` as four
  separate dims; the flattening is load-bearing for this trick.
- **Two forward kernels, deliberately kept both**: `flash_attention_forward.h`
  (one thread per query row — simple, low occupancy) and
  `flash_attention_warp_cooperative.h` (one **warp** per row, `AttentionAccumulator`
  distributed across lanes: `(m, ℓ)` replicated, `õ` sharded
  `fragment[f] = õ[f·32 + lane]`). The warp version requires
  **`kHeadDim` to be a multiple of 32** (`static_assert`'d) — the thread
  version has no such restriction. `MultiHeadAttention/` always uses the
  warp-cooperative kernel; if you need `kHeadDim` not a multiple of 32,
  you'll need to either extend the thread-per-row path into
  `MultiHeadAttention` or pad `kHeadDim`.
- **`AttentionAccumulator` merge is guarded on `max_value == -∞`, not
  `sum == 0`.** This was a real bug caught mid-session: causally masked
  scores produce accumulators with `max = -∞` but possibly nonzero `sum`
  from earlier merges, and merging two such accumulators via the naive path
  computes `exp(-∞ - (-∞)) = NaN`. The identical fix was needed in
  `Softmax/softmax_warp_fold_reduce.h`'s `merge()`. If you add a third
  accumulator-merging kernel, this guard is not optional.
- **Row-major GEMM via cuBLASLt's column-major-only API**: `cuBLASLt`
  computes `D = A·B` column-major, full stop. To get row-major
  `Out(m,n) = X(m,k)·W(k,n)` without any physical transpose, both
  `qkv_linear_maps.h` and `output_linear_map.h` use the identity
  `Out^⊤ = W^⊤X^⊤`, and the fact that a row-major `(r,c)` buffer read as
  column-major `(c,r)` **is** that matrix's transpose (same bytes, no copy).
  So the GEMM is called with operands swapped: `A := W`, `B := X`,
  `M_cublas = n`, `K_cublas = k`, `N_cublas = m`. Full derivation in
  `qkv_linear_maps.h`'s header comment. This is verified against a
  from-scratch CPU reference in the tests, not just trusted from the
  derivation — if you touch this, re-verify the same way.
- **Fused QKV linear maps** (one GEMM against
  `W_qkv = [W^Q | W^K | W^V]`, each further split into per-head blocks
  `[W^Q_0 | ... | W^Q_{NH-1}]`) instead of `3·NH` separate narrow GEMMs —
  better arithmetic intensity, matches llm.c's own convention. See
  `split_qkv_heads.h`'s header comment for the exact byte layout.
- **`streaming_load`/`streaming_store`** (`__ldcs`/`__stcs`, PTX
  cache-streaming hints) are used wherever a memory address is touched for
  the *last* time in a kernel — e.g. the final normalize-and-write pass of a
  softmax kernel. The rule of thumb documented across all three "block"/"warp
  streaming" softmax kernels: the *first* touch of an address that will be
  read again should use an ordinary load (keeps it warm in L1/L2); only the
  *last* touch should stream.
- **Draft cleanup:** the old `Drafts/LLM/AttentionForward/softmax.h` scratch
  kernels were deleted after confirming the useful ideas had either been
  promoted into `Transformer/Softmax/` (online reduction, recompute instead of
  caching, streaming cache hints) or captured in the backlog as profiling work
  (vectorized loads and attention-specific softmax variants).

## What's done (checklist)

- [x] Standard (non-flash) scaled dot-product attention, causal + non-causal
- [x] FlashAttention forward, both execution mappings, causal + non-causal
- [x] FlashAttention backward (single-head; see gap below)
- [x] Batch/head parallelism via `gridDim.y`, all forward kernels
- [x] Fused QKV linear maps (cuBLASLt GEMM + permute kernel)
- [x] Output linear map (merge-heads kernel + cuBLASLt GEMM)
- [x] Full `multi_head_attention()` composition, end-to-end tested against an
      independent CPU MHA reference (causal + non-causal)
- [x] IO-complexity benchmark (`AttentionIOBenchmark`)
- [x] Direct JAX reference path:
      `Python/jax_attention_reference.py` implements standard attention,
      online softmax, tiled FlashAttention, and FA-2 delayed normalisation;
      `AttentionReferenceDump` + `compare_cullm_jax_attention.py` compare
      CuLLM CUDA output and timings against JAX inside Docker.
- [x] Tex extended with FlashAttention-2 material (non-matmul FLOPs
      proposition, causal tile-skipping proposition, parallelism/warp-
      partitioning remarks) — not just FlashAttention v1
- [x] Tex clarified what becomes linear: FlashAttention makes peak auxiliary
      storage linear by not materialising S/P, but exact dense attention keeps
      quadratic query-key work.
- [x] Legacy `LLM/attention_forward.h` deleted after confirming its only
      remaining idea (packed fused-QKV input convention) had been promoted into
      `MultiHeadAttention/split_qkv_heads.h` and the Transformer attention
      kernels
- [x] Legacy `Drafts/LLM/AttentionForward/softmax.h` deleted after confirming
      its remaining useful tricks are either implemented in `Transformer/Softmax`
      or tracked as profiling/backlog items
- [x] Repo cleanup: `LLM/AttentionForward/FlashAttention.h` deleted (it
      silently renormalized every tile despite a comment claiming otherwise
      — an active anti-pattern, not just outdated); `LLM/AttentionForward/softmax.h`'s
      one genuinely novel technique (recompute-instead-of-cache +
      streaming loads) ported into `Softmax/softmax_warp_streaming_fused.h`
      and the original deleted

## What's NOT done (the actual backlog)

All seven items of the original backlog were completed 2026-07-02 (branch
`feat/mha-backward-and-backlog`; see that branch's commit messages for the
measurements). What they became:

1. Multi-head backward → `MultiHeadAttention/multi_head_attention_backward.h`
   (`linear_map_backward` uses `Setup`'s `is_transpose_on_A/B`;
   `split_heads`/`merge_qkv_heads` are the permutation adjoints; verified by
   central-difference gradient checks of dX, dW_qkv, dW^O, causal and not).
2. GEMM benchmark → `MultiHeadAttention/tiled_gemm.h` +
   `Benchmarks/Transformer/MultiHeadAttention/linear_map_gemm_benchmark.cu`.
   **Measured (RTX 3060): cuBLASLt wins 10–12x** (~8 TFLOP/s vs ~0.68) at
   d_model 256–1024 — keep cuBLASLt for the linear maps.
3. `kHeadDim % 32` → clear `static_assert` at the `multi_head_attention()`
   / `grouped_query_attention()` API boundaries.
4. float4 loads → K/V tile loads in the warp-cooperative forward, behind
   `if constexpr` (float only). **Measured ~1–1.5%** — confirming the
   occupancy hypothesis — so NOT propagated to the backward kernels.
5. Causal rebalancing → longest-row-block-first remap under `kCausal`
   (forward + backward Pass 1; Pass 2's natural order is already correct).
   **Measured: −8% at n=2048, −5.4% at n=4096**; ~10 µs slower at n≤1024
   where the grid fits in one or two waves.
6. MQA/GQA → `MultiHeadAttention/grouped_query_attention.h` +
   `split_grouped_qkv_heads`; the attention kernel's
   `(num_heads, kv_group_size)` params map query slice b·NH+h to K/V slice
   b·NKV+h/g (defaults = identity). Forward only.
7. `__half` end-to-end → `multi_head_attention_half_tests.cu`. Exposed and
   fixed a real MoreCUDA bug: `Setup<T>::setup()` forced CUDA_R_32F scale
   type for every T; COMPUTE_16F/64F need CUDA_R_16F/64F or the heuristic
   returns zero algorithms.

Remaining (new) backlog:

1. ~~GQA/MQA backward~~ DONE 2026-07-02:
   `grouped_query_attention_backward.h`. The backward gradient kernels take
   the same (num_heads, kv_group_size) slice map as the forward; Pass 2
   writes dK/dV as per-query-head partials (single-writer, no atomics) and
   `reduce_grouped_kv_gradients` performs the tex remark's group sum;
   `merge_grouped_qkv_heads` is the grouped split's adjoint. Verified by
   finite-difference gradient checks at g=2, causal g=2, and g=NH (MQA).
2. ~~bfloat16~~ DONE 2026-07-02: `__nv_bfloat16` runs the MHA pipeline
   end-to-end (`multi_head_attention_bfloat16_tests.cu`). Key wrinkle: no
   `CUBLAS_COMPUTE_16BF` exists — bf16 GEMMs are R_16BF data under
   COMPUTE_32F with *float* alpha/beta, hence `ComputeParameters::scale_type_`
   and `host_scale_type_t<T>` in the cuBLASWrappers. bf16 math intrinsics
   are guarded `__CUDA_ARCH__ >= 800`; conversions work on all archs.
3. **AttentionAccumulator::merge is not exercised by the production
   kernels** (the warp-cooperative kernel distributes the accumulator
   across lanes instead). It now uses `get_approximate_exponential`; if a
   sequential-tile kernel is ever built on it, benchmark that choice.

## Known gotchas

- **`cuBLASWrappers` and `StreamManagement` (both in `MoreCUDA/Source/`) had
  no `TARGET_INCLUDE_DIRECTORIES` of their own** — they silently relied on
  `MoreCUDA/Source/CMakeLists.txt`'s global `INCLUDE_DIRECTORIES`, which
  only runs when MoreCUDA is the top-level CMake project. Pulling either
  into another project's build via `ADD_SUBDIRECTORY` failed with
  "No such file or directory" on their own internal `#include`s until this
  was fixed (mirroring how `MoreCUDAUtilities`'s `CMakeLists.txt` already
  did it: `TARGET_INCLUDE_DIRECTORIES(<target> PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)`).
  **If you add a new MoreCUDA subdirectory library and pull it into CuLLM,
  check it has this line — it very likely doesn't yet.**
- `flash_attention_warp_cooperative`'s `static_assert` on `kHeadDim % 32 == 0`
  fires as a deeply nested template-instantiation error, not a friendly
  message at the call site — see backlog item 4.
- The tex file lives on a path under `/media/...` (an external/removable
  drive mount) but is the *same repo* as the `/home/.../workspace2/...`
  path — verify with `readlink -f` on both before assuming they're
  different clones; they aren't, they're the same working tree mounted
  twice.

## Git / branch policy

The root `AGENTS.md` (`InServiceOfX/AGENTS.md`) states repo-wide: **never
commit or push directly to `master`/`main`; the user merges manually.** All
work in this file's history was done on `feature/scaled-dot-product-attention`;
as of this writing `master` (and `origin/master`) have already been
fast-forwarded to include it — apparently by the user, outside any agent
session (no agent in this history ran `git merge` or `git push`). Start new
work on a fresh branch off `master`, not by committing to `master` directly,
even though it currently equals the feature branch tip.
