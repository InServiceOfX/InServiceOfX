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

The end state as of this file's writing: a complete multi-head attention
forward pass — dense QKV/output projections via cuBLASLt, IO-aware
FlashAttention core (two execution-mapping variants), causal masking,
batch/head parallelism — all unit-tested against independent CPU references,
plus the FlashAttention backward pass (currently single-head only — see
"What's NOT done" below).

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
4. **Git log** on `feature/scaled-dot-product-attention` (or `master`, which
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
          MultiHeadAttention/   # QKV/output projections + full MHA composition
          Softmax/              # standalone softmax kernels (design-space exploration)
        LLM/                   # legacy llm.c-style prototypes; currently empty after cleanup
        Benchmarks/            # AttentionIOBenchmark executable
        UnitTests/              # mirrors Source/ tree; gtest
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
| FlashAttention-2: Reducing Non-Matmul FLOPs | `Attention/flash_attention_warp_cooperative.h` (one **warp** per query row — the higher-performance variant; also the one `MultiHeadAttention` uses) |
| Gradients of Scaled Dot-Product Attention, Recomputation and the Logsumexp Statistic, The FlashAttention Backward Pass | `Attention/flash_attention_backward.h` |
| Parallelism and Work Partitioning | `flash_attention_forward.h`'s / `flash_attention_warp_cooperative.h`'s `gridDim.y` batch/head dimension; split-Q is literally what "one warp per row" *is* |

## Build & test

```bash
cd CUDALibraries/CuLLM/BuildGcc   # create with `mkdir BuildGcc` if absent
cmake ../Source
make Check -j4
./Check                             # full suite
./Check --gtest_filter='FlashAttention*'   # or any substring
```

As of this file's writing: **62 tests, 20 suites, all passing**, on an RTX
3070 Laptop (sm_86). `CMAKE_CUDA_ARCHITECTURES` is hardcoded to `75 86` in
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
  `qkv_projection.h` and `output_projection.h` use the identity
  `Out^⊤ = W^⊤X^⊤`, and the fact that a row-major `(r,c)` buffer read as
  column-major `(c,r)` **is** that matrix's transpose (same bytes, no copy).
  So the GEMM is called with operands swapped: `A := W`, `B := X`,
  `M_cublas = n`, `K_cublas = k`, `N_cublas = m`. Full derivation in
  `qkv_projection.h`'s header comment. This is verified against a
  from-scratch CPU reference in the tests, not just trusted from the
  derivation — if you touch this, re-verify the same way.
- **Fused QKV projection** (one GEMM against
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
- [x] Fused QKV projection (cuBLASLt GEMM + permute kernel)
- [x] Output projection (merge-heads kernel + cuBLASLt GEMM)
- [x] Full `multi_head_attention()` composition, end-to-end tested against an
      independent CPU MHA reference (causal + non-causal)
- [x] IO-complexity benchmark (`AttentionIOBenchmark`)
- [x] Tex extended with FlashAttention-2 material (non-matmul FLOPs
      proposition, causal tile-skipping proposition, parallelism/warp-
      partitioning remarks) — not just FlashAttention v1
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

Roughly in the order a next session would want to tackle them:

1. **Backward pass has no multi-head wiring.** `flash_attention_backward.h`
   exists and is tested (including batched-slice independence — see
   `flash_attention_backward_tests.cu`'s `BatchedMatchesPerSlice`), but
   there is no `multi_head_attention_backward()` tying it to
   `MultiHeadAttention/`'s projections (i.e., no `dW^Q`, `dW^K`, `dW^V`,
   `dW^O` — gradients w.r.t. the *projection weights*, not just Q/K/V).
   This needs: (a) backward through the two projection GEMMs (another
   cuBLASLt call per weight matrix, transposed appropriately — reuse the
   row-major trick above), (b) `merge_heads`/`split_qkv_heads`'s adjoints
   (these are pure permutations, so their adjoints are just the *inverse*
   permutation applied to the gradient — should be near-trivial given
   `merge_heads` is already `split_qkv_heads`'s documented inverse).
2. **cuBLASLt-projection vs. hand-written-GEMM benchmark never happened.**
   The original ask that led to `MultiHeadAttention/` was "should we use
   cuBLASLt or write our own GEMM, or benchmark both" — only the cuBLASLt
   path got built. A tiled shared-memory GEMM reference implementation (the
   pedagogical/correctness-baseline half of that plan) doesn't exist yet.
   Given cuBLASLt's tensor-core backing, expect it to win by a wide margin
   at realistic `d_model` sizes — but this is an assumption, not a measured
   result.
3. **`kHeadDim` not a multiple of 32** has no path through
   `MultiHeadAttention/` (it hard-requires `flash_attention_warp_cooperative`).
   Real transformer head dims are almost always 32/64/128 so this is low
   priority, but worth a `static_assert` with a clear message at the
   `multi_head_attention()` call site rather than a deep template error, if
   this becomes a real blocker.
4. **No vectorized (float4-style) loads anywhere.** `llm.c`'s
   `softmax_forward_kernel7`-derived `softmax_block_unrolled_fused.h`
   partially covers this via register-array unrolling
   (`kUnrollFactor`), but nothing in `Attention/` vectorizes its Q/K/V
   shared-memory loads. Worth profiling before investing here — the
   warp-cooperative kernel's speedup over thread-per-row (3–4x measured)
   suggests occupancy, not per-thread memory throughput, was the bottleneck
   being addressed so far.
5. **Causal tail-block load imbalance** (documented in the tex's Causal
   Tile Skipping remark) is unaddressed — no work-rebalancing scheme
   (e.g. scheduling long/unmasked row-blocks first) exists yet.
6. **MQA/GQA** (multi-query / grouped-query attention) is sketched as a
   tex remark (Parallelism and Work Partitioning section) but has zero code.
7. **No `__half`/`bfloat16` path exercised end-to-end.** `MathFunctions.h`
   and `AccumulationType.h` support `__half`, and individual kernels are
   templated on `T`, but no test instantiates the MHA pipeline at anything
   but `float`.

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
