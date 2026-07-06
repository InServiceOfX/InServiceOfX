# Presentation cut: CUDA C++ vs. JAX, economized

`CUDAvsJAXAttention.md` (same directory) is the full write-up and stays as
the first-draft source of truth — every number and claim here is pulled
from it, nothing new is asserted. This document is the prioritized,
timed cut for actually presenting: one long-form talk (~11-12 min) and two
shorts (each ≤2 min, each comparing at most 3 things). The concept —
hand-written CUDA vs. a compiler (JAX/XLA) vs. a vendor library (cuDNN),
measured honestly on the same hardware — is general and company-agnostic;
this is meant to be reused for any audience, not tied to one application.

Screenshots referenced below live in
`Data/Public/Jobs/CHAOSIndustries/` (outside this repo — produced assets,
not code; the subdirectory name is incidental, this material is general).
Filenames as of 2026-07-04:

- **Shot 1 (ladder)**: `2026-07-04_20-57WarpAttentionBenchmark.png`
  (headline, 20 repeats, N to 4096)
- **Shot 1b (stress, bonus)**: `2026-07-04_21-03WarpAttentionBenchmark-stress.png`
  (N to 8192 — use for the quadratic-scaling caveat below)
- **Shot 2 (tests)**: `2026-07-04_21-06CheckPassed.png` (89 C++ tests)
- **Shot 2b (tests, bonus)**: `2026-07-04_21-56test_jax_attention_reference.png`
  (5 Python pytest tests — mention "both sides tested" if time)
- **Shot 3 (JAX comparison)**: `2026-07-05_20-22benchmark.png` (superseding
  `2026-07-04_21-51benchmark.png` / `2026-07-04_22-33benchmark.png` — those
  two showed cuDNN at N=2048 non-causal as 6.5-7.1 ms, an unreproducible
  one-off; 4 independent reruns since, 3 back-to-back same-container plus
  this fresh screenshot, all land at 8.4-8.7 ms, matching the number this
  deck and script actually quote. Retired originals kept on disk but not
  to be used as presentation evidence.)
- **Shot 4 (memory wall, CUDA side)**: `2026-07-05_20-31AttentionMemoryReport.png`
  (live `cudaMalloc` failure at N=4096, plus the on-chip WMMA-vs-CuTe table)
- **Shot 4b (memory wall, JAX side)**: `2026-07-05_21-05jax_memory_report.png`
  (XLA's own compile-time memory plan — the independent cross-validation)
- Bonus/B-roll: `2026-07-04_21-09AttentionIOBenchmark.png` (exactness,
  max|diff| ~1e-8), `2026-07-04_21-10LinearMapGemmBenchmark.png`
  (cuBLASLt-vs-tiled-GEMM, measured library decision)

Exact on-screen text for every shot above (for captions/on-screen labels):
`AttentionBenchmarkScreenshotsTranscript.md`, same directory.

**6 widescreen infographic slides** (bar charts + real arithmetic, not
terminal screenshots): `CUDAvsJAXInfographicSlides.html`, same directory —
unlike the video production assets (which stay in `Data/Public/Generated/`,
produced artifact not code), this one is checked into the repo
deliberately: it's small, text-based, hand-edited, and actively revised
across sessions/machines, not a heavy binary export. Open it directly in a
browser (works from a bare `file://` path, fully self-contained) or via
`Artifact` in a Claude Code session. Slides: (1) the engine ladder, (2) the
full JAX/cuDNN comparison (all 5 implementations benchmarked), (3) the
memory wall, measured live — cudaMalloc failing at N=4096 plus XLA's
independent cross-validation and the WMMA-vs-CuTe on-chip table, (4) the
hand-written-FA-2-only comparison (CUTLASS/CuTe vs. WMMA vs. scalar CUDA
C++ vs. hand-written JAX), (5) arithmetic intensity (real FLOPs/bytes
calculations), and (6) throughput + the actual CUDA-vs-JAX answer. **Status
as of 2026-07-05: content complete (memory-wall slide added same day,
after independently reverifying both the CUDA and JAX measurements — see
`ReproducingTheBenchmarks.md`), not yet recorded** — see the handoff note
in `../AGENTS.md` for what's left and what's machine-local vs. portable.

---

## Long-form (~11-12 min)

Runtime grew from the original ~8-10 min estimate once code snippets were
added below (0:45-3:15 and 3:35-6:35 now each carry three snippets) —
flagging the change rather than quietly leaving the header stale.

**Title card, 0:00-0:15.** On-screen title: **"Hand-Written CUDA vs. JAX
— Settled with FlashAttention."** (Alternates, pick whichever fits the
platform: *"Is CUDA Still Worth Writing By Hand?"* — plainer, more
search-friendly; *"One Line of JAX Beat My CUDA Kernel. Here's What It
Took to Beat It Back."* — a number/stakes-led cold open if you want
something punchier for a Short.) Spoken, cold open over the title card:
"We're here to settle a question: are hand-written CUDA C++ kernels
actually more performant than Python — JAX, with its just-in-time
compiler? To find out, I picked the most consequential algorithm of this
era: the attention mechanism at the heart of the transformer, implemented
as FlashAttention."

**0:15-0:45 — Bridge.** "I implemented transformer attention at several
levels of the GPU software stack: scalar CUDA — straightforward, one
thread per element — up through warp-cooperative tensor cores, and
CUTLASS/CuTe. Then I benchmarked every version against JAX and XLA, and
NVIDIA's own production cuDNN library — same hardware, same inputs, same
shapes."

**0:45-3:15 — The engine ladder.** Show Shot 1. "Same algorithm every
time — this is FlashAttention, not standard attention: stream K/V tiles,
keep a running online softmax, never materialize the full score matrix.
Only the inner-product engine changes.

"One piece before the numbers: causal masking. A query at position i only
attends to keys at position j ≤ i — the future is masked out. My kernels
don't just zero those scores after computing them; they skip whole K/V
tiles that fall entirely in the future, so the wasted work never
happens:"

```cpp
// flash_attention_warp_cooperative.h:171-182
int number_of_tiles {
  (sequence_length + kTileColumns - 1) / kTileColumns};
if (kCausal)
{
  // Skip tiles past the block's last query row (uniform per block, so the
  // __syncthreads() below stay aligned).
  const int last_query_in_block {
    row_block * kWarpsPerBlock + kWarpsPerBlock - 1};
  const int last_needed_tile {last_query_in_block / kTileColumns};
  number_of_tiles = (last_needed_tile + 1 < number_of_tiles) ?
    last_needed_tile + 1 : number_of_tiles;
}
```

"That's worth a real ~2x on top of every engine tier below — not just
correctness, throughput.

"Now the part that actually changes: everything above — tiling, masking,
online softmax — stays identical across all three engines. Only this
inner loop differs. Scalar CUDA: one fused multiply-add per lane, per
element."

```cpp
// flash_attention_warp_cooperative.h:261-269 (scalar engine)
if (!masked)
{
  AccT dot {0};
  #pragma unroll
  for (int d {0}; d < kHeadDim; ++d)
  {
    dot += shared_queries[warp_rank][d] * shared_keys[lane][d];
  }
  score = dot * scale;
```

"151 ms at N=2048. First upgrade: WMMA. Same surrounding code — same
tiles, same mask, same online softmax — but that scalar loop is replaced
by one line: load two fragments, hand them to the tensor core."

```cpp
// flash_attention_tensor_core.h:206-211 (WMMA engine — replaces the scalar loop above)
wmma::load_matrix_sync(
  query_fragment, &shared_queries[warp_rank][0][f * kTile], kHeadDim);
wmma::load_matrix_sync(
  key_fragment, &shared_keys[0][f * kTile], kHeadDim);
wmma::mma_sync(
  score_fragment, query_fragment, key_fragment, score_fragment);
```

"52 ms. Same algorithm, 2.9x faster, purely from the engine. Second
upgrade: CUTLASS/CuTe. WMMA's fragments are opaque — a lane can't tell
which row its registers hold, which forces a round trip through shared
memory every tile to rescale. CuTe replaces that same spot with a call
against a coordinate tensor, so the accumulator stays in registers the
whole time:"

```cpp
// flash_attention_cute.h:273 (CuTe engine — replaces the same spot again)
gemm(tiled_mma, q_fragment, k_fragment, score_fragment);
```

"19 ms. Same math as the first version, 7.9x faster, purely from
changing what hardware does the arithmetic."

**3:15-3:40 — Tested First.** Show Shot 2. "We've written unit tests for
every kernel — each one checked against an independent CPU reference
before it's ever benchmarked. Eighty-nine tests passing here, another
hundred twenty-three in the shared numerics library underneath it."

**3:40-6:40 — Is hand-written CUDA even worth it?** Show Shot 3. "Now the
actual test: against JAX. Let's build up from the comparison closest to
my own code. This is FlashAttention-2 — my exact tiling algorithm — written
directly in JAX using `lax.fori_loop` instead of a CUDA kernel:"

```python
# jax_attention_reference.py:245-278 (excerpted -- the per-tile online-softmax
# merge; full function is jax_attention_reference.py:208-304)
def process_col_tile(col_tile, inner):
    max_i, denom_i, output_i = inner
    k_j = k_tiles[:, :, col_tile, :, :]
    v_j = v_tiles[:, :, col_tile, :, :]
    # ... tile mask, then:
    scores = jnp.einsum("bhid,bhjd->bhij", q_i, k_j) * scale
    masked_scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    tile_max = jnp.max(masked_scores, axis=-1)
    new_max = jnp.maximum(max_i, tile_max)
    # ... rescale old accumulator against new_max, then:
    tile_output = jnp.einsum("bhij,bhjd->bhid", weights, v_j)
    new_output = old_scale[..., None] * output_i + tile_output
    return new_max, new_denom, new_output
```

"Same tiles, same online softmax, same delayed normalization as my CUDA
kernel — but every tile's matmul here is dispatched to cuBLAS through
Python, through `lax.fori_loop`, one tile at a time. 74.4 ms — already
faster than my own scalar CUDA kernel's 151. Same algorithm, same tiling;
the difference is that even a naive per-tile JAX loop lands on cuBLAS's
tensor-core-backed GEMM, and my scalar kernel doesn't.

"Next: JAX's own built-in attention op, still routed through XLA rather
than cuDNN:"

```python
# jax_attention_reference.py:307-325 (full function -- compare to the
# ~100-line hand-rolled version above)
def jax_builtin_attention(q, k, v, causal=False, implementation=None):
    q_btnh = jnp.transpose(q, (0, 2, 1, 3))
    k_bsnh = jnp.transpose(k, (0, 2, 1, 3))
    v_bsnh = jnp.transpose(v, (0, 2, 1, 3))
    output = jax.nn.dot_product_attention(
        q_btnh, k_bsnh, v_bsnh,
        is_causal=causal,
        implementation=implementation,
    )
    return jnp.transpose(output, (0, 2, 1, 3))
```

"One call, `implementation=\"xla\"`. This doesn't know about tiling at
all — it's XLA choosing its own fusion. 66.1 ms, slightly ahead of my
hand-rolled JAX loop.

"And here's the version that actually wins on raw speed: three lines of
jitted matrix multiply and softmax, no tiling at all — the textbook
formula, not FlashAttention:"

```python
# jax_attention_reference.py:58-67 (full function)
def standard_attention(q, k, v, causal=False):
    head_dim = q.shape[-1]
    scale = jnp.asarray(1.0 / (head_dim ** 0.5), dtype=q.dtype)
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    if causal:
        mask = _causal_mask(q.shape[-2])
        scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhij,bhjd->bhid", weights, v)
```

"XLA fuses this into cuBLAS batched matmuls: 20.7 ms — beating my first
two kernels outright, and edged only barely by my CUTLASS/CuTe kernel at
19.0. But this version materializes the full N-by-N score matrix, which
is exactly the wall the next section is about. cuDNN's production flash
kernel, for reference: 8.4 ms, still 2.3x ahead of mine — down from 18x
when I started with scalar code."

**6:40-7:30 — The memory wall, measured live.** Show Shot 4. "That 6.4
gigabyte number isn't off a spec sheet — I measured it. At N=4096, my
flash kernels allocate 0.375 gigabytes, and it doesn't matter which
engine tier: scalar, WMMA, or CuTe all need exactly Q, K, V, and the
output, nothing else. The standard baseline's allocation call fails,
live, right here — six gigabytes short on a twelve gigabyte card. Now
show Shot 4b: on the JAX side, XLA's own compiler — completely
independently — computes standard attention's memory plan as exactly
twice batch-heads-times-N-squared. Three gigabytes at N=2048, matching
my measurement to the digit. Two different tools, two different
languages, the same number. And this same report is where the engine
tiers actually differ, not in HBM but on-chip: WMMA spends 48 kilobytes
of shared memory per block because its tensor-core fragments are
opaque; CuTe halves that by keeping the accumulator in registers
instead. That's the WMMA-to-CuTe upgrade, in hardware units."

**7:30-8:10 — The central lesson.** "GPU performance is two separate
games. The algorithm decides how many bytes you move. The hardware
engine decides how fast you crunch what's left. FlashAttention wins the
first game. Tensor cores win the second. cuDNN wins both in one kernel —
which is exactly why it's still ahead, and exactly what this project
measured the size of."

**8:10-9:40 — What this says about JAX.** "Prototype in JAX, always —
if your computation fits XLA's fusion patterns, you get tensor-core-class
performance for free, in Python, with autodiff included. Hand-write CUDA
only when you can name the specific thing XLA can't do for your case.
I found three: a memory wall XLA's standard formula can't route around
without owning the kernel; structured work-skipping — causal masking is
a real ~2x in a kernel that can skip masked tiles, but XLA's standard
path masks after computing, so it can't; and the last multiples — cuDNN's
remaining 2.3x lead decomposes into named, measurable techniques, not
magic."

**9:40-10:40 — Honest caveats.** "Three things that would have faked these
numbers, or oversold them, if I'd missed them. JAX's 'fp32' matmuls run
at TF32 precision by default on this hardware — an unqualified 'fp32 JAX
vs fp32 CUDA' comparison is quietly tensor-core-vs-scalar unless you
force full precision. JAX dispatch is asynchronous — timing without
waiting for the GPU to actually finish measures Python, not compute. And
the speedups don't keep scaling for free: show Shot 1b. Doubling the
context length from 4096 to 8192 costs about 4x, not 2x — exact
attention is still quadratic in compute. FlashAttention removes the
quadratic *memory*; it doesn't remove the quadratic *work*. Causal
masking stays near 2x faster throughout, because it skips tiles instead
of computing and discarding them." (Optional: also show Shot 5/6 as
quick proof frames — exactness column, the measured
cuBLASLt-vs-hand-written-GEMM decision.)

**10:40-11:25 — Scope + close.** "This is a forward-pass, fp16,
fixed-head-dim comparison on one consumer GPU — not a production claim.
What's next: bf16 in the tensor-core kernels, and a tensor-core backward
pass. Full derivation, all the code, and the complete report are linked
below."

---

## Short 1 — The engine ladder (≤2 min, 3 engines)

Compares: scalar CUDA vs. WMMA vs. CuTe — same algorithm, three engines.

Visual: Shot 1.

> My first FlashAttention kernel had the right algorithm — stream K/V
> tiles, online softmax, never build the full score matrix — but the
> dot products were scalar, one at a time, on regular CUDA cores. 151 ms
> at N=2048.
>
> First upgrade: WMMA. Same tiling, same masking, same softmax — but the
> two matrix products move onto tensor cores. 52 ms. Same algorithm,
> 3x faster, purely from the engine.
>
> Second upgrade: CUTLASS/CuTe. WMMA's fragments are opaque — a lane
> can't tell which row its registers belong to, which forces a
> round trip through shared memory every tile to rescale. CuTe gives me
> coordinates, so the accumulator stays in registers the whole time.
> 19 ms. Same math as the first version, 7.9x faster.
>
> Three implementations, one algorithm, one lesson: the hardware engine
> underneath your code can matter as much as the algorithm itself.

## Short 2 — Is hand-written CUDA even worth it? (≤2 min, 3 comparisons)

Compares: my CuTe kernel vs. JAX/XLA's fused standard attention vs.
cuDNN's production flash kernel.

Visual: Shot 3.

> I benchmarked my hand-written CUDA against Google's JAX compiler and
> NVIDIA's own cuDNN library, same hardware, same inputs.
>
> JAX/XLA's fused standard attention: 20.7 ms — three lines of Python,
> and it beats my first two kernel versions outright. JAX is a real
> compiler, not slow Python.
>
> My CUTLASS kernel: 19.0 ms, just ahead of it. cuDNN's production flash
> kernel: 8.4 ms — still 2.3x ahead of mine, down from 18x when I
> started with scalar code.
>
> But here's the actual reason to hand-write a kernel at all: XLA's
> standard attention has to build the full N-by-N score matrix in
> memory. At a 4K context, that's 6.4 gigabytes — more than my card can
> hold next to the model. My kernel streams it in tiles and never
> notices. That's the real argument for owning the kernel: not raw
> speed, memory you don't have.
