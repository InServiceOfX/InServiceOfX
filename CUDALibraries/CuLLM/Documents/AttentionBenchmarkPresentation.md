# Presentation cut: CUDA C++ vs. JAX, economized

`CUDAvsJAXAttention.md` (same directory) is the full write-up and stays as
the first-draft source of truth — every number and claim here is pulled
from it, nothing new is asserted. This document is the prioritized,
timed cut for actually presenting: one long-form talk (~8-10 min) and two
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
- Bonus/B-roll: `2026-07-04_21-09AttentionIOBenchmark.png` (exactness,
  max|diff| ~1e-8), `2026-07-04_21-10LinearMapGemmBenchmark.png`
  (cuBLASLt-vs-tiled-GEMM, measured library decision)

Exact on-screen text for every shot above (for captions/on-screen labels):
`AttentionBenchmarkScreenshotsTranscript.md`, same directory.

**5 widescreen infographic slides** (bar charts + real arithmetic, not
terminal screenshots): `CUDAvsJAXInfographicSlides.html`, same directory —
unlike the video production assets (which stay in `Data/Public/Generated/`,
produced artifact not code), this one is checked into the repo
deliberately: it's small, text-based, hand-edited, and actively revised
across sessions/machines, not a heavy binary export. Open it directly in a
browser (works from a bare `file://` path, fully self-contained) or via
`Artifact` in a Claude Code session. Slides: the engine ladder, the full
JAX/cuDNN comparison (all 5 implementations benchmarked), the
hand-written-FA-2-only comparison (CUTLASS/CuTe vs. WMMA vs. scalar CUDA
C++ vs. hand-written JAX), arithmetic intensity (real FLOPs/bytes
calculations), and throughput + the actual CUDA-vs-JAX answer. **Status as
of 2026-07-05: content complete, not yet recorded** — see the handoff note
in `../AGENTS.md` for what's left and what's machine-local vs. portable.

---

## Long-form (~8-10 min)

**0:00-0:30 — Hook.** "I implemented transformer attention at every level
of the GPU software stack — scalar CUDA I derived from the math myself, up
through tensor cores and NVIDIA's CUTLASS library — and benchmarked every
version against JAX/XLA and NVIDIA's production cuDNN, on the same
hardware, same inputs, same shapes. This is what actually separates a
compiler from hand-written kernels — measured, not guessed."

**0:30-2:00 — The engine ladder.** Show Shot 1. "Same algorithm every
time — stream K/V tiles, online softmax, never materialize the full
score matrix. Only the inner-product engine changes. Scalar CUDA cores:
151 ms at N=2048. Move the two matrix products onto tensor cores with
WMMA: 52 ms. Rebuild on CUTLASS/CuTe, with a register-resident
accumulator instead of round-tripping through shared memory: 19 ms. Same
math, 7.9x faster, purely from changing what hardware does the
arithmetic." (Second table on Shot 1, if time: causal masking — skip
future tiles instead of computing and discarding them, a consistent ~2x
on top of every engine tier.)

**2:00-2:20 — Tested, not vibes.** Show Shot 2. "Every kernel is checked
against an independent CPU reference before it's benchmarked — 89 tests
passing here, another 123 in the shared numerics library."

**2:20-4:20 — Is hand-written CUDA even worth it?** Show Shot 3. "Here's
the actual comparison against JAX. XLA's fused standard attention —
three lines of jitted einsum and softmax — hits 20.7 ms, beating my
first two kernels outright. JAX is not slow Python; it's a real
compiler, and it's good. My CUTLASS kernel: 19.0 ms, edging past it.
cuDNN's production flash kernel: 8.4 ms, still 2.3x ahead of mine — down
from 18x when I started with scalar code. But the memory table above
that is the real founding argument: XLA's standard attention has to
materialize the full N-by-N score matrix. At N=4096 that's 6.4 GB — more
than my 12 GB card can hold next to the model. My kernel streams tiles
and never notices. That's FlashAttention's actual argument: memory, not
raw speed."

**4:20-5:00 — The central lesson.** "GPU performance is two separate
games. The algorithm decides how many bytes you move. The hardware
engine decides how fast you crunch what's left. FlashAttention wins the
first game. Tensor cores win the second. cuDNN wins both in one kernel —
which is exactly why it's still ahead, and exactly what this project
measured the size of."

**5:00-6:30 — What this says about JAX.** "Prototype in JAX, always —
if your computation fits XLA's fusion patterns, you get tensor-core-class
performance for free, in Python, with autodiff included. Hand-write CUDA
only when you can name the specific thing XLA can't do for your case.
I found three: a memory wall XLA's standard formula can't route around
without owning the kernel; structured work-skipping — causal masking is
a real ~2x in a kernel that can skip masked tiles, but XLA's standard
path masks after computing, so it can't; and the last multiples — cuDNN's
remaining 2.3x lead decomposes into named, measurable techniques, not
magic."

**6:30-7:30 — Honest caveats.** "Three things that would have faked these
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

**7:15-8:00 — Scope + close.** "This is a forward-pass, fp16,
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
