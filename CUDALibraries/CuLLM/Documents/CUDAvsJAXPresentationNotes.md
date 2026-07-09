# Presentation notes — CUDA vs. JAX (not a script)

For someone who already knows this material and has given it before —
reminders of what to hit per slide, not lines to read. Slides:
`CUDAvsJAXCodeSlides.html`, same directory. Cross-referenced against
`../../../Galvatron/Documents/SpX/CHAOSIndustries_20260707_Postmortem.md`
(a real onsite debrief, not hypothetical prep) — the "if asked" section
below is largely that document's already-rehearsed answers, condensed.

## Slide 1 — Title

- One breath: 8 implementations, same math, same card, same shapes —
  real code, real measured numbers, not a spec sheet.

## Slide 2 — CUDA C++: baseline, then three engines

- Standard row: this is the thing Flash exists to avoid — say
  "materializes S" out loud, point at the line that writes `scores[...]`.
- Scalar → WMMA → CuTe: same tiling, same mask, same online softmax the
  whole way down — only the inner-product instruction changes each step.
  Don't let it sound like three different algorithms.
- WMMA→CuTe jump: fragments are opaque in WMMA (a lane can't address its
  own row), forces a shared-memory round trip; CuTe's coordinate tensor
  fixes that specifically — worth the "why," it's the most technical
  beat on this slide.

## Slide 3 — JAX: four ways to write the same thing

- **The reminder you flagged**: before or right after this slide, say
  the caching line. This is Q1 from the postmortem, already rehearsed —
  use it close to verbatim, it's short and it's exactly what the room
  wanted last time:

  > "It compiles once per input shape and caches the result — every
  > later call with the same shape just replays it, which is why we
  > warm up before we time."

  If pushed further: 3 warmups, 20 timed launches, both sides
  (`jax_attention_reference.py:328-346` and
  `warp_attention_benchmark.cu:77-106`) — same structure, an untimed
  warmup loop then a *blocking* timed loop (`block_until_ready` /
  `cudaEventSynchronize`), so the compile cost and dispatch-vs-completion
  gap are both excluded on purpose, not by accident.
- Last two rows (built-in xla vs. cudnn): let the "one string changes"
  beat land before saying the number — that's the slide's actual punch
  line, don't rush past it into the 8.4ms.

## Slide 4 — Results

- Point at the color legend before reading rows — blue = mine, don't
  let cuDNN's fast number get misread as "my CUDA."
- Close on the algorithm-vs-engine split, not the ranking: Flash's memory
  win travels to either language: tensor cores don't. That sentence is
  the actual thesis of the whole deck; the ranking is evidence for it.

---

## If asked — condensed from the 2026-07-07 postmortem

Lead with the mental model before reaching for an implementation trick —
that ordering is itself the thing being graded, per the postmortem's own
closing note.

**"Does JAX recompile every call?"** No — traces once per unique
shape/dtype combo into a Jaxpr, lowers to XLA HLO, compiles to GPU
kernels, caches the executable. Only a shape/dtype/static-arg change
forces a recompile. If they want the pipeline named: Python function →
trace (abstract tracers, no real values) → Jaxpr → HLO → XLA passes →
GPU kernels.

**"How do you find bottlenecks, not just wall-clock?"** Layered, name the
tools even without deep hands-on reps — that vocabulary is what's graded:
1. Per-region `cudaEvent`/timed-loop instrumentation (what this project
   actually does — every number in the deck comes from this, not a
   stopwatch around `main()`).
2. **Nsight Systems (`nsys`)** — system timeline, kernels vs. memcpys vs.
   gaps; the tool that actually answers "memory transfer or compute?"
3. **Nsight Compute (`ncu`)** — per-kernel occupancy, compute vs. memory
   throughput %, roofline classification, warp stall reasons.
4. `compute-sanitizer` for correctness (races, illegal access) — worth
   naming even though it's not a perf tool, same axis being graded.
5. The strongest available answer, and the one not used last time: **a
   custom tool already built for this** — `AttentionMemoryReport`,
   `cudaMemGetInfo` around real `cudaMalloc` calls, caught the flash
   kernels' flat 0.375 GiB footprint and the standard path's live
   allocation failure at 12.375 GiB requested. A working measured tool
   beats a described workflow.

**"What happens going from KB to MB to GB?"** The general, algorithm-
agnostic version — say this one without ever needing to mention
attention specifically:
- Two separate costs: $T_{compute}=F/P_{compute}$,
  $T_{memory}=B/P_{mem}$; wall-clock is $\max$ of the two, not their sum.
- Arithmetic intensity $I=F/B$; roofline:
  achieved FLOP/s $=\min(P_{compute}, I\cdot P_{mem})$; ridge point
  $I^{*}=P_{compute}/P_{mem}$.
- **Elementwise ops**: $I$ is constant in $N$ (both $F$ and $B$ scale the
  same way) — memory-bound at 1 KB, still memory-bound at 1 GB. Scaling
  never changes the regime.
- **Well-tiled matmul**: $I(N)\propto N$ — grows with problem size, so
  small matmuls are memory-bound and large ones become compute-bound.
  On this card, ridge point ≈ 35 FLOP/byte → crossover around N≈212.
  Intensity is a property of *how the kernel is written*, not the
  algorithm in the abstract — a naive re-read-from-HBM version of the
  same matmul doesn't get this for free.
- **KB/MB/GB skeleton**: KB is overhead-bound (launch latency, empty
  SMs, not measuring the kernel at all); MB is the real roofline
  crossover; GB is the capacity wall — either an intermediate no longer
  fits on-chip, or the input itself has to stream over PCIe, and
  $P_{mem}$ drops ~15-18× the moment that happens (fix becomes overlap
  transfer with compute, not write a faster kernel).

**Warp divergence, if it comes up**: SIMT means a warp serializes
divergent branches — masks the other lanes, runs each path in turn,
idle lanes still occupy scheduler slots, so it's a direct throughput
loss. Directly relevant here: causal masking in every kernel on this
deck is applied per-tile (skip or don't-skip a whole tile), specifically
so a warp never diverges thread-by-thread on the mask.

**Not this deck's content but keep in back pocket if asked**:
multiprocessing vs. multithreading / GIL — numpy and CUDA driver calls
release the GIL during the actual C-level work, so threading *can*
parallelize GIL-releasing native code even though pure-Python can't.
Full answer in the postmortem doc if it comes up.
