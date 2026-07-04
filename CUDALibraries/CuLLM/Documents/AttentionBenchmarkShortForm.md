# Video & presentation material: "I benchmarked my hand-written CUDA FlashAttention against JAX/XLA and NVIDIA's cuDNN — then closed an 18× gap to 2.3×"

Status: **the full arc is complete and benchmarked** (2026-07-03). This is
no longer "here's where I lost, next video I'll fix it" — the fix already
happened, twice (WMMA, then CUTLASS/CuTe). Full data:
`AttentionBenchmarkReport.md` (same directory, Sections 3–6 have the ladder).
Interview framing: `Data/Private/applications/<current-application>/DemoOnePager.md`.

## The four headline numbers (pick 3–4 max per video; don't cram all of them)

1. **18× → 2.3×** — how far behind NVIDIA's production cuDNN flash attention
   my hand-written kernel started, and where it ended, after two engine
   upgrades (WMMA tensor cores, then CUTLASS/CuTe). At N=1024 the final gap
   is **1.14×** — essentially tied.
2. **6.4 GB vs 0 GB** — what standard (non-flash) attention needs to hold
   its N×N score matrix at 4K context (batch 8, 12 heads, fp32) vs what my
   flash kernel needs. Standard attention can't run on a 12 GB card at that
   length; the flash kernels don't notice.
3. **~2× causal speedup** — my kernel skips the masked half of the tiles
   (and schedules the expensive rows first). JAX/XLA's *standard* attention
   computes all of it and throws half away — its causal time equals its
   non-causal time.
4. **One instruction: `mma_sync`** — the single hardware primitive
   (tensor-core matrix-multiply-accumulate) responsible for most of the 18×
   → 2.3× improvement. Same algorithm before and after; only the
   inner-product engine changed.

## Format options (pick based on time available before the interview)

**A. One video, the complete arc (60–90 s).** Safest choice with days left
— tells a finished story with a real ending. Beat sheet below is written
for this.

**B. Three-part series (45–60 s each), if there's time to produce more than
one.** Natural cliffhangers, each ending on a specific named next step:
  1. "I wrote FlashAttention from scratch and benchmarked it against
     Google's compiler" → ends on the 18× loss to cuDNN, names the fix
     (tensor cores) as the hook for part 2.
  2. "I put tensor cores in my CUDA kernel" (WMMA) → 18× → ~6×, ends on
     "still leaving performance on the table — here's why" (opaque
     fragments forcing a shared-memory round trip).
  3. "I rebuilt it on NVIDIA's own kernel-building library" (CUTLASS/CuTe)
     → 6× → 2.3×, ends on the named remaining rungs (bigger tiles, LDSM,
     swizzles) — a real "and there's more" instead of a fake cliffhanger.

Given the timeline, **format A is the safer bet**; only attempt B if there's
genuinely time to shoot/edit three.

## Beat sheet (format A — full arc)

- **Hook (0–5 s):** "I wrote FlashAttention from scratch in CUDA — from the
  math, not the paper's pseudocode — then benchmarked it against Google's
  JAX compiler and NVIDIA's hand-tuned cuDNN library. I started 18× slower
  than NVIDIA. Here's how I closed it to 2.3×."
- **Beat 1 — the win nobody expects (5–20 s):** show the 6.4 GB number.
  "Standard attention builds a giant N-by-N matrix — at 4K context that's
  6.4 GB, more than my 12 GB GPU can hold next to the model. My kernel
  streams it in tiles and never builds that matrix. That's the whole
  FlashAttention idea, and it's why I can run contexts the 'reference'
  implementation can't." (Bonus if time: causal masking — I skip half the
  work, ~2× faster; the naive version computes the masked half anyway.)
- **Beat 2 — the honest loss (20–35 s):** "But on raw speed, NVIDIA's cuDNN
  beat my first version by 18×. Why? My kernel did every multiply one at a
  time on regular CUDA cores. cuDNN uses tensor cores — dedicated hardware
  that does an entire 16×16 matrix multiply in one instruction. Right
  algorithm, wrong engine."
- **Beat 3 — closing the gap (35–55 s):** "So I put tensor cores in my
  kernel — first with CUDA's WMMA API, which got me to 6× behind. Then I
  rebuilt it on CUTLASS, NVIDIA's own kernel-building library, using its
  register-level layout tools to keep data on-chip instead of round-tripping
  through shared memory. That got me to 2.3× — and at shorter sequences,
  basically tied with NVIDIA's own library."
- **Beat 4 — the lesson (55–70 s):** "GPU performance is two separate games:
  the *algorithm* decides how many bytes you move, the *hardware engine*
  decides how fast you crunch what's left. I now know exactly which
  instruction bought each speedup — not 'it got faster,' but which specific
  hardware feature and why."
- **CTA:** "Full derivation (2,900-line math writeup), all the CUDA, and the
  complete benchmark report linked below."

## Honesty guardrails (do not cut these corners)

- The 18×/2.3×/1.14× numbers are all **fp16** (the tensor-core kernels are
  fp16-only today; scalar fp32 vs cuDNN fp16 is a different, larger gap —
  don't conflate them on screen).
- The memory-wall claim (6.4 GB) is for fp32 at batch 8 × 12 heads; it's a
  real measured allocation failure, not a hypothetical.
- Don't claim the causal ~2× is unique to hand-written CUDA — cuDNN skips
  masked tiles too; the contrast is specifically vs JAX/XLA's *standard*
  (non-flash) path.
- "Basically tied at N=1024" is honest (1.14×) — don't round it to "tied."
  The gap grows with sequence length (up to ~3.5× at N=4096); say so if
  asked.
- If asked "is this production-ready," answer no — forward-only at this
  perf tier, fp16-only, fixed head-dim/warp-count. That's the honest scope,
  and naming it precisely reads as more credible than overclaiming.

## For a 5–10 min technical talk or interview walkthrough instead

Use the benchmark report's Section 3 table (algorithm-vs-engine GFLOP/s
breakdown) as the first money slide, then Sections 5–6's ladder tables as
the "and here's what I did about it" follow-up — walk it bottom to top:
scalar FMA → WMMA fragments → CuTe register-resident accumulator + cp.async
pipelining. Close on the report's named remaining rungs (larger K/V tiles
per stage, LDSM shared→register copies, swizzled layouts) to show you know
precisely what's left, not just that something is left.
