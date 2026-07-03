# Short-form video: "I benchmarked my hand-written CUDA FlashAttention against Google's XLA and NVIDIA's cuDNN"

Target: 45–75 s vertical video (X / TikTok / Shorts / Reels), slightly
technical audience. One idea per beat, one number on screen at a time.
Full data: `AttentionBenchmarkReport.md` (same directory).

## The three headline numbers (the only ones on screen)

1. **6.4 GB vs 0.1 GB** — what standard attention vs FlashAttention needs
   for one layer's attention scores at sequence length 4096 (batch 8, 12
   heads, fp32). Standard attention can't even run on a 12 GB card; the
   flash kernels don't notice.
2. **~2× causal speedup** — my kernel skips the masked half of the work
   (and schedules the long rows first). XLA's standard attention computes
   all of it and throws half away: its causal time = its non-causal time.
3. **16×** — how much faster NVIDIA's cuDNN flash attention is than my
   kernel at fp16. Same algorithm. The entire difference is one thing:
   **tensor cores**.

## Beat sheet

- **Hook (0–5 s):** "I wrote FlashAttention from scratch in CUDA — from the
  math, not the paper's pseudocode. Then I benchmarked it against Google's
  compiler and NVIDIA's hand-tuned library. Here's where I won, and where I
  got destroyed."
- **Beat 1 — the win (5–20 s):** show the 6.4 GB number. "Standard
  attention builds a giant N-by-N matrix. At 4K context that's 6.4 GB — my
  12 GB GPU literally cannot hold it next to the model. My kernel streams
  it in tiles and never builds that matrix at all. That's the entire
  FlashAttention idea, and it works: I run 4K context; standard attention
  can't." (Bonus flash: causal masking — I skip half the work, 2× faster;
  the compiler's version computes the masked half anyway.)
- **Beat 2 — the loss (20–40 s):** show 16×. "But cuDNN beat me by 16×.
  Why? My kernel computes dot products one multiply at a time on regular
  CUDA cores. cuDNN uses tensor cores — dedicated matrix hardware doing a
  16×16 matrix multiply per instruction. Right algorithm, wrong engine."
- **Beat 3 — the lesson (40–60 s):** "So GPU performance is two separate
  games: the *algorithm* decides how much memory you touch, the *hardware
  engine* decides how fast you crunch. FlashAttention-the-paper is famous
  for the first. Production kernels win because they do both. Next video:
  I put tensor cores inside my kernel."
- **CTA:** "Code, math derivation (2,900-line LaTeX), and full benchmark
  report linked below."

## Honesty guardrails (do not cut these corners)

- Say "my kernel at fp16" when citing the 16× — the fp32 gap vs XLA is
  ~6.6×, a different number; don't mix them.
- The memory-wall claim is for fp32 at batch 8 × 12 heads; it's a real
  measured skip (allocation would exceed the card), not a hypothetical.
- Don't claim the causal 2× is unique to hand-written CUDA — cuDNN skips
  masked tiles too; the contrast is specifically vs XLA's *standard* path.

## For a 5–10 min technical talk instead

Use the report's Section 3 table as the single money slide: four
implementations, GFLOP/s, % of peak, inner-product engine. Walk it
bottom-up: scalar FMA → cuBLAS tiles → fused batched matmul → fused MMA.
Then Section 4 as the closing slide: "the gap decomposes into exactly one
missing thing, and here is the plan for it."
