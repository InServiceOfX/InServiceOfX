# FlashAttention: Mathematical Reference

## The Problem with Standard Attention

Standard scaled dot-product attention:

```
S = Q K^T / sqrt(d)        [N×N]   — pre-attention scores
P = softmax(S)              [N×N]   — attention weights
O = P V                     [N×d]   — output
```

where Q, K, V ∈ R^{N×d}, N = sequence length, d = head dimension.

**HBM traffic (dominant cost on GPU):**

| Operation | HBM reads | HBM writes |
|---|---|---|
| S = QK^T | 2Nd (Q,K) | N² (S) |
| P = softmax(S) | N² (S) | N² (P) |
| O = PV | N² + Nd (P,V) | Nd (O) |
| **Total** | **4Nd + 3N²** | **2N² + Nd** |

For N=4096, d=64: S and P alone are 4096²×4 bytes ≈ **64 MB** written and re-read.
The arithmetic is fast. The bottleneck is memory bandwidth, not FLOPs.

## Online Softmax

Key insight enabling tiling: softmax can be computed incrementally.

For a row x ∈ R^N, the standard softmax requires two passes:
1. Pass 1: find global max m = max_j x_j
2. Pass 2: compute exp(x_j - m) and normalize

The **online softmax** (Milakov & Gimelshein, 2018) fuses these into one pass
using a running max and running sum, correcting the sum as the max updates:

```
m^(0) = -inf,  l^(0) = 0

For each new element x_j:
  m^(j) = max(m^(j-1), x_j)
  l^(j) = exp(m^(j-1) - m^(j)) * l^(j-1) + exp(x_j - m^(j))

Final: softmax_i = exp(x_i - m^(N)) / l^(N)
```

When merging two partial results (block i, block j):
```
m_new = max(m_i, m_j)
l_new = exp(m_i - m_new) * l_i + exp(m_j - m_new) * l_j
```

This is the rescaling step in Algorithm 1 of the FlashAttention paper.

## FlashAttention Algorithm 1 (Dao et al., 2022)

**Goal:** compute O = softmax(QK^T / sqrt(d)) V without materializing the N×N
attention matrix to HBM.

**Tiling parameters:**
- Block row size Br (governs Q tiles)
- Block column size Bc (governs K, V tiles)
- Chosen so that tiles fit in SRAM: Br * d + 2 * Bc * d ≤ M (SRAM capacity)

**Algorithm:**

```
Divide Q into Tr = ceil(N/Br) row-blocks Q_1,...,Q_Tr, each Br × d
Divide K,V into Tc = ceil(N/Bc) col-blocks K_1,...,K_Tc, V_1,...,V_Tc, each Bc × d

For i = 1,...,Tr:
  Load Q_i from HBM → SRAM
  Initialize: O_i = 0 (Br×d),  l_i = 0 (Br),  m_i = -inf (Br)

  For j = 1,...,Tc:
    Load K_j, V_j from HBM → SRAM

    S_ij = Q_i K_j^T / sqrt(d)          [Br × Bc]  (in SRAM/registers)
    m_ij = rowmax(S_ij)                  [Br]
    P_ij = exp(S_ij - m_ij)             [Br × Bc]  (rowwise broadcast)
    l_ij = rowsum(P_ij)                  [Br]

    # Merge running stats with this tile's stats:
    m_i_new  = max(m_i, m_ij)                                [Br]
    l_i_new  = exp(m_i - m_i_new)*l_i + exp(m_ij - m_i_new)*l_ij   [Br]

    # Rescale accumulated output and add new contribution:
    O_i = diag(l_i_new)^{-1} * [
            diag(l_i) * exp(m_i - m_i_new) * O_i        (rescale old)
          + exp(m_ij - m_i_new) * P_ij V_j              (add new)
          ]

    m_i = m_i_new
    l_i = l_i_new

  Write O_i → HBM
```

The `diag(v)^{-1}` notation means element-wise division by vector v (one scalar per row).

## Why This Saves Memory Bandwidth

- S_ij and P_ij **never leave SRAM/registers** — they're computed and consumed within the inner loop.
- HBM writes: only O (N×d). No N×N matrices written.
- HBM reads: Q, K, V (each N×d). K and V are read Tr times, but Tr ≈ N/Br which is typically small.

**HBM traffic comparison (N=4096, d=64, Br=Bc=64):**

| Method | HBM traffic |
|---|---|
| Standard attention | ~4Nd + 5N² ≈ **320 MB** |
| FlashAttention | ~(Tr+1)(2Nd) ≈ **4 MB** |

## FlashAttention-2 Improvements (Dao, 2023)

1. **Fewer non-matmul FLOPs**: rescaling moved outside inner loop where possible.
2. **Parallelism over sequence dimension**: outer loop over Q tiles parallelized
   across thread blocks (not just batch × heads).
3. **Warp-level partitioning**: each warp handles a subset of columns of Q tile,
   reducing shared memory pressure.

## CUDA Kernel Design

```
Grid:  (Tr, B*H)        — Tr row blocks, B*H (batch × heads) independent streams
Block: (Br,  1  )        — Br threads, one per query row in Q_i tile
Shared memory layout:
  [Q_tile: Br × d]   [K_tile: Bc × d]   [V_tile: Bc × d]
Registers (per thread): O_row[d], m (scalar), l (scalar), S_row[Bc], P_row[Bc]
```

Template parameters `kBr`, `kBc`, `kHeadDim` are compile-time constants so
register arrays have known sizes and the compiler can unroll loops.

## References

### Foundational Papers (read in this order)

---

**[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., and Polosukhin, I. (2017).
*Attention Is All You Need.*
Advances in Neural Information Processing Systems (NeurIPS) 30.**

arXiv: https://arxiv.org/abs/1706.03762

The paper that introduced the Transformer architecture and the scaled
dot-product attention operator:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Multi-head attention applies this h times in parallel with learned
right-multiplication linear maps.
This is the algorithm FlashAttention accelerates — the math here is unchanged,
only the memory access pattern is optimised.

**What to read:** Section 3.2 (Scaled Dot-Product Attention) and Section 3.3
(Multi-Head Attention). The softmax over the full N×N matrix S = QK^T/√d is
what forces the O(N²) HBM traffic that FlashAttention eliminates.

---

**[2] Milakov, M. and Gimelshein, N. (2018).
*Online normalizer calculation for softmax.*
arXiv preprint.**

arXiv: https://arxiv.org/abs/1805.02867

Both authors are at NVIDIA. This is the **mathematical key** that makes
FlashAttention possible. Standard softmax requires two passes over the input
(one to find the global max, one to compute the normalizer). This paper shows
a single-pass, numerically stable algorithm using running accumulators:

```
# Standard two-pass:
m = max(x)               # pass 1 — HBM read
l = sum(exp(x - m))      # pass 2 — HBM read again
out_i = exp(x_i - m) / l

# Online one-pass:
m = -inf,  l = 0
for each new x_j:
    m_new = max(m, x_j)
    l = exp(m - m_new) * l + exp(x_j - m_new)
    m = m_new
out_i = exp(x_i - m) / l    # one final pass
```

The **merge formula** for two partial results (tiles A and B) is:

```
m_new = max(m_A, m_B)
l_new = exp(m_A - m_new) * l_A + exp(m_B - m_new) * l_B
```

This merge rule is exactly what appears in FlashAttention Algorithm 1's inner
loop. The paper reports up to 1.3× speedup for softmax alone and 5× for
softmax+TopK when fused as a single CUDA kernel.

**What to read:** the full paper — it is only 5 pages and is the most important
prerequisite for understanding FlashAttention at the kernel level.

---

**[3] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022).
*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*
Advances in Neural Information Processing Systems (NeurIPS) 35, 16344–16359.**

arXiv: https://arxiv.org/abs/2205.14135

The foundational paper. Key contributions:

- **IO complexity analysis:** proves that standard attention requires O(N²)
  HBM accesses, while FlashAttention requires O(N²/M) where M is SRAM size.
  For M ≫ d this is a significant reduction.

- **Tiling algorithm (Algorithm 1):** divides Q into row tiles (Br rows),
  K/V into column tiles (Bc cols). Inner loop loads one K/V tile, computes
  S_ij = Q_i K_j^T / √d in SRAM, applies online softmax to merge running
  accumulators, never writes S or P to HBM.

- **Backward pass (Algorithm 2):** recomputes attention weights on-the-fly
  from stored Q, K, V and output O during the backward pass (recomputation
  instead of storing the N×N matrix).

- **Block-sparse extension:** drop blocks of the attention matrix below a
  threshold, giving O(N√N) complexity with near-exact quality.

- **Measured results:** 15% end-to-end wall-clock speedup on BERT-large,
  3× on GPT-2, enables 64K-token sequences. RTX 3080 benchmarks included.

**What to read:** Section 2 (Background), Section 3.1 (Algorithm 1), Appendix B
(IO complexity proof), Appendix C (CUDA implementation details).

**The key identity (output rescaling at each tile merge):**
```
O_i ← diag(l_i_new)^{-1} [
        diag(l_i) * exp(m_i - m_i_new) * O_i      (rescale old)
      + exp(m_ij - m_i_new) * P_ij * V_j           (add new tile)
      ]
```
where O_i is maintained in a "pre-divided-by-l" normalized form.

---

**[4] Dao, T. (2023).
*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*
International Conference on Learning Representations (ICLR) 2024.**

arXiv: https://arxiv.org/abs/2307.08691

FlashAttention-1 achieved only 25–40% of theoretical GPU FLOPs utilization.
FA-2 pushes this to **50–73% on A100** (225 TFLOPs/s at 72% MFU during GPT
training). Three algorithmic changes:

1. **Fewer non-matmul FLOPs:** the rescaling `diag(l)^{-1}` is moved outside
   the inner Tc loop. The O accumulator is maintained unnormalized during the
   inner loop and normalized only once after the final K/V tile.

2. **Parallelism over sequence length:** FA-1 parallelized over batch × heads.
   FA-2 additionally parallelizes the outer Tr loop across thread blocks, so a
   single attention head can occupy multiple SMs.

3. **Warp-level work partitioning:** within each thread block, warps split
   along the column (K/V) dimension instead of the row (Q) dimension, reducing
   shared memory communication between warps.

**What to read:** Sections 2–3 (algorithm changes), Appendix B (warp diagrams
are invaluable for understanding SM-level occupancy).

---

**[5] Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T.
(2024).
*FlashAttention-3: Fast and Accurate Attention with Asynchrony and
Low-precision.*
arXiv preprint.**

arXiv: https://arxiv.org/abs/2407.08608

Targets NVIDIA Hopper architecture (H100, H200). FA-2 achieves only 35%
utilization on H100 because Hopper has async Tensor Core + TMA capabilities
that FA-2 does not use. FA-3 reaches **740 TFLOPs/s with FP16 (75%)** and
**~1.2 PFLOPs/s with FP8 (2.6× lower numerical error than baseline FP8)**.

Key techniques:

1. **Warp-specialization:** splits warps into "producer" (loads data via TMA)
   and "consumer" (runs Tensor Core matmuls) roles, overlapping memory and
   compute asynchronously via WGMMA + TMA.

2. **2-stage ping-pong pipelining:** while one warp-group computes S_ij V_j,
   the next tile K_{j+1}, V_{j+1} is already being loaded in shared memory.

3. **FP8 low-precision:** block quantization per tile; incoherent processing
   (random Hadamard transform on Q, K before quantizing) reduces outlier
   sensitivity.

**What to read:** Section 3 (core algorithm), Figure 2 (pipeline diagram),
Section 4 (FP8 incoherent processing). RTX 3060 = SM 8.6 (Ampere), so FA-3
Hopper-specific features won't run on our hardware, but the concepts are
important for understanding the roadmap.

---

### Supplementary Resources

**[6] Ye, Z. (2023).
*UW CSE 599M: Systems for ML — FlashAttention Lecture Notes.*
University of Washington.**

URL: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf

The clearest step-by-step algebraic derivation of the online softmax merge
rule and how it extends to the full O update. Recommended as a companion to
reading [3] for the first time.

---

**[7] Karpathy, A. (2024).
*llm.c — LLM training in simple C/CUDA.*
GitHub.*

URL: https://github.com/karpathy/llm.c

Reference implementation of the naive CUDA attention kernels. Useful for
understanding the baseline before FlashAttention optimisations are applied;
CuLLM's old `LLM/attention_forward.h` clone of `attention_query_key_kernel1`
has since been deleted after the Transformer attention path superseded it.

---

### Reading Roadmap

For working out the mathematics together:

1. Read [1] Section 3.2 → understand the operation being accelerated.
2. Read [2] in full (5 pages) → understand online softmax, the merge formula.
3. Read [3] Section 3.1 + Appendix B → derive Algorithm 1 step by step.
   Use [6] (UW notes) alongside to fill in algebra gaps.
4. Read [4] Sections 2–3 → understand the warp-level optimisations in FA-2.
5. [5] is optional unless targeting H100/H200.
