# Helion Hackathon — luminous-kernels submissions

**Event:** PyTorch Helion Hackathon @ Cerebral Valley, March 14 2026  
**GPU:** B200 (Nebius)  
**Leaderboard:** https://github.com/gpu-mode/popcorn-cli/blob/main/docs/helion-hackathon.md

---

## Kernels

### 1. `fp8_quant_py/` — FP8 Per-Token-Group Quantization

**What it does:** Quantizes a `[T, H]` float32 tensor to FP8 E4M3 format with per-group scale factors. Used in DeepSeek-V3, Llama 3, Qwen3 W8A8 inference.

**Algorithm:**
```
reshape [T,H] → [N, gsz]    where N = T * (H/group_size)
for each group row:
  scale = clamp(max(|x|), eps) / 448.0
  x_q   = clamp(x / scale, -448.0, 448.0)
```

**Helion kernel design:**
- `hl.tile(nrows)`: one thread block per tile of groups
- `hl.specialize(ncols)`: bakes `group_size` as constant → warp-shuffle reduction of known depth
- `torch.amax(torch.abs(row), -1)`: compiles to `log2(group_size)` steps of `__shfl_down_sync`
- `scale[:, None]` broadcast: vectorized divide across entire group tile
- **0% inline Triton/ASM** — pure Helion DSL

**Optimizations over baseline:**
- Baseline redundantly computes abs+amax 3 times and averages them — corrected to single pass
- Proper per-shape configs instead of `block_sizes=[1], num_warps=1, num_stages=1`
- `hl.specialize` on group_size enables optimal reduction scheduling

---

### 2. `causal_conv1d_py/` — Causal Depthwise 1D Convolution

**What it does:** Mamba/Mamba-2 causal depthwise conv1d. Each channel `d` is convolved independently with a `W`-tap filter; output `t` depends only on input `[t-W+1 .. t]`.

**Algorithm:**
```
out[b, d, t] = bias[d] + Σ_{k=0}^{W-1} weight[d,k] * x[b, d, t-W+1+k]
(zero-pad W-1 on left first)
```

**Helion kernel design:**
- Pre-pad on host → kernel has **no boundary checks** at all
- `hl.tile([B, D, S], block_size=[1, None, None])`: tile over channels and sequence
- `hl.specialize(W)`: bake filter width as constant → **fully unrolled** inner loop (no loop overhead, W independent fma chains)
- Weight `w[rd, j]` invariant over `rs` → compiler **hoists** weight load (weight reuse, shared-mem analogue)
- `hl.load(x_pad, [bi, rd, rs.index + j])`: overlapping windows → Helion auto-allocates **shared memory halo** of size `S_tile + W - 1`, loaded cooperatively once
- **0% inline Triton/ASM** — pure Helion DSL

**Optimizations over baseline:**
- Baseline computes acc1+acc2+acc3 and divides by 3 — corrected to single accumulator
- `hl.specialize(W)` added for full loop unrolling (baseline had dynamic W)
- Proper per-shape configs with tuned D_tile × S_tile for B200

---

## Configs strategy

Configs in `SHAPE_CONFIGS` are **placeholders** that should be replaced with autotuned results.

To autotune on the Nebius B200:
```bash
# From reference-kernels/problems/helion/
python eval.py both fp8_quant_py/
python eval.py both causal_conv1d_py/
```

For Helion native autotuner (best quality, ~10 min/shape):
```bash
python hackathon/fp8_quant_py/submission.py --autotune --effort quick
python hackathon/causal_conv1d_py/submission.py --autotune --effort quick
```

Then paste the printed configs back into `SHAPE_CONFIGS`.

---

## Submission

```bash
# Install popcorn-cli
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn register discord
popcorn join <INVITE_CODE>
popcorn setup   # select fp8_quant or causal_conv1d, B200_Nebius
# copy our submission.py into the generated folder, then:
popcorn submit
```

---

## File structure

```
hackathon/
  fp8_quant_py/
    fp8_group_quant.py      ← our kernel (descriptively named)
  causal_conv1d_py/
    causal_conv1d.py        ← our kernel (descriptively named)
  README.md
  STYLE_CONTEXT.md
```

Supporting research / CUDA baseline implementations are in `03-fp8-quant/` and `04-causal-conv1d/` (parent dirs).

---

## Submission workflow (popcorn-cli expects `submission.py`)

popcorn-cli bundles whichever file is named `submission.py` in the project dir.
Our kernel files are descriptively named in this repo — copy to `submission.py`
at submit time:

```bash
# fp8_quant
cp hackathon/fp8_quant_py/fp8_group_quant.py  <popcorn_project>/fp8_quant_py/submission.py
popcorn submit

# causal_conv1d
cp hackathon/causal_conv1d_py/causal_conv1d.py  <popcorn_project>/causal_conv1d_py/submission.py
popcorn submit
```

Multiple iterations per kernel are fine — each gets its own commit here
(e.g. `fp8_group_quant_v2.py`), and only the last-copied one goes to the leaderboard.
Only your best submission counts.
