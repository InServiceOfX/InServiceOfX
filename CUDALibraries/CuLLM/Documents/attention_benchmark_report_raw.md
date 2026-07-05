<!-- JAX 0.10.2, devices [CudaDevice(id=0)] -->
<!-- Device: NVIDIA GeForce RTX 3060 | d = 64, warps/block = 4, batch*heads = 96 | 20 repeats -->

## Benchmark shape

- `B=8` batch elements, `H=12` attention heads.
- `B*H=96` independent attention slices run in parallel.
- `N` is context length / sequence length.
- `d_head=64` is the per-head Q/K/V dimension; the implied model width for this benchmark is `H*d_head=768`.
- Timings are attention-core forward only: no QKV projection, no output projection, no backward pass.

## Memory scale (not runtime allocation accounting)

This table shows the algorithmic memory pressure that matters for the FlashAttention argument. It is deterministic from the tensor shapes; it is not `nvidia-smi` allocator reservation.

| N | standard score tensor `(B,H,N,N)` fp32 | Q/K/V/O tensors fp32 |
|---|---:|---:|
| 256 | 0.03 GB | 0.03 GB |
| 512 | 0.10 GB | 0.05 GB |
| 1024 | 0.40 GB | 0.10 GB |
| 2048 | 1.61 GB | 0.20 GB |
| 4096 | 6.44 GB | 0.40 GB |

## Attention forward core, non-causal (B=8, H=12, d_head=64, float32, mean ms)

| N | CuLLM warp-coop CUDA | JAX/XLA standard (fused) | JAX built-in (xla) | JAX FA-2 tiled (lax loops) |
|---|---|---|---|---|
| 256 | 2.231 | 0.524 | 4.522 | 1.513 |
| 512 | 8.509 | 1.566 | 6.210 | 5.227 |
| 1024 | 34.041 | 5.407 | 18.204 | 19.161 |
| 2048 | 136.381 | 20.618 | 63.398 | 73.471 |
| 4096 | 550.694 | — | — | 289.474 |

## Attention forward core, causal (B=8, H=12, d_head=64, float32, mean ms)

| N | CuLLM warp-coop CUDA | JAX/XLA standard (fused) | JAX built-in (xla) | JAX FA-2 tiled (lax loops) |
|---|---|---|---|---|
| 256 | 1.278 | 0.573 | 6.640 | 1.551 |
| 512 | 4.556 | 1.585 | 7.961 | 5.222 |
| 1024 | 17.681 | 5.485 | 22.108 | 19.078 |
| 2048 | 69.811 | 20.580 | — | 73.623 |
| 4096 | 278.768 | — | — | 289.993 |

## float16 context (B=8, H=12, d_head=64, mean ms)

| N | causal | CuLLM warp-coop (fp16 I/O, fp32 accum) | cuDNN flash attention via JAX (fp16) |
|---|---|---|---|
| 256 | False | 2.409 | 2.481 |
| 512 | False | 9.489 | 2.139 |
| 1024 | False | 37.842 | 2.726 |
| 2048 | False | 151.032 | 6.497 |
| 4096 | False | 608.430 | 20.359 |
| 256 | True | 1.449 | 1.709 |
| 512 | True | 5.261 | 1.708 |
| 1024 | True | 20.531 | 2.429 |
| 2048 | True | 79.565 | 4.613 |
| 4096 | True | 313.961 | 12.883 |

## Accuracy: CuLLM CUDA vs JAX FA-2, identical inputs (float32)

| N | d_head | causal | max abs difference |
|---|---|---|---|
| 64 | 32 | False | 7.918e-05 |
| 100 | 32 | False | 7.393e-05 |
| 128 | 64 | False | 5.394e-05 |
| 150 | 64 | True | 3.249e-04 |

## Readout for presentation

- `d_head=64` means each head uses 64-dimensional Q/K/V vectors; it is not the full model width.
- At `N=2048`, JAX/XLA standard attention is fast in float32 (20.618 ms) because XLA lowers the matmuls to optimized library kernels, but that path materializes the `B x H x N x N` score tensor.
- The `—` entries at `N=4096` are intentional: the standard paths would allocate about 6.4 GB just for fp32 scores `8*12*4096*4096*4`, before other tensors.
- CuLLM's causal tile skipping is visible at `N=2048`: 136.381 ms non-causal -> 69.811 ms causal (1.95x). JAX/XLA standard computes then masks, so its causal/non-causal times stay nearly equal.
- The fp16 cuDNN row is the production fused-flash baseline. At `N=2048`, cuDNN is 23.25x faster than the scalar CuLLM fp16 path. Use `WarpAttentionBenchmark` for the WMMA and CuTe ladder that closes this gap.
- Accuracy is a cross-language sanity check: max abs error is at most `3.249e-04` across the sampled CUDA-vs-JAX FA-2 cases.
