# Screenshot transcripts — exact text, for captions/editing

Plain-text transcription of the 9 screenshots in
`Data/Public/Jobs/CHAOSIndustries/` (outside this repo), so editing a
caption or on-screen label doesn't require re-reading the image. Every
number here was cross-checked against `CUDAvsJAXAttention.md`'s tables —
no discrepancies. Going forward, redirecting raw output to a `.txt` file
alongside each screenshot (`./WarpAttentionBenchmark | tee out.txt`) would
make this step unnecessary next time.

## `2026-07-04_20-57WarpAttentionBenchmark.png` — headline ladder

```
Fixed kernel shape: d = 64 per head, warps/block = 4. Runtime slices: batch*heads = 96.
Mean ms over 20 timed launches after 3 warmups.

Non-causal fp16 engine ladder
| N    | scalar fp16 | WMMA  | CuTe  | scalar/CuTe | WMMA/CuTe |
|------|------------:|------:|------:|------------:|----------:|
| 256  | 2.52        | 1.06  | 0.34  | 7.5x        | 3.2x      |
| 512  | 9.49        | 4.03  | 1.25  | 7.6x        | 3.2x      |
| 1024 | 37.86       | 13.74 | 4.82  | 7.9x        | 2.8x      |
| 2048 | 150.99      | 52.10 | 19.13 | 7.9x        | 2.7x      |
| 4096 | 606.83      | 229.35| 76.61 | 7.9x        | 3.0x      |

Causal tile skipping
| N    | fp32 non-causal | fp32 causal | speedup | CuTe non-causal | CuTe causal | speedup |
|------|-----------------:|------------:|--------:|----------------:|------------:|--------:|
| 256  | 2.23  | 1.28   | 1.74x | 0.34  | 0.24   | 1.42x |
| 512  | 8.51  | 4.56   | 1.87x | 1.25  | 0.74   | 1.69x |
| 1024 | 34.02 | 17.70  | 1.92x | 4.82  | 2.64   | 1.83x |
| 2048 | 136.39| 69.68  | 1.96x | 19.13 | 9.98   | 1.92x |
| 4096 | 549.57| 279.01 | 1.97x | 76.61 | 39.12  | 1.96x |

Headline at N=2048: scalar fp16 151.0 ms -> WMMA 52.1 ms -> CuTe 19.1 ms
(7.9x faster than scalar, 2.7x faster than WMMA).
```

## `2026-07-04_21-03WarpAttentionBenchmark-stress.png` — stress run (N to 8192)

Same tables, extended one more row each (5 repeats instead of 20, for
speed — values agree with the 20-repeat run to within run-to-run noise):

```
Mean ms over 5 timed launches after 3 warmups.

Non-causal: ... 4096 | 606.34 | 229.09 | 76.58 | 7.9x | 3.0x
             8192 | 2434.23 | 987.70 | 304.93 | 8.0x | 3.2x

Causal:     ... 4096 | 547.29 | 278.78 | 1.96x | 76.58 | 39.03 | 1.96x
             8192 | 2203.35 | 1113.05| 1.98x | 304.93| 153.90 | 1.98x

Stress note: N doubles 4096 -> 8192, but exact dense attention work is
quadratic. CuTe non-causal 76.6 -> 304.9 ms (3.98x); CuTe causal 39.0 ->
153.9 ms (3.94x). Causal stays near 2x faster because future tiles are
skipped.
```

## `2026-07-04_21-06CheckPassed.png` — tests, not vibes

```
[ RUN      ] FlashAttentionCuteTests.CuteLayoutAlgebraSmoke
[       OK ] FlashAttentionCuteTests.CuteLayoutAlgebraSmoke (0 ms)
... (MatchesCpuReferenceTileMultiple, RaggedSequenceLength,
     CausalMatchesCpuReference, CausalRaggedSequenceLength,
     MultiSliceIndependence — all OK)
[----------] 6 tests from FlashAttentionCuteTests (393 ms total)

[----------] Global test environment tear-down
[==========] 89 tests from 25 test suites ran. (1738 ms total)
[  PASSED  ] 89 tests.
root@b1fb8da03ca5:/InServiceOfX/CUDALibraries/CuLLM/BuildGcc#
```

## `2026-07-04_21-09AttentionIOBenchmark.png` — exactness column

```
root@b1fb8da03ca5:/InServiceOfX/CUDALibraries/CuLLM/BuildGcc# ./AttentionIOBenchmark
Device: NVIDIA GeForce RTX 3060 | d = 64, B_r = 64, B_c = 32 | 20 repeats

    n | standard ms | flash ms | warp ms | speedup | IO model | causal ms | max |diff|
------+-------------+----------+---------+---------+----------+-----------+-----------
  256 |      0.1105 |   0.2355 |  0.0431 |   2.56x |    2.00x |    0.0269 |  5.59e-08
  512 |      0.4029 |   0.4618 |  0.1206 |   3.34x |    2.00x |    0.0883 |  4.84e-08
 1024 |      1.5630 |   0.9240 |  0.3859 |   4.05x |    2.00x |    0.2495 |  3.17e-08
 2048 |      5.9254 |   1.8408 |  1.4333 |   4.13x |    2.00x |    0.6946 |  2.79e-08
 4096 |     23.6024 |   5.9366 |  5.3931 |   4.38x |    2.00x |    2.6484 |  2.07e-08
```

## `2026-07-04_21-10LinearMapGemmBenchmark.png` — measured library decision

```
root@b1fb8da03ca5:/InServiceOfX/CUDALibraries/CuLLM/BuildGcc# ./LinearMapGemmBenchmark
Linear-map GEMM: hand-written 32x32 tiled shared-memory kernel vs. cuBLASLt
(row-major Out = X W, 20 timed repeats after 3 warmups)

qkv d_model=256  B*T=2048 m=2048 k= 256 n= 768  | tiled 1.168 ms (689.7 GF/s) | cuBLASLt 0.119 ms (6747.6 GF/s) | speedup  9.8x | max|dt| 1.05e-05
qkv d_model=512  B*T=2048 m=2048 k= 512 n=1536  | tiled 4.836 ms (666.0 GF/s) | cuBLASLt 0.408 ms (7888.0 GF/s) | speedup 11.8x | max|dt| 0.00e+00
qkv d_model=768  B*T=2048 m=2048 k= 768 n=2304  | tiled 10.534 ms (688.0 GF/s)| cuBLASLt 0.894 ms (8110.8 GF/s)| speedup 11.8x | max|dt| 0.00e+00
qkv d_model=1024 B*T=2048 m=2048 k=1024 n=3072  | tiled 18.754 ms (687.1 GF/s)| cuBLASLt 1.609 ms (8009.7 GF/s)| speedup 11.7x | max|dt| 0.00e+00
out d_model=512  B*T=2048 m=2048 k= 512 n= 512  | tiled  1.520 ms (706.5 GF/s)| cuBLASLt 0.152 ms (7049.3 GF/s)| speedup 10.0x | max|dt| 1.81e-05
out d_model=1024 B*T=2048 m=2048 k=1024 n=1024  | tiled  6.366 ms (674.6 GF/s)| cuBLASLt 0.565 ms (7597.7 GF/s)| speedup 11.3x | max|dt| 0.00e+00
```

## `2026-07-04_21-56test_jax_attention_reference.png` — Python side tested too

```
root@b1fb8da03ca5:/InServiceOfX# python3 -m pytest CUDALibraries/CuLLM/Python/test_jax_attention_reference.py -q
Running 5 items in this shard
.....                                                                   [100%]
5 passed in 16.06s
```

## `2026-07-05_20-31AttentionMemoryReport.png` — the memory wall, live (CUDA side)

```
root@3cc9c2db4cac:/InServiceOfX/CUDALibraries/CuLLM/BuildGcc# ./AttentionMemoryReport
Device: NVIDIA GeForce RTX 3060 (sm_86) | VRAM 11.63 GiB total, 11.45 GiB free | B*H = 96, d = 64, fp32 element = 4 B

== 1. HBM working set, measured ==
N = 4096:
  flash (any rung): Q,K,V,O          requested   0.375 GiB -> measured device-memory delta   0.375 GiB (free: 11.452 -> 11.077 GiB)
  standard: Q,K,V,O + S + P          requested  12.375 GiB -> cudaMalloc FAILED (out of memory) after 6.375 GiB of it — this failure is the memory wall, measured

== 2. On-chip memory per kernel (where the ladder rungs actually differ) ==
  flash thread-per-row (fp32)               33024 B smem/block   254 regs/thread  block= 64  ->  3 blocks/SM ( 192 threads/SM)
  flash warp-cooperative (fp32)             17536 B smem/block    40 regs/thread  block=128  ->  5 blocks/SM ( 640 threads/SM)
  WMMA tensor-core (fp16)                   47872 B smem/block    41 regs/thread  block=128  ->  2 blocks/SM ( 256 threads/SM)
  CuTe/CUTLASS (fp16)                       23296 B smem/block   120 regs/thread  block=128  ->  4 blocks/SM ( 512 threads/SM)
  standard: attention_scores (fp32)           256 B smem/block    80 regs/thread  block=128  ->  6 blocks/SM ( 768 threads/SM)
```
(full output also includes N=1024/2048 HBM rows and the flash warp-cooperative fp16 on-chip row — omitted here, same story, see the PNG)

## `2026-07-05_21-05jax_memory_report.png` — the memory wall, live (JAX side, cross-validation)

```
root@3cc9c2db4cac:/InServiceOfX# PYTHONPATH=CUDALibraries/CuLLM/Python python3 CUDALibraries/CuLLM/Python/jax_memory_report.py
JAX 0.10.2 | device: NVIDIA GeForce RTX 3060 | B*H = 96, d = 64 | preallocate=false, allocator=platform

N = 2048 (fp32 args = 0.141 GiB):
  standard attention (fp32)          XLA plan: args  0.141  out  0.047  TEMP   3.000 GiB  ->  ran OK
  FlashAttention-1, lax tiles (fp32) XLA plan: args  0.141  out  0.047  TEMP   0.007 GiB  ->  ran OK
  FlashAttention-2, lax tiles (fp32) XLA plan: args  0.141  out  0.047  TEMP   0.009 GiB  ->  ran OK
  built-in attention, xla (fp32)     XLA plan: args  0.141  out  0.047  TEMP   3.000 GiB  ->  ran OK
  built-in attention, cudnn (fp16)   XLA plan: args  0.070  out  0.023  TEMP   0.070 GiB  ->  ran OK

N = 4096 (fp32 args = 0.281 GiB):
  standard attention (fp32)          lowering/compile failed: JaxRuntimeError: INTERNAL: Failed to get configs for: 2 out of 2 instructions. See logs for all failures. E
  FlashAttention-1, lax tiles (fp32) XLA plan: args  0.281  out  0.094  TEMP   0.007 GiB  ->  ran OK
  FlashAttention-2, lax tiles (fp32) XLA plan: args  0.281  out  0.094  TEMP   0.009 GiB  ->  ran OK
  built-in attention, xla (fp32)     lowering/compile failed: JaxRuntimeError: INTERNAL: Failed to get configs for: 2 out of 2 instructions. See logs for all failures. E
  built-in attention, cudnn (fp16)   XLA plan: args  0.141  out  0.047  TEMP   0.141 GiB  ->  ran OK
```
(full output also includes the N=1024 row, all consistent with the same law; captured clean after the `TF_CPP_MIN_LOG_LEVEL=3` fix — an earlier run without it buried the two N=4096 failures under several minutes of autotuner retry warnings)

## `2026-07-05_20-22benchmark.png` — the JAX/cuDNN comparison (current)

```
## Memory scale (not runtime allocation accounting)
| N    | standard score tensor (B,H,N,N) fp32 | Q/K/V/O tensors fp32 |
|------|--------------------------------------:|----------------------:|
| 256  | 0.03 GB | 0.03 GB |
| 512  | 0.10 GB | 0.05 GB |
| 1024 | 0.40 GB | 0.10 GB |
| 2048 | 1.61 GB | 0.20 GB |
| 4096 | 6.44 GB | 0.40 GB |

## Attention forward core, non-causal (B=8, H=12, d_head=64, float32, mean ms)
| N    | CuLLM warp-coop CUDA | JAX/XLA standard (fused) | JAX built-in (xla) | JAX FA-2 tiled (lax loops) |
|------|----------------------:|---------------------------:|---------------------:|-----------------------------:|
| 256  | 2.230   | 0.673  | 5.384  | 1.655  |
| 512  | 8.508   | 1.672  | 6.704  | 5.155  |
| 1024 | 33.975  | 5.530  | 19.602 | 19.111 |
| 2048 | 136.277 | 20.602 | 65.291 | 73.640 |
| 4096 | 549.294 | —      | —      | 288.516|

## float16 context (B=8, H=12, d_head=64, mean ms)
| N    | causal | CuLLM warp-coop (fp16 I/O, fp32 accum) | cuDNN flash attention via JAX (fp16) |
|------|-------:|----------------------------------------:|----------------------------------------:|
| 2048 | False  | 150.981 | 8.585  |
| 2048 | True   | 79.573  | 4.591  |
(full table: N = 256..4096, both causal states, in the report; this
screenshot's frame ends at the float16-context table, no accuracy table
in view this time)
```

**Provenance note (2026-07-05): this screenshot supersedes
`2026-07-04_21-51benchmark.png` and `2026-07-04_22-33benchmark.png`.**
Those two showed cuDNN at N=2048 non-causal as 6.505 / 7.097 ms — an
unreproducible one-off. Re-verified same day: 3 back-to-back reruns in
the same container session gave 8.712 / 8.650 / 8.676 ms, then this
fresh screenshot gave 8.585 ms — four independent readings within
8.4-8.7 ms, consistent with `AttentionBenchmarkReport.md`'s figure
(8.43 ms, measured 2026-07-03) that this deck and script have quoted
throughout as "8.4 ms." The two 07-04 screenshots stay on disk but are
retired from presentation use — most likely a one-off cuDNN
algorithm-selection quirk in that container session.
