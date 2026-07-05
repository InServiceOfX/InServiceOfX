# Reproducing the CUDA-vs-JAX benchmarks (and the screenshot shot list)

Two purposes, one document: (1) anyone can rerun every number in
`CUDAvsJAXAttention.md` / `AttentionBenchmarkReport.md` from scratch;
(2) a prioritized shot list for capturing presentation evidence on the
benchmark machine (RTX 3060 desktop). Screenshots of *live terminal
output* are the credibility currency here — a rendered markdown table
says "trust me," a terminal with the device name in frame says "watch."

## The executables (what exists, where, what each proves)

CUDA side — all in `CuLLM/BuildGcc/` (host) and `CuLLM/BuildDocker/`
(container); same CMake tree builds both:

| Executable | What it does | What it proves |
|---|---|---|
| `Check` | 89 gtest unit tests, every kernel vs. independent CPU references, gradient checks, device guard | correctness discipline — the kernels are *tested*, not just fast |
| `WarpAttentionBenchmark` | the ladder: scalar fp32/fp16, WMMA, CuTe rows at B·H=96, d=64, N=256–4096, causal both (CSV lines plus readable speedup tables) | the 18×→2.3× engine story, one screen |
| `AttentionIOBenchmark` | standard vs. flash (thread) vs. flash (warp) vs. causal sweep, with a max-diff exactness column | the IO/algorithm story + bit-level honesty column |
| `LinearMapGemmBenchmark` | hand-written tiled GEMM vs. cuBLASLt at linear-map shapes, correctness delta gated before timing | "measure before deciding" — cuBLASLt won 10–12×, so we kept it |
| `AttentionReferenceDump` | dumps kernel output for deterministic inputs (consumed by the Python comparison) | the cross-language accuracy bridge |
| `AttentionMemoryReport` | (1) HBM working set measured via cudaMemGetInfo around real cudaMallocs — flash Q/K/V/O vs. the standard baseline's two B·H·N² workspaces, with the N=4096 allocation *failing live* on the 12 GB card; (2) per-kernel on-chip resources (cudaFuncGetAttributes): smem/block, regs/thread, occupancy | memory measured, not asserted — and the on-chip table is where the ladder rungs actually differ (HBM is identical across rungs by design) |

MoreCUDA side — `MoreCUDA/BuildGcc/Check`: 123 more unit tests (math
functions, cuBLASLt wrappers, memory utilities the kernels build on).

JAX side — `CuLLM/Python/`, run inside the `propulsion-with-cuda:26.02-py3`
container (it has JAX 0.10.2 + cuDNN; the repo mounts in, no image
changes needed):

| Script | What it does |
|---|---|
| `test_jax_attention_reference.py` | pytest, 5 tests: online softmax, tiling, causal, multi-head layout, built-in wrapper |
| `benchmark_report.py` | the full comparison: JAX/XLA standard, built-in, FA-2 lax-loop, cuDNN — plus CuLLM CSV ingestion and the CuLLM-vs-JAX accuracy table; emits the report tables as markdown |
| `compare_cullm_jax_attention.py` | smaller accuracy + timing comparison (superseded by `benchmark_report.py` for the full matrix) |
| `jax_memory_report.py` | XLA's own memory accounting (`compiled.memory_analysis()`) for standard / FA-1 / FA-2 / built-in / cuDNN, plus live execution outcome; handles the preallocation gotcha for you | the JAX-side mirror of `AttentionMemoryReport` — same memory law from an independent measurement |

## Rerunning everything

### Host (CUDA side; device 0 *is* the RTX 3060 — CUDA orders fastest-first)

```bash
cd <repo>/CUDALibraries/CuLLM/BuildGcc
cmake ../Source && cmake --build . --target Check WarpAttentionBenchmark \
    AttentionIOBenchmark LinearMapGemmBenchmark -j$(nproc)

./Check                      # 89 tests; first FlashAttentionTensorCore test prints the GPU name
./WarpAttentionBenchmark     # the ladder CSV (~2 min: N=4096 rows dominate)
./AttentionIOBenchmark       # standard-vs-flash sweep (~1-2 min)
./LinearMapGemmBenchmark     # tiled GEMM vs cuBLASLt (~1 min)
./AttentionMemoryReport      # memory: measured HBM working sets + live OOM at N=4096 + on-chip/occupancy table (seconds)

cd ../../MoreCUDA/BuildGcc && cmake ../Source && cmake --build . --target Check -j$(nproc)
./Check                      # 123 tests
```

### Container (JAX side + the combined report; GPU 1 in nvidia-smi order = the 3060)

```bash
docker run --rm --gpus '"device=1"' \
  -v <repo-absolute-path>:/InServiceOfX propulsion-with-cuda:26.02-py3 bash

# inside the container:
cd /InServiceOfX/CUDALibraries/CuLLM/BuildDocker
cmake ../Source && make WarpAttentionBenchmark AttentionReferenceDump -j$(nproc)

cd /InServiceOfX
python3 -m pytest CUDALibraries/CuLLM/Python/test_jax_attention_reference.py -q   # "5 passed"
PYTHONPATH=CUDALibraries/CuLLM/Python python3 \
  CUDALibraries/CuLLM/Python/benchmark_report.py \
  --build-dir CUDALibraries/CuLLM/BuildDocker    # several minutes: jit compiles + full sweep

PYTHONPATH=CUDALibraries/CuLLM/Python python3 \
  CUDALibraries/CuLLM/Python/jax_memory_report.py   # JAX-side memory (~2 min)
```

Notes: the XLA-standard rows deliberately skip N=4096 (the N² score
buffer would be 6.4 GB — that skip *is* a result, not a gap). JAX
timings include dispatch overhead; read N=256 rows with that in mind.

### WarpAttentionBenchmark flags

`WarpAttentionBenchmark` keeps the stable `CSV dtype,n,batch_heads,causal,ms`
lines consumed by `benchmark_report.py`, then prints two presentation-ready
tables: the non-causal fp16 engine ladder (scalar fp16 → WMMA → CuTe) and
causal tile-skipping speedups.

Useful variants:

```bash
./WarpAttentionBenchmark --csv-only
./WarpAttentionBenchmark --repeats 5 --warmups 2
./WarpAttentionBenchmark --batch-heads 192 --repeats 5
./WarpAttentionBenchmark --stress
```

What is configurable here: `batch*heads`, warmups/repeats, and the optional
stress sweep that adds `N=8192`. What is deliberately fixed by the compiled
kernels: `d=64` and `warps/block=4`. Here `d` is the per-head query/key/value
dimension, not model width; `d=64` is both a common transformer value and the
current CuTe kernel's supported shape. `warps/block=4` is the CuTe copy and
tiling shape. To benchmark `d=128` honestly, add a separate templated
benchmark instantiation and accept that the current CuTe row will drop out
until that kernel supports the wider head dimension.

In these tables `N` is the context length / sequence length. The stress run is
for the video caveat: doubling `N` from 4096 to 8192 makes exact dense
attention do about 4x the dot-product work, even though FlashAttention avoids
the quadratic score-matrix allocation. The benchmark prints this explicitly as
a `Stress note:` line; include that line if the screenshot is meant to teach
"linear memory, still quadratic compute."

## The shot list (prioritized; each shot = one claim made visible)

Composition rules for all shots: dark terminal, font large enough to read
in a slide/phone screenshot, and **include the command line you typed at
the top of the frame** — output with its provenance visible is the whole
point. Where two terminals are named, use a side-by-side split.

1. **The ladder, live.** `./WarpAttentionBenchmark` full output — the
   `Device: NVIDIA GeForce RTX 3060` header line **must be in frame**
   with either the `float32 / float16 / wmma16 / cute` CSV rows or the
   readable "Non-causal fp16 engine ladder" summary below it. This single
   frame is the scalar → WMMA → CuTe story with hardware provenance.
   For the `--stress` version, include the `Stress note:` line if possible:
   it makes the 4096→8192, 4x-runtime caveat explicit without requiring the
   viewer to compare rows manually.
2. **Tests, not vibes.** `./Check` tail: `[ PASSED ] 89 tests.` — and a
   second shot (or the same scrollback) catching
   `Running on CUDA device 0: NVIDIA GeForce RTX 3060 (sm_86)` from the
   tensor-core suite's device guard. For a hiring audience this shot
   outranks every benchmark: it says the numbers sit on tested code.
3. **The JAX side is real, same machine.** `benchmark_report.py` output
   inside the container: the `JAX 0.10.2, devices [CudaDevice(id=0)]`
   header plus the float32 comparison table, and the accuracy table
   (`max |CuLLM−JAX|` ~1e-4/1e-5 rows) in a second shot. This is the
   direct evidence the comparison isn't two disconnected experiments.
4. **GPU actually working.** A second terminal running
   `watch -n 0.5 nvidia-smi` while `WarpAttentionBenchmark` runs —
   utilization pegged, the RTX 3060 row visible. Cheap shot, disarms
   "did this really run" instantly, and looks good in motion if you
   screen-record instead of screenshot.
5. **The exactness column.** `./AttentionIOBenchmark` output — the sweep
   with its `max |diff|` column (~1e-8) against the standard-attention
   baseline. The claim it makes visible: flash attention here is *exact*,
   not an approximation.
6. **Measure before deciding.** `./LinearMapGemmBenchmark` output — the
   tiled-GEMM-vs-cuBLASLt table with its `max|dt|` correctness gate and
   the ~10–12× column. The claim: library-vs-hand-written was a
   *measured* decision, both directions (kept cuBLASLt for GEMMs, wrote
   kernels where fusion required it).
7. **Memory, measured — two claims in one frame.** `./AttentionMemoryReport`
   output: the N=4096 block where flash shows `0.375 GiB` measured while
   the standard path prints `cudaMalloc FAILED (out of memory)` — the
   memory wall demonstrated, not computed. Scroll to the on-chip table in
   the same shot if it fits: thread-per-row's 254 regs/thread → 192
   threads/SM (why it was slow, in one number), WMMA's 47,872 B
   smem/block → 2 blocks/SM vs. CuTe's 23,296 B + register-resident
   accumulator → 4 blocks/SM (the WMMA→CuTe design choice, visible in
   hardware terms). For a GPU-engineering audience this shot rivals shot 1.
   **Companion frame (container): `jax_memory_report.py`** — XLA's own
   plan showing standard attention's TEMP at exactly 2·B·H·N² (3.000 GiB
   at N=2048, matching the CUDA-side measured workspaces to the digit),
   FA-1/FA-2 lax temps flat at ~7–9 MiB across all N, and the N=4096
   standard path failing at *compile* time. Two independent measurement
   methods agreeing on the same law is the strongest single memory claim
   in the whole project.
8. **The second test suite.** MoreCUDA `./Check` tail: `[ PASSED ] 123
   tests.` Optional but cheap — 212 total tests across the two suites is
   a better sentence than 89.
9. **pytest, 5 passed.** The JAX reference implementations are tested
   too — one small shot inside the container. Optional; include if the
   audience is Python-fluent.

For the polished slide/thumbnail version of the result table, use the
rendered markdown table from `CUDAvsJAXAttention.md` — but pair it with
shot 1 somewhere nearby: rendered table for legibility, terminal for
proof.

## What a GPU-engineering employer is looking for in these shots

Not the speedup itself — evidence of *practice*: tests passing before
benchmarks (shots 2, 7, 8), the device name and utilization in frame
(shots 1, 4), an exactness column sitting next to every performance claim
(shots 3, 5, 6), and honest methodology notes (the N=4096 skip, dispatch
overhead, TF32 defaults — all documented in the report rather than
hidden). The shot list is ordered so that if only three screenshots
survive into a presentation, shots 1–3 carry the full argument:
performance ladder with provenance, tested code, and a same-machine
cross-framework comparison.
