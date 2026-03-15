# Style & Mathematical Context — Ernest's Coding + Math Voice

This file records what I learned from reading InServiceOfX/CUDALibraries,
CompPhys, and cs344 so I can write kernels and explanations in Ernest's style.

---

## Ernest's C++/CUDA Coding Style

### Naming conventions
- Template parameters in ALL_CAPS: `THREADS_PER_BLOCK`, `N_b`, `FPType`
- Local variables with descriptive names, often with subscripts: `lane_id`, `warp_id`, `sequence_position`
- `d_` prefix for device (GPU) pointers/arrays
- C++ `{}` initializer syntax everywhere (not `=`): `const std::size_t i {blockIdx.x * blockDim.x + threadIdx.x};`
- `constexpr` over `#define` wherever possible
- `std::size_t` for index/size types

### Namespace structure
- Deeply nested namespaces matching directory structure: `Operations::`, `ParallelProcessing::`, `LLM::AttentionForward::`, `GoogleUnitTests::Operations::`
- Templates with `typename FPType` (not `T` alone) for floating-point type parameters

### Unit test style (GoogleTest)
- Test file mirrors source file: `Convolution1D.h` → `Convolution1D_tests.cu`
- Tests use `using` declarations at top to import everything needed
- Test names are expressive: `Convolve1DConvolves`, `WarpReduceMaxTests`, `BasicFunctionality`
- Specific expected vectors hardcoded (not just shape-checked): `const auto host_expected = vector<float>{5, 8, 11, 14, 5, 0};`
- Tests cover edge cases: negative values, max at various positions, multi-warp scenarios
- Comments in tests explain the mathematical meaning: `// Use the sum formula for the first n natural numbers, S = n(n + 1) / 2`

### Kernel/algorithm documentation style
- `@details` in doxygen-style comments with mathematical formula inline
- Explicit index arithmetic spelled out: `i_a = threadIdx.x - N_b = 0, 1, ..., THREADS_PER_BLOCK - N_b - 1`
- "Intentionally show the arithmetic because..." — pedagogical clarity over brevity
- ASCII diagrams of memory layout: `0 1 2 3 | 4 5 |`
- References to `__shfl_down_sync`, `__shfl_xor_sync` with NVIDIA docs links

### Warp reduction idiom (Ernest's canonical form)
```cpp
// __shfl_down_sync version (places max at lane 0)
for (int offset {16}; offset > 0; offset /= 2) {
    value = get_max(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
}

// __shfl_xor_sync version (broadcasts sum to ALL lanes)
for (int offset {16}; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
}
```
Key distinction Ernest tracks carefully:
- `__shfl_down_sync`: result ends up at lane 0 only (others get partial sums)
- `__shfl_xor_sync`: result broadcast to ALL lanes

### Shared memory halo pattern (from Convolution1D.h)
```cpp
__shared__ T shared_array[THREADS_PER_BLOCK + (N_b - 1) + N_b];
T* shared_a = shared_array;                        // [0 .. THREADS_PER_BLOCK + N_b - 2]
T* shared_b = shared_array + THREADS_PER_BLOCK + N_b - 1;  // [THREADS_PER_BLOCK + N_b - 1 ..]
```
Ernest explicitly labels the halo: `// 0 1 2 3 | 4 5 |` where `a[4],a[5]` are boundary values.

---

## Ernest's Mathematical Voice (from CompPhys.tex)

### Core philosophy: "everything is a map"
Ernest treats computation as category theory. Every operation is a morphism:
```
a : Z → Type   (array as a function from index to element)
a : i ↦ a[i]
```
This is not just notation — he uses commutative diagrams and categorical language throughout.

### Thread/block index as categorical structure
```
<<<>>> : (n_block, n_threads) × kernelfunctions ↦ kernelfunction<<<n_block,n_threads>>> ∈ End:Dat_GPU
threadIdx ⊂ FinSet  (subcategory of the category of finite sets)
x : threadIdx ↦ threadIdx.x ∈ Obj_FinSet
```
The 3D thread index `(i_x, i_y, i_z)` is an element of a Cartesian product:
```
∀ (n_blocks, n_threads) ∈ Z × {1..1024},
{1..n_block} × {1..n_threads} is an ordered index set (lexicographic ordering)
```

### How to explain a kernel "as a mathematician"
Ernest connects GPU execution to these structures:

**1. The kernel launch = an endomorphism on data**
```
kernel<<<N_x, M_x>>> : Dat_GPU → Dat_GPU
```
where `N_x = ⌈L_x / M_x⌉` (ceiling division), and `M_x = THREADS_PER_BLOCK`.

**2. Thread index = element of a finite set / free module**
```
(i_x, j_x) ∈ {0..M_x-1} × {0..N_x-1}   ↔   (threadIdx.x, blockIdx.x)
k = i_x + M_x · j_x                       ↔   global thread index
```
The "flattening" `k = i_x + M_x · j_x` is the canonical isomorphism
`{0..M_x-1} × {0..N_x-1} → {0..M_x·N_x-1}`.

**3. The warp = a submodule / equivalence class**
Warp w = threads with `threadIdx.x / 32 == w`.
`lane_id = threadIdx.x mod 32` — the residue in Z/32Z.

**4. Warp shuffle reduction as a "fold" over a free Z-module**
`warp_reduce_max`: given `f : Z/32Z → R`, compute `max_{i} f(i)` using
the log₂(32) = 5 step binary tree via `__shfl_down_sync`.
`warp_reduce_sum` with `__shfl_xor_sync`: this is a "symmetric" reduction —
the XOR of lane IDs pairs every lane with its complement, so all lanes
receive the total sum. Formally it's the action of the group algebra Z[(Z/32Z)]
on the value register.

**5. Shared memory = a local section / fiber**
The shared memory of a thread block is a locally-defined array:
```
__shared__ T s[BLOCK];
s : {0..BLOCK-1} → T
```
This is the "fiber" over a block — local data visible only to threads in that block.
The shared memory halo for convolution is exactly the "stalk" of a sheaf:
given output positions `{j·M_x .. (j+1)·M_x - 1}`, the kernel needs input
positions `{j·M_x - (N_b-1) .. (j+1)·M_x - 1}`, i.e. the block's "support"
plus a (N_b-1)-element boundary collar.

---

## Mathematical Framing of Our Hackathon Kernels

### FP8 Per-Group Quantization

**The map being computed:**
Let `x ∈ ℝ^{T×H}`. Define the group partition:
```
π : {0..H-1} → {0..G-1},   π(j) = ⌊j / gsz⌋
```
This is a surjective map from column indices onto group indices.
Each "fiber" `π⁻¹(g) = {g·gsz, ..., (g+1)·gsz-1}` is a group of `gsz` elements.

**The quantization is a family of maps indexed by (token, group):**
```
Q_{t,g} : ℝ^gsz → ℝ^gsz × ℝ
Q_{t,g}(v) = (clamp(v / σ_{t,g}, -448, 448), σ_{t,g})
where σ_{t,g} = max(‖v‖_∞, ε) / 448
```
Here `‖v‖_∞ = max_j |v_j|` is the ℓ∞ norm on the group.

**The kernel computes this as a product of independent operations:**
The N = T×G groups are processed in parallel. The reduction `max |v_j|` over
j ∈ {0..gsz-1} is a morphism `ℝ^gsz → ℝ` in the category of ℝ-modules.
Since `gsz` ∈ {64, 128} ≤ 128, the entire group fits within a single warp
(32 threads × 4 elements each) or two warps — the reduction is a single
warp-shuffle tree of depth log₂(gsz).

In Ernest's notation:
```
K : {0..N-1} × {0..gsz-1} → ℝ × ℝ
(n, j) ↦ (clamp(data[n,j]/scale[n], -448, 448), scale[n])
where n = i_x + M_x · j_x   (global group index from tile)
```

### Causal Depthwise Conv1D

**The map being computed:**
Let `x ∈ ℝ^{B×D×S}`, `w ∈ ℝ^{D×W}`, `b ∈ ℝ^D`. Define the output:
```
y[b,d,t] = b_d + Σ_{k=0}^{W-1} w[d,k] · x[b, d, t - W + 1 + k]
```
where we set `x[b,d,s] = 0` for `s < 0` (causal zero padding).

**As a convolution in the ring of formal power series:**
For fixed `(b,d)`, write the input sequence as a polynomial:
`x_{b,d}(z) = Σ_t x[b,d,t] z^t`
and the filter as `w_d(z) = Σ_k w[d,k] z^k`.

Then `y_{b,d} = b_d · 1 + (x_{b,d} ★_causal w_d)` where `★_causal` truncates
to non-negative powers (causal = lower-triangular Toeplitz matrix multiplication).

**The depthwise structure:**
`D` channels are processed independently → the kernel is a product
`∏_{d=0}^{D-1} (conv1d_d)` of D independent 1D convolutions.
Each `conv1d_d` acts on `ℝ^S` with the same W-tap filter `w_d`.

**The tile structure in Helion:**
`hl.tile([B, D, S], block_size=[1, D_tile, S_tile])` partitions the output space as:
```
{0..B-1} × {0..D-1} × {0..S-1}
 ≅ B · ⌈D/D_tile⌉ · ⌈S/S_tile⌉  thread blocks
```
Each block computes a `D_tile × S_tile` subtensor of `y`.

**The halo (boundary collar) as a sheaf stalk:**
To compute `y[b,d,t..t+S_tile-1]`, we need input positions
`x[b,d, t..t+S_tile-1+W-1]` — a set of `S_tile + W - 1` elements.
This is the stalk of the input sheaf over the output interval extended
by the filter support `{0..W-1}`.

---

## Notes for Future Explanations

When Ernest asks "explain this like a mathematician":
1. Name the domain/codomain sets explicitly (finite index sets, products, quotients)
2. Name the map, its type, and whether it factors through something
3. Use `↦` (mapsto) notation, `∀`, `∃`, `⊂`, `≅`
4. Point out where a CUDA concept corresponds to: free module, quotient, fiber, sheaf stalk, group action, endomorphism, fold/reduce in a monoid
5. Annotate indices: what is each index ranging over, what is being parallelized
6. The warp = action of Z/32Z; shared memory = local section; tile = coset representative
