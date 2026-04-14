# Hybrid-EP Implementation Guide

## Table of Contents
1. [Overview](#1-overview)
2. [Interface](#2-interface)
3. [Config](#3-config)
4. [Buffer Management](#4-buffer-management)
5. [Executor](#5-executor)
6. [JIT Compiler](#6-jit-compiler)
7. [Extensions](#7-extensions)
8. [Hybrid-EP Kernels](#8-hybrid-ep-kernels)
9. [Allocator](#9-allocator)

---

## 1. Overview

### Architecture Diagram

![Hybrid-EP Workflow](../figures/hybrid-ep-img/Hybrid-EP-workflow.svg)

### Code Structure
```
csrc/hybrid_ep/
├── hybrid_ep.*                    # Main HybridEPBuffer class
├── pybind_hybrid_ep.cu            # PyBind bindings
├── config.cuh                     # Config definitions
├── utils.cuh                      # Utility helpers and macros
├── allocator/                     # MNNVL/IPC memory allocator
├── backend/                       # Core dispatch/combine kernels
│   ├── hybrid_ep_backend.cuh      # Kernel implementations
│   ├── ibvcore.h                  # InfiniBand verbs definitions
│   └── topo_detection.cuh         # GPU topology detection
├── buffer/                        # Buffer coordinators
│   ├── intranode.*                # NVLCoordinator (intra-node communication)
│   └── internode.*                # RDMACoordinator (inter-node communication)
├── executor/                      # Kernel execution (dispatch/combine core)
├── extension/                     # Extensions (allgather, permute)
└── jit/                           # JIT kernel compiler
    
deep_ep/
├── hybrid_ep_buffer.py            # Python interface
└── buffer.py                      # Buffer management

tests/
├── test_hybrid_ep.py              # Functional tests
└── test_graphed_hybrid_ep.py      # CUDA Graph tests
```

---

## 2. Interface

### `__init__`

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `group` | `torch.distributed.ProcessGroup` | PyTorch distributed process group |
| `hidden_dim` | `int` | Hidden dimension of tokens |
| `max_num_of_tokens_per_rank` | `int` | Maximum tokens per rank, used for buffer allocation|
| `num_local_experts` | `int` | Number of experts on each rank |
| `use_fp8` | `bool` | Use FP8 quantization (default: False) |
| `num_sms_dispatch_api` | `int` | SMs for dispatch kernel |
| `num_sms_combine_api` | `int` | SMs for combine kernel |
| `num_sms_preprocessing_api` | `int` | CUDA blocks for preprocessing kernel |
| `num_blocks_permute` | `int` | CUDA blocks for permute (standalone or fused permute-dispatch) |
| `num_blocks_unpermute` | `int` | CUDA blocks for unpermute (standalone or fused unpermute-combine) |
| `load_cached_kernels` | `bool` | Load pre-compiled JIT kernels (default: False) |
| `use_shared_buffer` | `bool` | Share intra-node buffer between dispatch/combine (default: True) |
| `enable_custom_allgather` | `bool` | Use optimized intra-node allgather (default: False) |

### `dispatch` / `dispatch_with_permute`

Dispatch tokens to target experts. Use `dispatch_with_permute` for integrated permutation (see [7.2 Permutation](#72-permutation)).

> **Routing Input Modes** (choose one):
> - **Index-based**: `topk_idx` + `topk_weights` + `num_of_experts`
> - **Map-based**: `routing_map` + `probs`

**Common Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden` | `Tensor[N, D]` | Input token embeddings |
| `topk_idx` | `Tensor[N, K]` | Top-K expert indices per token |
| `topk_weights` | `Tensor[N, K]` | Top-K routing weights |
| `num_of_experts` | `int` | Total number of experts |
| `routing_map` | `Tensor[N, E]` | Boolean routing map |
| `probs` | `Tensor[N, E]` | Routing probabilities |
| `scaling_factor` | `Tensor` | FP8 scaling factor |
| `handle` | `tuple` | Cached metadata from previous call |

**Additional Inputs for `dispatch_with_permute`:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `num_of_experts_per_rank` | `int` | Experts per rank |
| `pad_multiple` | `int` | Pad output to multiple (for GEMM alignment) |
| `num_permuted_tokens` | `int` | Expected output size (for non_blocking) |
| `non_blocking` | `bool` | Skip sync, use GPU-side metadata (default: False) |
| `fuse_permute_dispatch` | `bool` | Fuse permute into the dispatch kernel (default: False) |

> **Non-blocking Mode:** When `non_blocking=True`, stream synchronizations are skipped. Output buffer is sized by `num_permuted_tokens`; overflow sets `overflow_flag=True` and drops excess tokens. In non-blocking mode, `num_dispatched_tokens_tensor` and `tokens_per_expert` are GPU tensors; otherwise they reside in CPU pinned memory.

**Outputs:**
| Return | `dispatch` | `dispatch_with_permute` |
|--------|------------|-------------------------|
| `dispatched_token` | Tokens for local experts | Permuted tokens grouped by expert |
| `dispatched_probs` | Routing probabilities | Routing probabilities |
| `dispatched_scaling_factor` | FP8 scaling factors | FP8 scaling factors |
| `tokens_per_expert` | - | `Tensor[E]`: Token count per expert |
| `handle` | For `combine` | For `combine_with_unpermute` |

### `combine` / `combine_with_unpermute`

Combine tokens from experts back to original positions. Use corresponding method based on dispatch variant.

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden` | `Tensor` | Expert output embeddings |
| `probs` | `Tensor` | Routing probabilities for weighted sum |
| `handle` | `tuple` | Metadata from dispatch (required) |
| `pad_multiple` | `int` | Padding alignment (`combine_with_unpermute` only) |
| `fuse_unpermute_combine` | `bool` | Fuse unpermute into the combine kernel (default: False) |

**Outputs:**
| Return | Type | Description |
|--------|------|-------------|
| `combined_token` | `Tensor[N, D]` | Combined tokens in original order |
| `combined_probs` | `Tensor` | Aggregated probabilities |

### `empty_jit_cache`

Clear all cached JIT-compiled kernels from disk.

### Handle Structure

The `handle` returned by dispatch methods contains precomputed metadata for the corresponding dispatch/combine call.

#### `dispatch` Handle

```python
handle = (
    sparse_to_dense_map,         # [0] Tensor
    rdma_to_attn_map,            # [1] Tensor
    attn_to_rdma_map,            # [2] Tensor
    num_dispatched_tokens_tensor,# [3] Tensor: Total dispatched token count
    local_expert_routing_map,    # [4] Tensor: Local expert routing information
    num_of_tokens,               # [5] int: Number of tokens per rank
    config,                      # [6] HybridEpConfigInstance: Runtime configuration
)
```

#### `dispatch_with_permute` Handle (independent permute)

```python
handle = (
    sparse_to_dense_map,         # [0] Tensor
    rdma_to_attn_map,            # [1] Tensor
    attn_to_rdma_map,            # [2] Tensor
    num_dispatched_tokens_tensor,# [3] Tensor: Total dispatched token count 
    local_expert_routing_map,    # [4] Tensor: Local expert routing information
    row_id_map,                  # [5] Tensor: Row permutation mapping for unpermute
    num_of_tokens_per_rank,      # [6] int: Number of tokens per rank
    config,                      # [7] HybridEpConfigInstance: Runtime configuration
    overflow_flag,               # [8] Tensor: Buffer overflow indicator
)
```

#### `dispatch_with_permute` Handle (fused permute)

When `fuse_permute_dispatch=True`, `row_id_map` is replaced by fused-mode metadata:

```python
handle = (
    sparse_to_dense_map,         # [0] Tensor
    rdma_to_attn_map,            # [1] Tensor
    attn_to_rdma_map,            # [2] Tensor
    num_dispatched_tokens_tensor,# [3] Tensor: Total dispatched token count
    local_expert_routing_map,    # [4] Tensor: Local expert routing information
    dense_chunk_layout,          # [5] Tensor: Per-chunk start positions in the per-rank buffer
    dense_to_expert_map,         # [6] Tensor: Token-to-local-expert mapping
    tokens_per_expert,           # [7] Tensor: Token count per local expert
    num_of_tokens_per_rank,      # [8] int: Number of tokens per rank
    config,                      # [9] HybridEpConfigInstance: Runtime configuration
    overflow_flag,               # [10] Tensor: Buffer overflow indicator
)
```

---

## 3. Config

Hybrid-EP uses two configuration structures defined in [`config.cuh`](../csrc/hybrid_ep/config.cuh):

- **`BufferConfig`**: Parameters used solely for buffer size calculation, stored persistently in the buffer object.
- **`HybridEpConfigInstance`**: Contains all parameters needed for JIT-compiling and launching Hybrid-EP kernels. A new instance is created for each invocation.

Each run compares the new `HybridEpConfigInstance` against `BufferConfig` to detect whether existing buffers are sufficient. If not, a free-reallocate cycle is triggered (see [4. Buffer Management](#4-buffer-management)).

### Parameter Reference

#### Runtime Parameters

These parameters are typically derived from the model configuration:

| Parameter | Description |
|-----------|-------------|
| `hidden_dim` | Hidden size (must match model hidden dimension) |
| `max_num_of_tokens_per_rank` | Maximum sequence length for dispatch kernel input |
| `num_of_experts_per_rank` | Number of experts hosted by each rank |
| `num_of_nodes` | Number of NVLink domains (not OS nodes/containers) |
| `num_of_ranks_per_node` | Number of ranks within one NVLink domain |

#### Performance Tuning Parameters

These parameters are pre-tuned for optimal performance. Adjustments are generally not recommended, but can be made via environment variables for specific hardware configurations:

| Parameter | Env Variable | Description |
|-----------|--------------|-------------|
| `num_of_threads_per_block_preprocessing_api` | `NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API` | Thread-block width for preprocessing kernel |
| `num_of_blocks_preprocessing_api` | - | Grid size for preprocessing kernel |
| `num_of_stages_dispatch_api` | `NUM_OF_STAGES_DISPATCH_API` | Pipeline depth for dispatch. Larger improves occupancy but increases shared memory usage. Reduce if `hidden_dim` is very large |
| `num_of_blocks_dispatch_api` | - | Number of CTAs for dispatch; controls SM utilization |
| `num_of_stages_g2s_combine_api` | `NUM_OF_STAGES_G2S_COMBINE_API` | Pipeline depth for global-to-shared in combine. Same shared memory trade-off as dispatch |
| `num_of_stages_s2g_combine_api` | `NUM_OF_STAGES_S2G_COMBINE_API` | Pipeline depth for shared-to-global in combine |
| `num_of_blocks_combine_api` | - | Number of CTAs for combine kernels |
| `num_of_blocks_permute` | - | Number of CUDA blocks for the permute portion (standalone or fused dispatch) |
| `num_of_stages_permute_block_dispatch_api` | `NUM_OF_STAGES_PERMUTE_BLOCK_DISPATCH_API` | Shared-memory pipeline stages for permute blocks in fused dispatch |
| `num_of_in_flight_s2g_permute_block_dispatch_api` | `NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK_DISPATCH_API` | In-flight S2G token entries for permute blocks in fused dispatch |
| `num_of_blocks_unpermute` | - | Number of CUDA blocks for the unpermute portion (standalone or fused combine) |
| `num_of_stages_g2s_unpermute_block` | `NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK` | Pipeline depth for G2S in unpermute blocks (fused combine) |
| `num_of_stages_s2g_unpermute_block` | `NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK` | Pipeline depth for S2G in unpermute blocks (fused combine) |
| `num_of_additional_in_flight_s2g_unpermute_block_combine_api` | `NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_UNPERMUTE_BLOCK_COMBINE_API` | In-flight S2G token entries for unpermute blocks in fused combine |

> **Fused-mode constraint:** When `fuse_permute_dispatch=True`, `is_valid()` additionally requires all chunk sizes (`num_of_tokens_per_chunk_dispatch_api`, `num_of_tokens_per_chunk_combine_api`, `num_of_tokens_per_chunk_preprocessing_api`) to be identical.

### Note on `max_num_of_tokens_per_rank`

During JIT compilation, `max_num_of_tokens_per_rank` serves as a template parameter for static resource allocation. At runtime, the actual `num_of_tokens_per_rank` is passed and must satisfy:

```
num_of_tokens_per_rank <= max_num_of_tokens_per_rank
```

Since `max_num_of_tokens_per_rank` also determines buffer allocation size, Hybrid-EP automatically updates this value on each run to ensure sufficient capacity

---

## 4. Buffer Management

Hybrid-EP uses two types of coordinators to manage communication buffers:
- **NVLCoordinator**: Manages intra-node communication buffers
- **RDMACoordinator**: Manages inter-node communication buffers

### 4.1 Buffer Types

Hybrid-EP uses two categories of GPU memory:

- **Registered Buffer**: GPU memory registered for cross-rank access. For inter-node, memory is registered with RDMA; for intra-node, a CUDA IPC handle is exported.
- **Normal Buffer**: Standard `cudaMalloc` memory for local computation, not accessible by other ranks.

**GPU-NIC Mapping for RDMA**

For RDMA scenarios, buffer registration requires establishing a GPU-NIC mapping to specify which network interface each GPU uses for communication. The backend supports automatic topology discovery by default. If manual configuration is needed, the following environment variables can be used:

| Environment Variable | Description |
|---------------------|-------------|
| `HYBRID_EP_ENABLE_MANUAL_NIC_MAPPING` | Set to `1` to enable manual NIC mapping; otherwise, automatic topology discovery is used |
| `HYBRID_EP_NIC_MAPPING` | GPU-to-NIC mapping string in the format `<gpu_id>:<nic_name>,...` |

### 4.2 Buffer Allocation

Hybrid-EP uses a **worst-case preallocation strategy** to handle dynamic token routing. The maximum token count assumes all tokens could be routed to a single expert:

```
max_num_of_tokens = max_num_of_tokens_per_rank × num_of_ranks_per_node × num_of_nodes
```

**Intra-node buffers** (per rank):
- `token_buffer`: `max_tokens × hidden_dim × sizeof(dtype)`
- `prob_buffer`: `max_tokens × (num_of_experts_per_rank × num_of_ranks_per_node) × sizeof(float)`
- `scaling_factor_buffer`: `max_tokens × (hidden_dim / 128) × sizeof(float)` — FP8 only

**Inter-node RDMA buffers** (when `num_of_nodes > 1`, per rank):
- `rdma_token_buffer`: `max_tokens_per_rank × (num_nodes - 1) × hidden_dim × sizeof(dtype)`
- `rdma_prob_buffer`: `max_tokens_per_rank × (num_nodes - 1) × (num_of_experts_per_rank × num_of_ranks_per_node) × sizeof(float)`

Per RDMA buffer footprint is capped at ≤1 GB.

**Example (EP64, 256 experts, BF16, hidden_dim=7168, 4096 tokens/rank):**

| Scenario | Buffer | Calculation | Size |
|----------|--------|-------------|------|
| **MNNVL** (64 ranks × 1 node) | max_tokens | 4096 × 64 × 1 | 262,144 |
| | Intra-node token | 262,144 × 7168 × 2B | ~3.5 GB |
| | Intra-node prob | 262,144 × 256 × 4B | ~256 MB |
| | **Total per rank** | | **~3.75 GB** |
| **RDMA** (8 ranks × 8 nodes) | max_tokens | 4096 × 8 × 8 | 262,144 |
| | Intra-node token | 262,144 × 7168 × 2B | ~3.5 GB |
| | Intra-node prob | 262,144 × 32 × 4B | ~32 MB |
| | RDMA token | 4096 × 7 × 7168 × 2B | ~392 MB |
| | RDMA prob | 4096 × 7 × 32 × 4B | ~3.5 MB |
| | **Total per rank** | | **~4 GB** |

Because this buffer is globally unique per rank, overall GPU memory usage remains controllable.

### 4.3 Shared Buffer

By default (`use_shared_buffer=True`), dispatch and combine share the same buffer since they execute sequentially. This halves memory usage but means concurrent Hybrid-EP operations sharing the same buffer cannot overlap.

> **Important:** Because the buffer is global, any two Hybrid-EP communication operations that share the same buffer **cannot overlap**.

### 4.4 Buffer Registration Workflow

Buffer allocation and registration are time-consuming operations, so they are performed only during initialization. The workflow is:

1. **Allocate**: Each rank allocates local buffers using `ExtendedMemoryAllocator`
2. **Get Handle**: Create IPC-exportable handles for registered buffers
3. **Exchange**: Use `torch.distributed.all_gather` to exchange handles across ranks
4. **Open Handle**: Each rank opens handles from other ranks to obtain remote buffer addresses

![Buffer Registration](../figures/hybrid-ep-img/RegisterBuffer.png)

### 4.5 Dynamic Buffer Update

`BufferConfig` is a subset of `HybridEpConfigInstance`. Each run compares them to check if reallocation is needed:

```
RuntimeHybridEpConfigInstance ──compare──► StaticBufferConfig
              │                                   │
              └─ need larger? ─── No ────────────►│ Use existing
                              └── Yes ──► release ──► update ──► allocate ──► exchange
```

> **Warning:** Buffer reallocation is slow due to memory registration and handle exchange. Pre-allocate sufficient capacity via `max_num_of_tokens_per_rank` to avoid runtime reallocation.

---

## 5. Executor

The Executor orchestrates all communication operations. Each API is divided into three stages: **preprocess**, **core**, and **postprocess**.

| API | Preprocess | Core | Postprocess |
|-----|------------|------|-------------|
| `metadata_preprocess` | - | Allgather + metadata kernel + permute preprocessing | - |
| `dispatch` | D2D to RDMA buffer* | Dispatch kernel | D2D from buffer |
| `dispatch_with_permute` | D2D to RDMA buffer*| Dispatch kernel | Permute kernel |
| `dispatch_with_permute` (fused) | D2D to RDMA buffer*  | Fused dispatch+permute kernel | - |
| `combine` | D2D to buffer | Combine kernel | - |
| `combine_with_unpermute` | Unpermute kernel | Combine kernel | - |
| `combine_with_unpermute` (fused) | - | Fused combine+unpermute kernel | - |

\* Only when `num_of_nodes > 1`

In fused mode, the permute/unpermute operations are executed by dedicated permute blocks within the same kernel grid as the dispatch/combine blocks (see [8. Hybrid-EP Kernels](#8-hybrid-ep-kernels)).

---

## 6. JIT Compiler

Hybrid-EP uses NVCC JIT compilation to generate optimized kernels based on runtime configuration.

### 6.1 JIT Workflow

1. **Code Generation**: Generate kernel source code from `HybridEpConfigInstance` template parameters
2. **Compile**: Execute `nvcc` via command line to compile the `.cu` source to a `.so` shared library
3. **Load**: Open the compiled `.so` with `dlopen` and resolve function symbols via `dlsym`
4. **Cache**: Store the function pointer in `KernelCache` for subsequent calls

```
HybridEpConfigInstance ──► Generate .cu ──► nvcc compile ──► .so ──► dlopen/dlsym ──► function pointer
```

When `fuse_permute_dispatch=True` or `fuse_unpermute_combine=True`, the `build()` method passes `-DHYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE` to nvcc, which activates the fused permute/unpermute code paths in the kernel templates. The same macro controls both dispatch-side permute fusion and combine-side unpermute fusion. Fused and non-fused kernels produce separate `.so` files; each is loaded with `RTLD_LOCAL` to prevent symbol conflicts between the two variants.

### 6.2 Cache Management

All ranks compile identical kernels, but use unique filenames (including `node_rank`, `local_rank`, and timestamp) to avoid conflicts. After compilation, `std::filesystem::rename` atomically merges them into a single cached `.so`.

**Cache Directory:**
- Environment variable: `HYBRID_EP_CACHE_DIR`
- Default: `$HOME/.deepep/hybrid_ep/jit` (fallback: `/tmp/.deepep/hybrid_ep/jit`)

**Reusing Cache:**
Set `load_cached_kernels=True` in `HybridEPBuffer.__init__()` to load pre-compiled kernels from the cache directory, avoiding recompilation overhead.

---

## 7. Extensions

### 7.1 Allgather

> **Experimental Feature**

Custom allgather implementation that replaces NCCL's allgather for routing map collection. This avoids the overhead of copying from NCCL's internal buffer to PyTorch tensors. Enable via `enable_custom_allgather=True` in initialization.

### 7.2 Permutation

Since dispatch output and combine input reside in registered buffers (see [4. Buffer Management](#4-buffer-management)), they cannot be managed by PyTorch's caching allocator. This normally requires an extra D2D copy to move results into standard PyTorch tensors—which is exactly what `dispatch` does.

`dispatch_with_permute` integrates permutation with this data movement: it permutes tokens by expert while simultaneously copying to PyTorch-managed memory, making the output directly usable by expert kernels.

Conversely, `combine_with_unpermute` reverses this process: it unpermutes expert outputs back to original token order while copying from PyTorch tensors to registered buffers before the combine operation.

Additionally, some training frameworks require token alignment per expert (e.g., for efficient GEMM). The `pad_multiple` parameter enables padding to meet these alignment requirements.

Hybrid-EP supports two execution modes for permutation:

- **Independent mode** (default): The permute/unpermute operation runs as a separate kernel after/before the dispatch/combine kernel.
- **Fused mode** (`fuse_permute_dispatch=True` / `fuse_unpermute_combine=True`): The permute/unpermute operation is fused into the dispatch/combine communication kernel itself. Dedicated permute blocks run within the same kernel grid as the dispatch/combine blocks, reading from per-rank buffers as chunks become ready and writing directly to the output tensors.

#### Buffer Allocation Challenge

When using `dispatch_with_permute` / `combine_with_unpermute`, we face a fundamental challenge: **the number of permuted tokens is unknown before preprocessing completes** due to the dynamic nature of MoE routing.

Allocating buffers for the worst case would require `worst_case_dispatch_output × min(topk, num_experts_per_rank)` tokens—an unacceptable memory overhead. Additionally, the permuted output serves as expert input (activation memory) and cannot be globally reused. Furthermore, integrating permutation into Hybrid-EP **eliminates the opportunity for recomputing permutation** during backward pass.

#### Default Mode (Blocking)

By default, `dispatch_with_permute` performs a **stream synchronization** after the preprocessing kernel to obtain the exact token count, then allocates a precisely-sized buffer using `torch.empty()`:

#### Blocking (default)

A stream synchronization obtains the exact token count, then allocates a precisely-sized buffer. The dispatch kernel is followed by a permute step -- either a separate permute kernel (independent) or fused permute blocks within the same kernel grid (`fuse_permute_dispatch=True`):

```
┌───────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────────────────────┐
│ Permute           │────▶│ Stream Sync  │────▶│ Allocate Buffer │────▶│ Dispatch Kernel + Permute        │
│ Preprocessing     │     │ (blocking)   │     │ (exact size)    │     │ (independent or fused)           │
└───────────────────┘     └──────────────┘     └─────────────────┘     └──────────────────────────────────┘
```

#### Non-Blocking

For CUDA graph capture, `non_blocking=True` allows users to provide an estimated token count (`num_permuted_tokens`) to avoid synchronization. The permute step is likewise independent or fused:

```
┌───────────────────┐     ┌─────────────────────┐     ┌──────────────────────────────────┐
│ Permute           │────▶│ Allocate Buffer     │────▶│ Dispatch Kernel + Permute        │
│ Preprocessing     │     │ (user-estimated)    │     │ (independent or fused)           │
└───────────────────┘     └─────────────────────┘     └──────────────────────────────────┘
```

**Output behavior**:
- **Overflow** (actual > estimated): Excess tokens are dropped, `overflow_flag = True`
- **Underflow** (actual ≤ estimated): Trailing portion contains garbage data, use `tokens_per_expert` to find valid ranges

---

## 8. Hybrid-EP Kernels

The dispatch and combine kernels in Hybrid-EP are **warp-specialized persistent kernels**:

- **Persistent**: Each CUDA block exclusively occupies one SM for its entire lifetime
- **Warp-specialized**: Different warp groups within a block form a pipelined dataflow
- **Independent blocks**: Data is divided into chunks evenly distributed across blocks

**Data type support:**
- Dispatch kernel supports BF16 and FP8. For FP8, scaling factors are transmitted alongside tokens (one 32-bit scaling factor(4 uint8 or 1 fp32) per 128 elements)
- Combine kernel only supports BF16

**RDMA communication pattern:**
We define GPUs with the same local rank across different nodes as *peers* (e.g., GPU 0 on node A peers with GPU 0 on node B). Similar to DeepEP, Hybrid-EP first performs IB communication between peer GPUs, then each rank distributes the received inter-node data locally. This overlaps IB communication with intra-node NVLink operations to hide latency.

### 8.1 Dispatch Kernel

![Dispatch Kernel](../figures/hybrid-ep-img/hybrid_ep_dispatch.png)

The dispatch kernel moves tokens from attention output to expert input buffers:

1. **G2S Warp Group**: Loads tokens/probs/scaling_factors from global memory to shared memory
2. **SMEM Cyclic FIFO**: Staging buffer between pipeline stages
3. **S2G Warp Group**: Writes data to remote ranks' buffers via NVLink
4. **RDMA Warp Group** (multi-node only): Sends data to corresponding ranks on other nodes; also receives RDMA data from other nodes

#### Fused Permute Blocks

When `fuse_permute_dispatch=True` (compiled with `-DHYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE`), the kernel grid is extended with `NUM_OF_PERMUTE_BLOCKS` additional blocks appended after the `NUM_OF_BLOCKS` dispatch blocks. Block role is determined by `blockIdx.x`:

- `blockIdx.x < NUM_OF_BLOCKS` — dispatch block (original logic above)
- `blockIdx.x >= NUM_OF_BLOCKS` — permute block

Each permute block contains two warp groups:

5. **Permute G2S Warp Group**: Reads dispatched tokens from per-rank NVLink buffers. Waits on chunk-ready flags (`intra_node_expert_output_chunk_flags`) set by dispatch S2G warp groups when a chunk's write to the per-rank buffer completes.
6. **Permute S2G Warp Group**: Writes permuted tokens directly to the output tensor (`local_expert_output_token`), grouped by expert. Handles padding initialization for `pad_multiple` alignment.

**Synchronization**: Dispatch S2G warp groups notify permute G2S warp groups via per-chunk flags. A monotonically increasing `expected_permute_flag_value` tracks completion across invocations without resetting the flags.

### 8.2 Combine Kernel

![Combine Kernel](../figures/hybrid-ep-img/hybrid_ep_combine.png)

The combine kernel aggregates expert outputs back to original token positions.

1. **G2S Warp Group**: Loads tokens/probs/scaling_factors from global memory to shared memory
   - *Inter-node*: Loads data for local distribution
   - *Intra-node* (multi-node only): Loads data to be sent to peers
2. **SMEM Cyclic FIFO**: Staging buffer between pipeline stages
3. **Reduce Warp Group**: Reduces data from different ranks
   - *Inter-node*: Reduces data received from all ranks
   - *Intra-node* (multi-node only): Pre-reduces data within node before sending to peers
4. **S2G Warp Group**: Writes data to buffers
   - *Inter-node*: Writes to intra-node peers' buffers via NVLink
   - *Intra-node* (multi-node only): Writes to RDMA buffer for cross-node transfer
5. **RDMA Warp Group** (multi-node only): Sends data to peer ranks on other nodes; also receives RDMA data from peers

#### Fused Unpermute Blocks

When `fuse_unpermute_combine=True` (compiled with `-DHYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE`), the kernel grid is extended with `NUM_OF_UNPERMUTE_BLOCKS` additional blocks appended after the `NUM_OF_BLOCKS` combine blocks. Block role is determined by `blockIdx.x`:

- `blockIdx.x < NUM_OF_BLOCKS` — combine block (original logic above)
- `blockIdx.x >= NUM_OF_BLOCKS` — unpermute block

Each unpermute block contains two warp groups:

6. **Unpermute G2S Warp Group**: Reads expert output tokens from the user's input tensor (`local_expert_input_token`) using `dense_to_expert_map` to traverse tokens in expert-grouped order. Produces token entries into the shared memory G2S FIFO with `num_of_stages_g2s_unpermute_block` pipeline depth.
7. **Unpermute Red Warp Group**: Consumes token entries from the shared memory G2S FIFO, rearranges them from expert-grouped order to chunk-based order, and writes to the local rank's NVLink buffer (`expert_input_token[local_rank]`). Sets chunk-ready flags (`intra_node_expert_input_chunk_flags`) on all ranks after each chunk write completes, notifying combine G2S warp groups that the data is ready for reading.

## 9. Allocator

### 9.1 Memory Allocation

The allocator provides cross-rank accessible memory with these operations: `allocate`, `free`, `get_handle`, `open_handle`, `close_handle`.

- **Fabric mode**: Uses `cuMemCreate`/`cuMemMap` and `cuMemExportToShareableHandle` for MNNVL systems
- **IPC mode**: Uses `cudaMalloc` and `cudaIpcGetMemHandle` for standard multi-GPU systems

### 9.2 Topology Detection

The allocator detects which ranks share the same NVLink domain via `detect_accessible_ranks()`, which exchanges test memory handles and checks accessibility.

To override automatic detection, set:

```bash
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
```

This environment variable specifies the number of ranks that can directly access each other's GPU memory within a single NVLink domain. It determines:
- `local_rank = rank % NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` — rank index within the NVLink domain
- `node_rank = rank // NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` — which NVLink domain this rank belongs to
- `num_of_nodes = group_size // NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` — total number of NVLink domains

The value must evenly divide the total number of ranks.

To disable MNNVL (fabric) support and fall back to IPC-based memory sharing, set:

```bash
export USE_MNNVL=0   # or USE_MNNVL=false (case-insensitive)
```

When this variable is set to `0` or `false`, the allocator forces `support_fabric_` to `false` regardless of hardware capability, causing all allocations to use `cudaMalloc` / `cudaIpc*` instead of fabric handles.

---

## 10. Optimizations for Latent MoE Configurations

These optimizations target latent MoE configurations where `HIDDEN_DIM` is small (e.g., 512) and `NUM_LOCAL_EXPERTS * NUM_RANKS` is large (e.g., 32 * 72 = 2304), making metadata (probs, routing maps) dominate over token data in NVLink traffic.

### 10.1 Sparse Prob Optimization (Dispatch)

**Problem:** In the forward dispatch, the S2G warp writes the full `E_per_rank * R_per_node` prob vector per token to each destination rank, but only the destination rank's `E_per_rank` slice is non-zero. For H=512, E=32, R=72: 9216 B of prob per token, of which only 128 B is useful (98.6% waste). Prob traffic dominated 90% of total NVLink bytes.

**Fix:** Change the S2G TMA write to send only the destination rank's `E_per_rank` slice (128 B) instead of the full `E_per_rank * R_per_node` vector (9216 B on NVL72). Source offset adjusted to `remote_rank_id * E_per_rank` in the existing SMEM buffer. Static asserts enforce TMA alignment (`E_per_rank * sizeof(float) >= 16` and `% 16 == 0`).

**Files changed:** `hybrid_ep_backend.cuh` — dispatch S2G prob TMA write (12 lines changed).

**Impact:** 72x prob traffic reduction on NVL72, dispatch kernel time improved from 228 us to 102 us (2.2x) on B300 NVL8.

### 10.2 Sparse Prob Optimization (Combine)

**Problem:** In backward combine (`BACKWARD_COMBINE=true`), each G2S NVLink read loads the full `E_per_rank * R_per_node` prob vector from each source rank's buffer, but only the source rank's `E_per_rank` slice is non-zero (set by the dispatch sparse prob optimization). Same waste pattern as dispatch.

**Fix:** Shrink the G2S prob SMEM buffer from `[stages][E*R]` to `[stages][E_per_rank]`. Add a per-stage `int` source rank ID field so the reduction warp group places the `E_per_rank` elements at the correct offset in the `E*R` accumulator. Both intra-node and inter-node G2S paths updated.

**Files changed:** `hybrid_ep_backend.cuh` (SMEM structs, G2S reads, reduction accumulation), `config.cuh` (SMEM size calculation).

**Impact on NVL72:** G2S prob SMEM per stage drops from 9216 B to 128 B, allowing ~89 G2S stages per pipeline (vs 10 without the fix). Critical for the batched accumulation optimization below.

### 10.3 Batched Combine Reduction

**Problem:** The combine kernel's reduction warp group processes source tokens one at a time with per-source synchronization: `mbarrier_wait → barrier → accumulate → barrier → mbarrier_free` for each source. With N~8 source ranks per output token, 2*N barriers per output token dominate latency.

**Fix:** Batch B sources together: wait for B mbarriers sequentially (typically instant since G2S pipelines all reads ahead), then a single barrier, accumulate all B sources from SMEM without interruption, then a single barrier, then batch-free all B slots. Reduces barriers from 2*N to 2*ceil(N/B) per output token.

**Configuration:** Batch size is configurable via `NUM_OF_COMBINE_REDUCE_BATCH_SIZE_API` environment variable. Default: auto (half the G2S pipeline depth to allow G2S/consumer overlap). The value becomes a JIT template parameter `NUM_OF_COMBINE_REDUCE_BATCH_SIZE`.

**Files changed:** `hybrid_ep_backend.cuh` (inter_node_red reduction loop rewritten), `config.cuh` (new config field + env var + auto-resolution), `compiler.cu` (JIT codegen + cache key).

**Impact:** Combine kernel improved from 969 us to 202 us (4.8x) on B300 NVL8 with optimal config.

### 10.4 Dense Routing Map

**Problem:** The scan kernel (metadata preprocessing) takes a boolean routing map `[T, E_total]` where `E_total = E_per_rank * R * N_nodes`. For NVL72: `E_total = 2304`, so each token's row is 2304 bytes with only K=36 non-zero entries. The allgather of this map across R ranks costs `T * R * E_total` bytes (1.2 GB for T=8192, R=72). The scan kernel loads `E_per_rank * R_per_node` bytes per token per step.

**Fix:** Accept a dense `topk_idx [T, K]` of `uint16` (2 bytes per index) instead of the sparse bool map. The scan kernel is templated with `TOPK` parameter: when `TOPK > 0`, per-token routing is computed by range-checking K expert indices against rank boundaries instead of OR-reducing E_per_rank bools.

**API change:** Pass `dense_routing=True` to `dispatch()` or `dispatch_with_permute()`:

```python
# Before (sparse bool, default)
dispatched = buffer.dispatch(hidden=hidden, topk_idx=topk_idx,
    topk_weights=topk_weights, num_of_experts=E_total)

# After (dense routing, new)
dispatched = buffer.dispatch(hidden=hidden, topk_idx=topk_idx,
    topk_weights=topk_weights, num_of_experts=E_total,
    dense_routing=True)
```

**Files changed:** `hybrid_ep_backend.cuh` (scan kernel Steps 0, 2, 3), `config.cuh` (topk field), `executor.cu` (allgather dtype handling), `compiler.cu` + `compiler.cuh` (JIT codegen), `pybind_hybrid_ep.cu` (config binding), `hybrid_ep_buffer.py` (Python API).

**Impact:** Allgather size reduced from T * E_total to T * K * 2 bytes (32x on NVL72 with K=36). Scan kernel per-token load reduced from E*R bytes to K*2 bytes.

### 10.5 Tuning Environment Variables

All combine kernel parameters are configurable via environment variables and become JIT template parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_OF_STAGES_G2S_COMBINE_API` | 10 | G2S SMEM pipeline stages (total, split across pipelines) |
| `NUM_OF_STAGES_S2G_COMBINE_API` | 2 | S2G SMEM pipeline stages |
| `NUM_OF_COMBINE_REDUCE_BATCH_SIZE_API` | 0 (auto) | Batch size for combine reduction. 0 = half pipeline depth |
| `NUM_OF_TOKENS_PER_GROUP_COMBINE_API` | 4 | Output tokens per group assigned to each pipeline |
| `NUM_OF_TOKENS_PER_CHUNK_COMBINE_API` | 64 | Chunk size for inter-rank synchronization |
| `NUM_SMS_COMBINE` | 24 | Number of SMs for combine kernel |
| `NUM_SMS_DISPATCH` | 24 | Number of SMs for dispatch kernel |
| `HYBRID_EP_DEBUG_JIT_CONFIG` | 0 | Set to 1 to print kernel config at JIT compilation time |

**Recommended config for latent MoE (H=512, E=32, TOPK=36) on B300 NVL8:**

```bash
NUM_OF_STAGES_G2S_COMBINE_API=64 NUM_OF_STAGES_S2G_COMBINE_API=8 \
NUM_OF_COMBINE_REDUCE_BATCH_SIZE_API=16 NUM_OF_TOKENS_PER_GROUP_COMBINE_API=1 \
NUM_SMS_DISPATCH=32 NUM_SMS_COMBINE=64
```

Note: When using fused permute-dispatch or fused combine-unpermute, all `NUM_OF_TOKENS_PER_CHUNK_*_API` variables (dispatch, combine, preprocessing) must match.

### 10.6 Performance Summary

**B300 SXM NVL8, H=512, E_per_rank=32, TOPK=36, T=8192 tokens/rank, BF16.**

Original baseline config: `NUM_SMS_DISPATCH=24, NUM_SMS_COMBINE=24`, default G2S/S2G stages.

#### Kernel-only timings (no d2d, no device_sync)

| Kernel | Original (us) | Optimized (us) | Config | Speedup |
|--------|---------------|----------------|--------|---------|
| dispatch (w/ probs) | 227.9 | 102.5 | SMS=32 | 2.2x |
| dispatch (no probs) | — | 94.3 | SMS=32 | — |
| combine (w/ probs) | 969.1 | 299.0 | SMS=32, G2S=36, S2G=4, GROUP=2 | 3.2x |
| combine (w/ probs) | 969.1 | 238.8 | SMS=64, G2S=64, S2G=8, BATCH=32, GROUP=1 | 4.1x |
| combine (no probs) | — | 262.3 | SMS=32, G2S=36, S2G=4, GROUP=2 | — |
| combine (no probs) | — | 201.8 | SMS=64, G2S=64, S2G=8, BATCH=16, GROUP=1 | — |
| fused dispatch+permute | 602.4 | 589.5 | SMS=32 | ~1.0x |
| fused combine+unpermute | 1138.0 | 1110.8 | SMS=32 | ~1.0x |

The original baseline did not separately measure dispatch-no-prob or combine-no-prob; both were measured with `with_probs=True`.

#### Torch API timings (includes d2d, device_sync, kernel)

Config: `SMS_DISPATCH=32, SMS_COMBINE=32, G2S=36, S2G=4, GROUP=2`:

| API | Optimized (us) |
|-----|----------------|
| dispatch (BF16, w/ probs) | 227.0 |
| dispatch (BF16, no probs) | 173.9 |
| combine (w/ probs) | 369.2 |
| combine (no probs) | 308.7 |
| dispatch+permute | 507.1 |
| combine+unpermute | 587.0 |

#### NVLink bandwidth utilization (66.73 MB per direction)

| Kernel | GB/s | % of B300 NVL18 peak (900 GB/s/dir) |
|--------|------|--------------------------------------|
| dispatch (w/ probs) | 651 | 72% |
| dispatch (no probs) | 708 | 79% |
| combine (w/ probs, best) | 275 | 31% |
| combine (no probs, best) | 331 | 37% |

#### Key observations

- **Dispatch** is close to NVLink-bandwidth-limited. The sparse prob optimization eliminated the dominant traffic source (prob vectors), leaving only token data.
- **Combine** is latency-limited by the serial per-output-token reduction (read N source ranks and accumulate sequentially). Batched accumulation and increased G2S pipeline depth improved throughput by reducing barrier overhead, but the fundamental serial dependency remains.
- **Fused paths** (dispatch+permute, combine+unpermute) did not benefit from these optimizations as they are dominated by the permute/unpermute block computation, which is unchanged.
- **GROUP=1 with more SMs** is faster for combine because each SM processes one output token with the full SMEM FIFO (no pipeline splitting), but wastes half the warps. This is a viable tradeoff since the kernel is latency-bound, not compute-bound.