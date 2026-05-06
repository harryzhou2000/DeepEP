// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

// ============================================================================
// Direct-permute address computation (optimized).
//
// Produces direct_write_map[T_per_rank, TOPK]: for each of the local rank's
// tokens and each TOPK routing slot, the absolute row index in the target
// rank's permuted (expert-grouped) output buffer.
//
// Three-kernel pipeline:
//
//   Kernel 1 (count_per_source): One block per source rank. Each block scans
//     its T_per_rank tokens from the allgathered global routing map and counts
//     how many go to each (target_rank, expert) pair on the local node.
//     Output: counts[R_source, R_target * E_per_rank]
//
//   Kernel 2 (compute_metadata): Single block (lightweight). Reads counts from
//     all source ranks, computes:
//       - expert_base[R_target * E]: padded exclusive prefix within each target
//       - my_prefix[R_target * E]: sum of counts from ranks 0..local_rank-1
//       - tokens_per_expert[E]: real count per local expert (unpadded)
//       - padded_tokens_per_expert[E]: padded count per local expert (for GEMM)
//       - overflow_flag: 1 if total padded size > num_permuted_tokens
//       - Initializes position_counters[R_target * E] to 0
//
//   Kernel 3 (assign_positions): Multi-block parallel. Each thread processes
//     a few tokens, uses atomicAdd on GMEM counters (9 KB, L2-resident) to
//     assign unique positions. No ordering needed — positions are unique by
//     atomic guarantee.
//     Output: direct_write_map[T_per_rank, TOPK]
// ============================================================================

#include "direct_permute.cuh"
#include <ATen/cuda/CUDAContext.h>

#define DP_CUDA_CHECK(call)                                                      \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err));                                           \
      abort();                                                                   \
    }                                                                            \
  } while (0)

// ============================================================================
// Kernel 1: count_per_source
// ============================================================================
__global__ void count_per_source_kernel(
    const int16_t* __restrict__ global_routing_map,  // [T_total, TOPK]
    int32_t* __restrict__ counts,                    // [R_per_node, R_per_node * E_per_rank]
    const int T_per_rank,
    const int R_per_node,
    const int E_per_rank,
    const int TOPK,
    const int node_rank) {
  const int source_rank = blockIdx.x;
  const int experts_per_node = R_per_node * E_per_rank;
  const int RE = R_per_node * E_per_rank;

  extern __shared__ int32_t smem_counts[];  // [RE]

  for (int i = threadIdx.x; i < RE; i += blockDim.x) {
    smem_counts[i] = 0;
  }
  __syncthreads();

  const int token_start = source_rank * T_per_rank;
  for (int t = threadIdx.x; t < T_per_rank; t += blockDim.x) {
    const int16_t* topk = global_routing_map + (int64_t)(token_start + t) * TOPK;
    for (int k = 0; k < TOPK; k++) {
      int eg = (int)topk[k];
      if (eg < 0) continue;
      int expert_node = eg / experts_per_node;
      if (expert_node != node_rank) continue;
      int local_eg = eg - node_rank * experts_per_node;
      int target_rank = local_eg / E_per_rank;
      int local_expert = local_eg % E_per_rank;
      atomicAdd_block(&smem_counts[target_rank * E_per_rank + local_expert], 1);
    }
  }
  __syncthreads();

  int32_t* out = counts + (int64_t)source_rank * RE;
  for (int i = threadIdx.x; i < RE; i += blockDim.x) {
    out[i] = smem_counts[i];
  }
}

// ============================================================================
// Kernel 2: compute_metadata
//
// Single block. Reads counts[], computes prefix sums and metadata.
// Writes expert_base[], my_prefix[] to GMEM for Kernel 3.
// Also produces tokens_per_expert, padded_tokens_per_expert, overflow_flag.
// Initializes position_counters to 0.
// ============================================================================
__global__ void compute_metadata_kernel(
    const int32_t* __restrict__ counts,                  // [R_per_node, RE]
    int32_t* __restrict__ expert_base,                   // [RE] — output (for Kernel 3)
    int32_t* __restrict__ my_prefix,                     // [RE] — output (for Kernel 3)
    int32_t* __restrict__ position_counters,             // [RE] — output (init to 0)
    int32_t* __restrict__ tokens_per_expert_out,         // [E_per_rank]
    int64_t* __restrict__ padded_tokens_per_expert_out,  // [E_per_rank]
    int32_t* __restrict__ overflow_flag_out,             // [1]
    const int R_per_node,
    const int E_per_rank,
    const int pad_multiple,
    const int local_rank,
    const int64_t num_permuted_tokens) {

  const int RE = R_per_node * E_per_rank;

  // Phase 1: compute totals, my_prefix, padded counts
  // Use SMEM for intermediate per-target padded counts (needed for prefix sum)
  extern __shared__ int32_t smem[];
  // smem layout: [0..RE): padded_counts_per_target_expert
  int32_t* padded_counts = smem;

  for (int idx = threadIdx.x; idx < RE; idx += blockDim.x) {
    int32_t total = 0;
    int32_t prefix = 0;
    for (int s = 0; s < R_per_node; s++) {
      int32_t c = counts[(int64_t)s * RE + idx];
      if (s < local_rank) prefix += c;
      total += c;
    }
    my_prefix[idx] = prefix;
    position_counters[idx] = 0;

    int32_t padded = (pad_multiple > 0)
                         ? ((total + pad_multiple - 1) / pad_multiple * pad_multiple)
                         : total;
    padded_counts[idx] = padded;

    // Output per-local-expert info (target_rank == local_rank)
    int target_rank = idx / E_per_rank;
    int expert_id = idx % E_per_rank;
    if (target_rank == local_rank) {
      tokens_per_expert_out[expert_id] = total;
      padded_tokens_per_expert_out[expert_id] = (int64_t)padded;
    }
  }
  __syncthreads();

  // Phase 2: exclusive prefix sum across experts within each target rank
  // E_per_rank is small (32), single-thread per target rank
  if (threadIdx.x < R_per_node) {
    int target = threadIdx.x;
    int32_t acc = 0;
    for (int e = 0; e < E_per_rank; e++) {
      int32_t padded = padded_counts[target * E_per_rank + e];
      expert_base[target * E_per_rank + e] = acc;
      acc += padded;
    }
    // Check overflow for local_rank's buffer
    if (target == local_rank && num_permuted_tokens >= 0) {
      overflow_flag_out[0] = (acc > (int32_t)num_permuted_tokens) ? 1 : 0;
    }
  }

  // If num_permuted_tokens < 0, no overflow check needed
  if (threadIdx.x == 0 && num_permuted_tokens < 0) {
    overflow_flag_out[0] = 0;
  }
}

// ============================================================================
// Kernel 3: assign_positions
//
// Multi-block parallel. Scans local_rank's own tokens, uses atomicAdd on
// GMEM position_counters to get unique positions. Counters are L2-resident
// (9 KB for NVL72).
// ============================================================================
__global__ void assign_positions_kernel(
    const int16_t* __restrict__ global_routing_map,  // [T_total, TOPK]
    const int32_t* __restrict__ expert_base,         // [RE]
    const int32_t* __restrict__ my_prefix,           // [RE]
    int32_t* __restrict__ position_counters,         // [RE] — atomicAdd target
    int32_t* __restrict__ direct_write_map,          // [T_per_rank, TOPK]
    const int T_per_rank,
    const int R_per_node,
    const int E_per_rank,
    const int TOPK,
    const int local_rank,
    const int node_rank) {

  const int experts_per_node = R_per_node * E_per_rank;
  const int64_t my_token_start = (int64_t)local_rank * T_per_rank;

  // Grid-stride loop over tokens
  const int total_threads = gridDim.x * blockDim.x;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t = global_tid; t < T_per_rank; t += total_threads) {
    const int16_t* topk = global_routing_map + (my_token_start + t) * TOPK;
    for (int k = 0; k < TOPK; k++) {
      int eg = (int)topk[k];
      if (eg < 0) {
        direct_write_map[(int64_t)t * TOPK + k] = -1;
        continue;
      }
      int expert_node = eg / experts_per_node;
      if (expert_node != node_rank) {
        direct_write_map[(int64_t)t * TOPK + k] = -1;
        continue;
      }
      int local_eg = eg - node_rank * experts_per_node;
      int target_rank = local_eg / E_per_rank;
      int local_expert = local_eg % E_per_rank;
      int idx = target_rank * E_per_rank + local_expert;

      int pos = atomicAdd(&position_counters[idx], 1);
      int dest_row = expert_base[idx] + my_prefix[idx] + pos;
      direct_write_map[(int64_t)t * TOPK + k] = dest_row;
    }
  }
}

// ============================================================================
// Host launcher
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_direct_write_map(
    torch::Tensor global_routing_map,
    int T_per_rank,
    int R_per_node,
    int E_per_rank,
    int TOPK,
    int pad_multiple,
    int local_rank,
    int node_rank,
    int64_t num_permuted_tokens) {

  TORCH_CHECK(global_routing_map.dtype() == torch::kInt16,
              "global_routing_map must be int16");
  TORCH_CHECK(global_routing_map.is_cuda(), "global_routing_map must be on CUDA");
  TORCH_CHECK(global_routing_map.size(0) == (int64_t)T_per_rank * R_per_node,
              "global_routing_map shape[0] must be T_per_rank * R_per_node");
  TORCH_CHECK(global_routing_map.size(1) == TOPK,
              "global_routing_map shape[1] must be TOPK");
  TORCH_CHECK(local_rank >= 0 && local_rank < R_per_node, "invalid local_rank");

  auto stream = at::cuda::getCurrentCUDAStream();
  const int RE = R_per_node * E_per_rank;

  // Allocate intermediate and output tensors
  auto counts = torch::empty({R_per_node, RE},
                             torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto expert_base = torch::empty({RE},
                                  torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto my_prefix = torch::empty({RE},
                                torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto position_counters = torch::empty({RE},
                                        torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto direct_write_map = torch::full({T_per_rank, TOPK}, -1,
                                      torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto tokens_per_expert = torch::empty({E_per_rank},
                                        torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto padded_tokens_per_expert = torch::empty({E_per_rank},
                                               torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto overflow_flag = torch::zeros({1},
                                    torch::dtype(torch::kInt32).device(torch::kCUDA));

  // Kernel 1: count per source rank
  {
    const int block_size = 256;
    const int smem_size = RE * sizeof(int32_t);
    count_per_source_kernel<<<R_per_node, block_size, smem_size, stream>>>(
        global_routing_map.data_ptr<int16_t>(),
        counts.data_ptr<int32_t>(),
        T_per_rank, R_per_node, E_per_rank, TOPK, node_rank);
    DP_CUDA_CHECK(cudaGetLastError());
  }

  // Kernel 2: compute metadata (expert_base, my_prefix, tokens_per_expert, etc.)
  {
    const int block_size = 256;
    const int smem_size = RE * sizeof(int32_t);
    compute_metadata_kernel<<<1, block_size, smem_size, stream>>>(
        counts.data_ptr<int32_t>(),
        expert_base.data_ptr<int32_t>(),
        my_prefix.data_ptr<int32_t>(),
        position_counters.data_ptr<int32_t>(),
        tokens_per_expert.data_ptr<int32_t>(),
        padded_tokens_per_expert.data_ptr<int64_t>(),
        overflow_flag.data_ptr<int32_t>(),
        R_per_node, E_per_rank, pad_multiple, local_rank, num_permuted_tokens);
    DP_CUDA_CHECK(cudaGetLastError());
  }

  // Kernel 3: assign positions (multi-block, GMEM atomics on L2-resident counters)
  {
    const int block_size = 256;
    // Use enough blocks to keep GPU busy but not too many for atomic contention
    // Target: ~4 tokens per thread for good throughput
    const int num_blocks = std::min(32, (T_per_rank + block_size - 1) / block_size);
    assign_positions_kernel<<<num_blocks, block_size, 0, stream>>>(
        global_routing_map.data_ptr<int16_t>(),
        expert_base.data_ptr<int32_t>(),
        my_prefix.data_ptr<int32_t>(),
        position_counters.data_ptr<int32_t>(),
        direct_write_map.data_ptr<int32_t>(),
        T_per_rank, R_per_node, E_per_rank, TOPK, local_rank, node_rank);
    DP_CUDA_CHECK(cudaGetLastError());
  }

  return std::make_tuple(direct_write_map, tokens_per_expert,
                         padded_tokens_per_expert, overflow_flag);
}
