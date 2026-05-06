// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once
#include <torch/torch.h>
#include <cuda_runtime.h>

/**
 * @brief Compute direct_write_map: for each of the local rank's tokens and each TOPK slot,
 * the absolute row index in the target rank's permuted (expert-grouped) output buffer.
 *
 * This enables the dispatch S2G warp to write directly to the final expert-grouped
 * positions over NVLink, eliminating the intermediate staging buffer and permute kernel.
 *
 * Three-kernel pipeline:
 *   Kernel 1 (count_per_source): One block per source rank. Counts how many tokens
 *     each source sends to each (target_rank, expert) pair on the local node.
 *   Kernel 2 (compute_metadata): Single block, computes expert_base (padded prefix),
 *     rank_prefix for local_rank, tokens_per_expert, padded_tokens_per_expert,
 *     overflow_flag. Initializes GMEM atomic counters.
 *   Kernel 3 (assign_positions): Multi-block parallel scan of own tokens.
 *     Uses GMEM atomicAdd on L2-resident counters (9 KB) for position assignment.
 *
 * @param global_routing_map   [T_per_rank * R_per_node, TOPK] int16 — allgathered dense routing
 * @param T_per_rank           Tokens per rank
 * @param R_per_node           Ranks per NVLink node
 * @param E_per_rank           Local experts per rank
 * @param TOPK                 Top-k routing width
 * @param pad_multiple         Expert region padding alignment (0 = no padding)
 * @param local_rank           This rank's index within the node
 * @param node_rank            This node's index (for multi-node expert ID decode)
 * @param num_permuted_tokens  Max buffer size (-1 = unlimited)
 * @return Tuple of:
 *   - direct_write_map:        [T_per_rank, TOPK] int32 — absolute dest row per topk slot
 *   - tokens_per_expert:       [E_per_rank] int32 — real (unpadded) count per local expert
 *   - padded_tokens_per_expert:[E_per_rank] int64 — padded counts for grouped GEMM
 *   - overflow_flag:           [1] int32 — 1 if buffer exceeds num_permuted_tokens
 */
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
    int64_t num_permuted_tokens = -1);
