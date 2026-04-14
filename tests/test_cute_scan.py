# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Test the CuTe DSL scan kernel against a vectorized CPU reference.

Usage:
    python tests/test_cute_scan.py
    python tests/test_cute_scan.py --num-tokens 256 --num-experts 4 --num-ranks 8 --topk 4
"""

import argparse
import sys
import time
import torch

try:
    from deep_ep.cute_kernels.scan import metadata_preprocess_cute
    HAS_CUTE_DSL = True
except ImportError as e:
    HAS_CUTE_DSL = False
    CUTE_IMPORT_ERROR = str(e)


def reference_scan_vectorized(
    routing_data: torch.Tensor,
    num_tokens_per_rank: int,
    E: int,
    R: int,
    N: int,
    node_rank: int,
    local_rank: int,
    topk: int,
) -> dict:
    """
    Vectorized CPU reference for the scan kernel.

    Returns sparse_to_dense_map and num_dispatched_tokens.
    """
    total_tokens = num_tokens_per_rank * R * N
    EPN = E * R  # experts per node

    # --- Compute per-rank routing flags [total_tokens, R] ---
    if topk > 0:
        # routing_data: [total_tokens, topk] int16
        idx = routing_data.long()  # [T, K]
        node_start = node_rank * EPN
        # Mask: expert is on local node and not dropped (-1)
        on_node = (idx >= node_start) & (idx < node_start + EPN)  # [T, K]
        local_eid = (idx - node_start).clamp(min=0)  # [T, K]
        rank_of_expert = local_eid // E  # [T, K]
        # For each rank r, check if any topk entry maps to it
        # rank_of_expert is in [0, R) when on_node is True
        needed = torch.zeros(total_tokens, R, dtype=torch.bool)
        for r in range(R):
            needed[:, r] = ((rank_of_expert == r) & on_node).any(dim=1)
    else:
        # routing_data: [total_tokens, E * R * N] bool
        rd = routing_data.view(total_tokens, N, R, E)
        # Slice to local node, any expert per rank
        node_slice = rd[:, node_rank, :, :]  # [T, R, E]
        needed = node_slice.any(dim=2)  # [T, R]

    # --- Exclusive prefix sum per rank ---
    # cumsum along token dimension, then shift right by 1
    needed_int = needed.int()  # [T, R]
    cumsum = needed_int.cumsum(dim=0)  # [T, R]
    # Exclusive: shift down by 1, first element is 0
    ex_prefix = torch.zeros_like(cumsum)
    ex_prefix[1:] = cumsum[:-1]

    # --- Build sparse_to_dense_map [tokens_per_rank * N, R] ---
    # Only for tokens owned by local_rank
    sparse_to_dense_map = torch.full(
        (num_tokens_per_rank * N, R), -1, dtype=torch.int32,
    )

    for token_id in range(total_tokens):
        token_node = token_id // (num_tokens_per_rank * R)
        token_lr = (token_id % (num_tokens_per_rank * R)) // num_tokens_per_rank
        token_lid = token_id % num_tokens_per_rank

        if token_lr == local_rank:
            row = token_node * num_tokens_per_rank + token_lid
            for r in range(R):
                if needed[token_id, r]:
                    sparse_to_dense_map[row, r] = ex_prefix[token_id, r].item()

    # num_dispatched = total tokens going to local_rank
    nd = cumsum[-1, local_rank].item() if total_tokens > 0 else 0

    return {
        "sparse_to_dense_map": sparse_to_dense_map,
        "num_dispatched_tokens": nd,
    }


def test_cute_scan(
    num_tokens: int = 32,
    E: int = 2,
    R: int = 2,
    topk: int = 2,
    N: int = 1,
    node_rank: int = 0,
    local_rank: int = 0,
    num_blocks: int = 2,
    num_threads: int = 64,
):
    """Test CuTe DSL scan against vectorized CPU reference."""
    if not HAS_CUTE_DSL:
        print(f"SKIP: CuTe DSL not available: {CUTE_IMPORT_ERROR}")
        return False

    device = torch.device("cuda:0")
    total_experts = E * R * N
    total_tokens = num_tokens * R * N

    print(f"\nTest: T={num_tokens}, E={E}, R={R}, K={topk}, N={N}, "
          f"blocks={num_blocks}, threads={num_threads}")

    # Generate routing data
    if topk > 0:
        routing_data = torch.stack([
            torch.randperm(total_experts, device="cpu")[:topk]
            for _ in range(total_tokens)
        ]).to(torch.int16)
    else:
        routing_data = torch.zeros(total_tokens, total_experts, dtype=torch.bool)
        for t in range(total_tokens):
            experts = torch.randperm(total_experts)[:topk if topk > 0 else 2]
            routing_data[t, experts] = True

    # CPU reference
    t0 = time.time()
    ref = reference_scan_vectorized(
        routing_data.cpu(), num_tokens, E, R, N, node_rank, local_rank, topk,
    )
    ref_time = time.time() - t0
    print(f"  CPU ref: {ref_time*1000:.1f}ms, num_dispatched={ref['num_dispatched_tokens']}")

    # CuTe DSL kernel
    routing_gpu = routing_data.to(device)
    t0 = time.time()
    result = metadata_preprocess_cute(
        routing_data=routing_gpu,
        num_of_tokens_per_rank=num_tokens,
        num_of_experts_per_rank=E,
        num_of_ranks_per_node=R,
        num_of_nodes=N,
        node_rank=node_rank,
        local_rank=local_rank,
        topk=topk,
        num_blocks=num_blocks,
        num_threads=num_threads,
    )
    torch.cuda.synchronize()
    gpu_time = time.time() - t0
    print(f"  GPU (incl compile): {gpu_time*1000:.1f}ms")

    # Compare
    s2d_cute = result["sparse_to_dense_map"].cpu()
    s2d_ref = ref["sparse_to_dense_map"]
    s2d_match = torch.equal(s2d_cute, s2d_ref)

    nd_cute = result["num_dispatched_tokens"].cpu().item()
    nd_ref = ref["num_dispatched_tokens"]
    nd_match = (nd_cute == nd_ref)

    if s2d_match and nd_match:
        print(f"  PASS: num_dispatched={nd_cute}")
        return True
    else:
        if not s2d_match:
            mismatches = (s2d_cute != s2d_ref).sum().item()
            print(f"  FAIL: s2d_map {mismatches}/{s2d_cute.numel()} mismatches")
            idx = (s2d_cute != s2d_ref).nonzero()[:5]
            for i in range(min(5, len(idx))):
                r, c = idx[i].tolist()
                print(f"    [{r},{c}]: cute={s2d_cute[r,c].item()}, ref={s2d_ref[r,c].item()}")
        if not nd_match:
            print(f"  FAIL: num_dispatched cute={nd_cute}, ref={nd_ref}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=2, help="Experts per rank")
    parser.add_argument("--num-ranks", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=64)
    args = parser.parse_args()

    passed = 0
    failed = 0

    # Smallest possible test first (fast compilation)
    configs = [
        # (T, E, R, K, blocks, threads)
        (32, 2, 2, 2, 2, 64),   # tiny
        (args.num_tokens, args.num_experts, args.num_ranks, args.topk,
         args.num_blocks, args.num_threads),
    ]

    for T, E, R, K, B, THR in configs:
        if test_cute_scan(T, E, R, K, num_blocks=B, num_threads=THR):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
