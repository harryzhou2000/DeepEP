# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Test the CuTe DSL dispatch kernel against a CPU reference.

Single-GPU test: source and destination are both in local memory.
Validates that the scatter operation produces correct results.

Usage:
    python tests/test_cute_dispatch.py
"""

import argparse
import sys
import time
import torch

try:
    from deep_ep.cute_kernels.dispatch import dispatch_cute
    HAS_CUTE_DSL = True
except ImportError as e:
    HAS_CUTE_DSL = False
    CUTE_IMPORT_ERROR = str(e)


def reference_dispatch(
    hidden: torch.Tensor,      # [T, H] bf16
    probs: torch.Tensor,       # [T, E*R] f32 or None
    s2d_map: torch.Tensor,     # [T, R] int32
    rdma_map: torch.Tensor,    # [T] bool
    output_tokens: list,       # R tensors [max_out, H]
    output_probs: list,        # R tensors [max_out, E*R] or None
    R: int,
) -> None:
    """CPU reference: scatter tokens to per-rank output buffers."""
    T, H = hidden.shape
    for t in range(T):
        if not rdma_map[t].item():
            continue
        for r in range(R):
            dst_idx = s2d_map[t, r].item()
            if dst_idx >= 0:
                output_tokens[r][dst_idx] = hidden[t]
                if probs is not None and output_probs is not None:
                    output_probs[r][dst_idx] = probs[t]


def test_dispatch(
    T=256, H=512, R=8, E=4, topk=4, max_out=256,
    num_blocks=4,
):
    """Test CuTe DSL dispatch against CPU reference."""
    if not HAS_CUTE_DSL:
        print(f"SKIP: CuTe DSL not available: {CUTE_IMPORT_ERROR}")
        return False

    device = torch.device("cuda:0")
    total_experts = E * R

    print(f"\nTest dispatch: T={T}, H={H}, R={R}, E={E}, K={topk}, max_out={max_out}")

    # Generate test data
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    probs = torch.randn(T, total_experts, dtype=torch.float32, device=device)

    # Generate routing: each token goes to ~topk ranks
    s2d_map = torch.full((T, R), -1, dtype=torch.int32, device=device)
    rdma_map = torch.zeros(T, dtype=torch.bool, device=device)

    # Simple routing: assign contiguous output positions
    rank_counters = [0] * R
    for t in range(T):
        # Pick topk random ranks for this token
        ranks = torch.randperm(R, device=device)[:topk]
        for r_idx in ranks:
            r = r_idx.item()
            if rank_counters[r] < max_out:
                s2d_map[t, r] = rank_counters[r]
                rank_counters[r] += 1
                rdma_map[t] = True

    # Allocate output buffers
    out_tokens_ref = [torch.zeros(max_out, H, dtype=torch.bfloat16, device=device) for _ in range(R)]
    out_probs_ref = [torch.zeros(max_out, total_experts, dtype=torch.float32, device=device) for _ in range(R)]
    out_tokens_cute = [torch.zeros(max_out, H, dtype=torch.bfloat16, device=device) for _ in range(R)]
    out_probs_cute = [torch.zeros(max_out, total_experts, dtype=torch.float32, device=device) for _ in range(R)]

    # CPU reference
    t0 = time.time()
    reference_dispatch(hidden, probs, s2d_map, rdma_map,
                       out_tokens_ref, out_probs_ref, R)
    ref_ms = (time.time() - t0) * 1000
    print(f"  CPU ref: {ref_ms:.1f}ms")

    # CuTe DSL dispatch
    t0 = time.time()
    dispatch_cute(
        hidden=hidden,
        probs=probs,
        sparse_to_dense_map=s2d_map,
        rdma_to_attn_map=rdma_map,
        output_tokens=out_tokens_cute,
        output_probs=out_probs_cute,
        num_ranks=R,
        num_experts_per_rank=E,
        num_blocks=num_blocks,
    )
    torch.cuda.synchronize()
    gpu_ms = (time.time() - t0) * 1000
    print(f"  GPU (incl compile): {gpu_ms:.1f}ms")

    # Compare tokens
    token_match = True
    for r in range(R):
        if not torch.equal(out_tokens_ref[r], out_tokens_cute[r]):
            mismatches = (out_tokens_ref[r] != out_tokens_cute[r]).sum().item()
            print(f"  FAIL: rank {r} tokens: {mismatches}/{out_tokens_ref[r].numel()} mismatches")
            # Show first mismatch
            idx = (out_tokens_ref[r] != out_tokens_cute[r]).nonzero()
            if len(idx) > 0:
                row, col = idx[0].tolist()
                print(f"    [{row},{col}]: ref={out_tokens_ref[r][row,col].item():.4f}, "
                      f"cute={out_tokens_cute[r][row,col].item():.4f}")
            token_match = False

    # Compare probs
    prob_match = True
    for r in range(R):
        if not torch.equal(out_probs_ref[r], out_probs_cute[r]):
            mismatches = (out_probs_ref[r] != out_probs_cute[r]).sum().item()
            print(f"  FAIL: rank {r} probs: {mismatches}/{out_probs_ref[r].numel()} mismatches")
            prob_match = False

    if token_match and prob_match:
        print(f"  PASS: all ranks match")
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=4)
    args = parser.parse_args()

    passed = 0
    failed = 0

    # Small test first
    if test_dispatch(T=32, H=64, R=2, E=2, topk=2, max_out=32, num_blocks=2):
        passed += 1
    else:
        failed += 1

    # Larger test
    if test_dispatch(
        T=args.num_tokens, H=args.hidden_dim, R=args.num_ranks,
        E=args.num_experts, topk=args.topk, max_out=args.num_tokens,
        num_blocks=args.num_blocks,
    ):
        passed += 1
    else:
        failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
