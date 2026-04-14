# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
End-to-end test: dispatch → expert (identity) → combine.

The dispatch scatters tokens to per-rank buffers, the expert is the
identity op (multiply by num_experts_hitting_this_rank), and the combine
gathers and accumulates. The result should be input * TOPK.

Single-GPU test with simulated per-rank buffers.

Usage:
    python tests/test_cute_dispatch_combine.py
"""

import argparse
import sys
import time
import torch

try:
    from deep_ep.cute_kernels.dispatch import dispatch_cute
    from deep_ep.cute_kernels.combine import combine_cute
    HAS_CUTE_DSL = True
except ImportError as e:
    HAS_CUTE_DSL = False
    CUTE_IMPORT_ERROR = str(e)


def test_dispatch_combine(
    T=64, H=128, R=4, E=2, topk=2, max_buf=128,
    num_blocks=4, atol=5e-3,
):
    """Test dispatch→identity→combine round trip."""
    if not HAS_CUTE_DSL:
        print(f"SKIP: CuTe DSL not available: {CUTE_IMPORT_ERROR}")
        return False

    device = torch.device("cuda:0")
    total_experts = E * R

    print(f"\nTest dispatch→combine: T={T}, H={H}, R={R}, E={E}, K={topk}")

    # Input data
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    probs = torch.rand(T, total_experts, dtype=torch.float32, device=device)

    # Build routing: each token goes to exactly topk random ranks
    s2d_map = torch.full((T, R), -1, dtype=torch.int32, device=device)
    rdma_map = torch.zeros(T, dtype=torch.bool, device=device)

    rank_counters = [0] * R
    for t in range(T):
        ranks = torch.randperm(R)[:topk]
        for r_idx in ranks:
            r = r_idx.item()
            if rank_counters[r] < max_buf:
                s2d_map[t, r] = rank_counters[r]
                rank_counters[r] += 1
                rdma_map[t] = True

    # Allocate per-rank dispatch output buffers
    dispatch_tokens = [torch.zeros(max_buf, H, dtype=torch.bfloat16, device=device) for _ in range(R)]
    dispatch_probs = [torch.zeros(max_buf, total_experts, dtype=torch.float32, device=device) for _ in range(R)]

    # Dispatch
    t0 = time.time()
    dispatch_cute(
        hidden=hidden,
        probs=probs,
        sparse_to_dense_map=s2d_map,
        rdma_to_attn_map=rdma_map,
        output_tokens=dispatch_tokens,
        output_probs=dispatch_probs,
        num_ranks=R,
        num_experts_per_rank=E,
        num_blocks=num_blocks,
    )
    torch.cuda.synchronize()
    dispatch_ms = (time.time() - t0) * 1000
    print(f"  Dispatch: {dispatch_ms:.1f}ms (incl compile)")

    # "Expert" identity op: just keep the tokens as-is.
    # The combine will accumulate all copies. Each token is dispatched to
    # topk ranks, so after combine, output = sum over topk copies = topk * input.

    # Combine
    t0 = time.time()
    combined_tokens, combined_probs = combine_cute(
        input_tokens=dispatch_tokens,
        input_probs=dispatch_probs,
        sparse_to_dense_map=s2d_map,
        rdma_to_attn_map=rdma_map,
        num_tokens=T,
        num_ranks=R,
        num_experts_per_rank=E,
        hidden_dim=H,
        num_blocks=num_blocks,
    )
    torch.cuda.synchronize()
    combine_ms = (time.time() - t0) * 1000
    print(f"  Combine: {combine_ms:.1f}ms (incl compile)")

    # Verify: combined_tokens should be topk * hidden
    # (each token is dispatched to exactly topk ranks, combine sums them)
    expected = hidden.float() * topk
    actual = combined_tokens.float()

    # Only check tokens that were actually routed
    routed_mask = rdma_map.cpu()
    max_err = 0.0
    for t in range(T):
        if routed_mask[t]:
            err = (actual[t] - expected[t]).abs().max().item()
            max_err = max(max_err, err)

    token_pass = max_err < atol
    print(f"  Token max error: {max_err:.6f} (atol={atol})")

    # Verify probs: combined should be topk * probs
    if combined_probs is not None:
        expected_probs = probs * topk
        prob_max_err = 0.0
        for t in range(T):
            if routed_mask[t]:
                err = (combined_probs[t] - expected_probs[t]).abs().max().item()
                prob_max_err = max(prob_max_err, err)
        prob_pass = prob_max_err < 1e-4
        print(f"  Prob max error: {prob_max_err:.6f}")
    else:
        prob_pass = True

    if token_pass and prob_pass:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-ranks", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=4)
    args = parser.parse_args()

    passed = 0
    failed = 0

    # Small test
    if test_dispatch_combine(T=16, H=64, R=2, E=2, topk=2, max_buf=32, num_blocks=2):
        passed += 1
    else:
        failed += 1

    # Larger test
    if test_dispatch_combine(
        T=args.num_tokens, H=args.hidden_dim, R=args.num_ranks,
        E=args.num_experts, topk=args.topk, max_buf=args.num_tokens * 2,
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
