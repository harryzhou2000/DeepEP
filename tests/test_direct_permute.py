#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Standalone test for compute_direct_write_map.

Verifies correctness by comparing CUDA kernel output against a pure-Python
reference that computes the same direct-write addressing map.

Run: python tests/test_direct_permute.py
     python tests/test_direct_permute.py --benchmark

No multi-GPU or distributed required — simulates multiple ranks on a single GPU.

== Architecture ==

compute_direct_write_map produces direct_write_map[T_per_rank, TOPK]:
for each of the local rank's tokens and each TOPK slot, the absolute row
index in the target rank's permuted (expert-grouped) output buffer.

This enables the dispatch S2G warp to NVLink-write directly to the final
expert-grouped positions, eliminating the staging buffer + permute kernel.

Input:
  global_routing_map [T_per_rank * R_per_node, TOPK] int16
    — allgathered dense routing map (global expert IDs per token)

Outputs:
  direct_write_map [T_per_rank, TOPK] int32
    — absolute position in target rank's permuted buffer (-1 if off-node)
  tokens_per_expert [E_per_rank] int32
    — unpadded real count per local expert
  padded_tokens_per_expert [E_per_rank] int64
    — padded count per local expert (for grouped GEMM consumption)
  overflow_flag [1] int32
    — 1 if total padded buffer exceeds num_permuted_tokens limit

== Three-Kernel Pipeline ==

Kernel 1 (count_per_source): One block per source rank (R_per_node blocks).
  Each block scans its T_per_rank tokens, counts per (target_rank, expert).
  Output: counts[R_source, R_target * E_per_rank]
  Bottleneck: SMEM atomics over T_per_rank * TOPK entries per block.
  Current: ~184 us for T=8192, K=36 (dominates total time).

Kernel 2 (compute_metadata): Single block, lightweight.
  Reads counts[], computes expert_base (padded prefix per target rank),
  my_prefix (rank offset for local_rank), tokens_per_expert,
  padded_tokens_per_expert, overflow_flag. Initializes GMEM counters.
  Current: 2-30 us depending on R_per_node.

Kernel 3 (assign_positions): Multi-block (32 blocks).
  Grid-stride loop over own T_per_rank tokens. atomicAdd on GMEM counters
  (L2-resident, 9 KB for NVL72) for unique position assignment.
  Current: 22-69 us.

== Performance (B300 NVL8, T=8192, E=32, K=36) ==

  count_per_source:  184 us  (dominates — optimization TODO)
  assign_positions:   69 us
  compute_metadata:    3 us
  Total kernel:      256 us

Optimization opportunity for Kernel 1: use multiple blocks per source rank
with warp-level reduction, or vectorized loads with ballot-based counting.
"""

import torch
import time
import argparse


def reference_direct_write_map(
    global_routing_map: torch.Tensor,
    T_per_rank: int,
    R_per_node: int,
    E_per_rank: int,
    TOPK: int,
    pad_multiple: int,
    local_rank: int,
    node_rank: int,
):
    """
    Pure-Python reference implementation.

    Returns:
        direct_write_map: [T_per_rank, TOPK] int32
        tokens_per_expert: [E_per_rank] int32
    """
    experts_per_node = R_per_node * E_per_rank
    T_total = T_per_rank * R_per_node

    # Step 1: Count tokens per (source_rank, target_rank, expert)
    # counts[source][target * E + expert]
    counts = torch.zeros(R_per_node, R_per_node * E_per_rank, dtype=torch.int32)

    routing = global_routing_map.cpu().to(torch.int32)  # [T_total, TOPK]
    for s in range(R_per_node):
        for t in range(T_per_rank):
            global_t = s * T_per_rank + t
            for k in range(TOPK):
                eg = routing[global_t, k].item()
                if eg < 0:
                    continue
                expert_node = eg // experts_per_node
                if expert_node != node_rank:
                    continue
                local_eg = eg - node_rank * experts_per_node
                target_rank = local_eg // E_per_rank
                local_expert = local_eg % E_per_rank
                counts[s, target_rank * E_per_rank + local_expert] += 1

    # Step 2: Compute expert_base (padded prefix) and my_prefix
    RE = R_per_node * E_per_rank
    expert_base = torch.zeros(RE, dtype=torch.int32)
    my_prefix = torch.zeros(RE, dtype=torch.int32)
    tokens_per_expert = torch.zeros(E_per_rank, dtype=torch.int32)

    for idx in range(RE):
        total = counts[:, idx].sum().item()
        prefix = counts[:local_rank, idx].sum().item()
        my_prefix[idx] = prefix

        padded = total if pad_multiple <= 0 else ((total + pad_multiple - 1) // pad_multiple * pad_multiple)
        expert_base[idx] = padded  # temporarily padded count

        target_rank = idx // E_per_rank
        expert_id = idx % E_per_rank
        if target_rank == local_rank:
            tokens_per_expert[expert_id] = total

    # Convert expert_base to exclusive prefix sum within each target rank
    for target in range(R_per_node):
        acc = 0
        for e in range(E_per_rank):
            padded = expert_base[target * E_per_rank + e].item()
            expert_base[target * E_per_rank + e] = acc
            acc += padded

    # Step 3: Scan own tokens, assign positions
    direct_write_map = torch.full((T_per_rank, TOPK), -1, dtype=torch.int32)
    local_count = torch.zeros(RE, dtype=torch.int32)

    for t in range(T_per_rank):
        global_t = local_rank * T_per_rank + t
        for k in range(TOPK):
            eg = routing[global_t, k].item()
            if eg < 0:
                continue
            expert_node = eg // experts_per_node
            if expert_node != node_rank:
                continue
            local_eg = eg - node_rank * experts_per_node
            target_rank = local_eg // E_per_rank
            local_expert = local_eg % E_per_rank
            idx = target_rank * E_per_rank + local_expert

            pos = local_count[idx].item()
            local_count[idx] += 1
            direct_write_map[t, k] = expert_base[idx].item() + my_prefix[idx].item() + pos

    return direct_write_map, tokens_per_expert


def generate_routing_map(T_per_rank, R_per_node, E_per_rank, TOPK, node_rank=0):
    """Generate a random dense routing map (simulating allgathered result)."""
    T_total = T_per_rank * R_per_node
    num_experts_total = E_per_rank * R_per_node  # single-node for now

    # Each token picks TOPK distinct experts
    routing = torch.zeros(T_total, TOPK, dtype=torch.int16, device="cuda")
    for t in range(T_total):
        experts = torch.randperm(num_experts_total, device="cuda")[:TOPK]
        # Add node_rank offset to make them global IDs
        experts = experts + node_rank * num_experts_total
        routing[t, :] = experts.to(torch.int16)

    return routing


def verify_properties(direct_write_map, tokens_per_expert, global_routing_map,
                      T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank):
    """Verify structural properties of the direct_write_map."""
    experts_per_node = R_per_node * E_per_rank

    # 1. All non-(-1) values should be non-negative
    valid = direct_write_map[direct_write_map >= 0]
    assert (valid >= 0).all(), "Found negative positions in valid entries"

    # 2. Positions should be unique PER TARGET RANK (each target has its own buffer)
    routing_cpu = global_routing_map.cpu().to(torch.int32)
    my_start = local_rank * T_per_rank
    for target in range(R_per_node):
        # Find all entries destined for this target rank
        target_positions = []
        for t in range(T_per_rank):
            for k in range(TOPK):
                eg = routing_cpu[my_start + t, k].item()
                if eg < 0:
                    continue
                if eg // experts_per_node != node_rank:
                    continue
                local_eg = eg - node_rank * experts_per_node
                t_rank = local_eg // E_per_rank
                if t_rank == target:
                    pos = direct_write_map[t, k].item()
                    if pos >= 0:
                        target_positions.append(pos)
        if target_positions:
            positions_t = torch.tensor(target_positions, dtype=torch.int32)
            assert positions_t.unique().numel() == positions_t.numel(), \
                f"Duplicate positions for target_rank={target}: {positions_t.numel()} entries but {positions_t.unique().numel()} unique"

    # 3. Check valid entry count
    print(f"  Properties OK: {valid.numel()} valid entries, unique per target rank, "
          f"max_pos={valid.max().item() if valid.numel() > 0 else 'N/A'}")


def test_correctness(T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank, num_permuted_tokens=-1):
    """Test CUDA implementation against Python reference."""
    import hybrid_ep_cpp

    print(f"\nTest: T={T_per_rank}, R={R_per_node}, E={E_per_rank}, K={TOPK}, "
          f"pad={pad_multiple}, local_rank={local_rank}, node_rank={node_rank}, "
          f"max_buf={num_permuted_tokens}")

    routing = generate_routing_map(T_per_rank, R_per_node, E_per_rank, TOPK, node_rank)

    # CUDA implementation
    direct_write_map_cuda, tokens_per_expert_cuda, padded_tpe_cuda, overflow_cuda = \
        hybrid_ep_cpp.compute_direct_write_map(
            routing, T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple,
            local_rank, node_rank, num_permuted_tokens
        )
    torch.cuda.synchronize()

    # Python reference
    direct_write_map_ref, tokens_per_expert_ref = reference_direct_write_map(
        routing, T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank
    )

    # Compare tokens_per_expert
    tpe_cuda = tokens_per_expert_cuda.cpu()
    assert torch.equal(tpe_cuda, tokens_per_expert_ref), \
        f"tokens_per_expert mismatch:\n  CUDA: {tpe_cuda}\n  Ref:  {tokens_per_expert_ref}"
    print(f"  tokens_per_expert: PASS")

    # Verify padded_tokens_per_expert
    padded_ref = torch.zeros(E_per_rank, dtype=torch.int64)
    for e in range(E_per_rank):
        v = tokens_per_expert_ref[e].item()
        padded_ref[e] = v if pad_multiple <= 0 else ((v + pad_multiple - 1) // pad_multiple * pad_multiple)
    padded_cuda = padded_tpe_cuda.cpu()
    assert torch.equal(padded_cuda, padded_ref), \
        f"padded_tokens_per_expert mismatch:\n  CUDA: {padded_cuda}\n  Ref:  {padded_ref}"
    print(f"  padded_tokens_per_expert: PASS")

    # Verify overflow_flag
    total_padded = padded_ref.sum().item()
    expected_overflow = 1 if (num_permuted_tokens >= 0 and total_padded > num_permuted_tokens) else 0
    actual_overflow = overflow_cuda.cpu().item()
    assert actual_overflow == expected_overflow, \
        f"overflow_flag mismatch: CUDA={actual_overflow}, expected={expected_overflow} " \
        f"(total_padded={total_padded}, limit={num_permuted_tokens})"
    print(f"  overflow_flag: PASS (total_padded={total_padded}, flag={actual_overflow})")

    # Compare direct_write_map
    dm_cuda = direct_write_map_cuda.cpu()
    dm_ref = direct_write_map_ref

    # First check: same number of valid entries
    valid_cuda = (dm_cuda >= 0).sum().item()
    valid_ref = (dm_ref >= 0).sum().item()
    assert valid_cuda == valid_ref, f"Valid entry count mismatch: CUDA={valid_cuda}, ref={valid_ref}"

    # Second check: same -1 pattern (same slots are invalid)
    assert torch.equal(dm_cuda == -1, dm_ref == -1), "Mismatch in which slots are -1 (off-node)"

    # Third check: for each (target_rank, expert), the set of assigned positions should match.
    experts_per_node = R_per_node * E_per_rank
    routing_cpu = routing.cpu().to(torch.int32)
    my_start = local_rank * T_per_rank

    for target in range(R_per_node):
        for expert in range(E_per_rank):
            target_expert_global = node_rank * experts_per_node + target * E_per_rank + expert
            mask = torch.zeros(T_per_rank, TOPK, dtype=torch.bool)
            for t in range(T_per_rank):
                for k in range(TOPK):
                    if routing_cpu[my_start + t, k].item() == target_expert_global:
                        mask[t, k] = True

            if not mask.any():
                continue

            positions_cuda = dm_cuda[mask].sort()[0]
            positions_ref = dm_ref[mask].sort()[0]
            assert torch.equal(positions_cuda, positions_ref), \
                f"Position set mismatch for target_rank={target}, expert={expert}:\n" \
                f"  CUDA: {positions_cuda.tolist()}\n  Ref:  {positions_ref.tolist()}"

    print(f"  direct_write_map: PASS (position sets match for all target×expert pairs)")
    verify_properties(dm_cuda, tpe_cuda, routing, T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank)


def test_benchmark(T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank, warmup=20, iters=100):
    """Benchmark the CUDA implementation."""
    import hybrid_ep_cpp

    routing = generate_routing_map(T_per_rank, R_per_node, E_per_rank, TOPK, node_rank)

    # Warmup
    for _ in range(warmup):
        hybrid_ep_cpp.compute_direct_write_map(
            routing, T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank
        )
    torch.cuda.synchronize()

    # Wall-clock benchmark
    start = time.perf_counter()
    for _ in range(iters):
        hybrid_ep_cpp.compute_direct_write_map(
            routing, T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1e6  # microseconds

    # Kineto benchmark (GPU kernel time only)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(10):
            hybrid_ep_cpp.compute_direct_write_map(
                routing, T_per_rank, R_per_node, E_per_rank, TOPK, pad_multiple, local_rank, node_rank
            )
        torch.cuda.synchronize()

    # Show all CUDA kernels with breakdown
    all_cuda = [e for e in prof.key_averages() if e.device_time_total > 0]
    all_cuda.sort(key=lambda e: e.device_time_total, reverse=True)
    total_us = sum(e.device_time_total for e in all_cuda) / 10.0
    print(f"  Benchmark: wall={elapsed:.1f} us, GPU total={total_us:.1f} us "
          f"(T={T_per_rank}, R={R_per_node}, E={E_per_rank}, K={TOPK})")
    for e in all_cuda[:8]:
        print(f"    {e.key}: {e.device_time_total/10.0:.1f} us ({e.count//10}x)")

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Test direct-permute addressing")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness tests")
    args = parser.parse_args()

    if not args.skip_correctness:
        print("=" * 60)
        print("CORRECTNESS TESTS")
        print("=" * 60)

        # Small case (easy to debug)
        test_correctness(T_per_rank=32, R_per_node=4, E_per_rank=4, TOPK=4,
                         pad_multiple=0, local_rank=0, node_rank=0)
        test_correctness(T_per_rank=32, R_per_node=4, E_per_rank=4, TOPK=4,
                         pad_multiple=0, local_rank=2, node_rank=0)

        # With padding
        test_correctness(T_per_rank=64, R_per_node=8, E_per_rank=8, TOPK=8,
                         pad_multiple=16, local_rank=3, node_rank=0)

        # Latent MoE config (target workload)
        test_correctness(T_per_rank=128, R_per_node=8, E_per_rank=32, TOPK=36,
                         pad_multiple=1, local_rank=0, node_rank=0)
        test_correctness(T_per_rank=128, R_per_node=8, E_per_rank=32, TOPK=36,
                         pad_multiple=1, local_rank=5, node_rank=0)

        # Larger (closer to production but still fast for correctness)
        test_correctness(T_per_rank=512, R_per_node=8, E_per_rank=32, TOPK=36,
                         pad_multiple=1, local_rank=7, node_rank=0)

        # Overflow tests
        # With a very small buffer limit, overflow should trigger
        test_correctness(T_per_rank=128, R_per_node=8, E_per_rank=32, TOPK=36,
                         pad_multiple=1, local_rank=0, node_rank=0,
                         num_permuted_tokens=10)  # way too small → overflow=1
        # With unlimited buffer, no overflow
        test_correctness(T_per_rank=128, R_per_node=8, E_per_rank=32, TOPK=36,
                         pad_multiple=1, local_rank=0, node_rank=0,
                         num_permuted_tokens=-1)  # unlimited → overflow=0

        print("\n" + "=" * 60)
        print("ALL CORRECTNESS TESTS PASSED")
        print("=" * 60)

    if args.benchmark:
        print("\n" + "=" * 60)
        print("BENCHMARKS")
        print("=" * 60)

        # NVL8 latent MoE (target)
        test_benchmark(T_per_rank=8192, R_per_node=8, E_per_rank=32, TOPK=36,
                       pad_multiple=1, local_rank=0, node_rank=0)

        # NVL72 latent MoE (scale target)
        test_benchmark(T_per_rank=8192, R_per_node=72, E_per_rank=32, TOPK=36,
                       pad_multiple=1, local_rank=0, node_rank=0)

        # NVL8 smaller model
        test_benchmark(T_per_rank=4096, R_per_node=8, E_per_rank=8, TOPK=8,
                       pad_multiple=16, local_rank=0, node_rank=0)


if __name__ == "__main__":
    main()
