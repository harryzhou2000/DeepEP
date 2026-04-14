# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Benchmark the CuTe DSL scan kernel.

Measures kernel-only execution time (excluding compilation and allgather).
Also benchmarks the original C++ JIT scan if run via multi-process with DeepEP.

Usage:
    # CuTe DSL kernel only (single GPU, no distributed):
    python tests/bench_cute_scan.py

    # With original C++ JIT for comparison (needs 8 GPUs):
    python tests/bench_cute_scan.py --with-original --num-processes 8

    # Production config:
    python tests/bench_cute_scan.py --num-tokens 8192 --num-experts 32 --num-ranks 8 --topk 36
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


def generate_routing_data(total_tokens, total_experts, topk, device):
    """Generate random routing data."""
    if topk > 0:
        return torch.stack([
            torch.randperm(total_experts, device=device)[:topk]
            for _ in range(total_tokens)
        ]).to(torch.int16)
    else:
        routing_map = torch.zeros(total_tokens, total_experts, dtype=torch.bool, device=device)
        for t in range(total_tokens):
            experts = torch.randperm(total_experts, device=device)[:4]
            routing_map[t, experts] = True
        return routing_map


def bench_cute_scan(
    num_tokens=8192, E=32, R=8, topk=36, N=1,
    num_blocks=24, num_threads=128,
    warmup=20, iters=100,
):
    """Benchmark CuTe DSL scan kernel."""
    if not HAS_CUTE_DSL:
        print(f"SKIP: CuTe DSL not available: {CUTE_IMPORT_ERROR}")
        return

    device = torch.device("cuda:0")
    total_experts = E * R * N
    total_tokens = num_tokens * R * N

    print(f"\nBenchmark: T={num_tokens}, E={E}, R={R}, K={topk}, N={N}, "
          f"blocks={num_blocks}, threads={num_threads}")
    print(f"  Total tokens: {total_tokens}, total experts: {total_experts}")

    routing_data = generate_routing_data(total_tokens, total_experts, topk, device)

    # Warmup (includes first-time compilation)
    print(f"  Warming up ({warmup} iters, first includes JIT compile)...")
    t0 = time.time()
    for i in range(warmup):
        result = metadata_preprocess_cute(
            routing_data=routing_data,
            num_of_tokens_per_rank=num_tokens,
            num_of_experts_per_rank=E,
            num_of_ranks_per_node=R,
            num_of_nodes=N,
            node_rank=0,
            local_rank=0,
            topk=topk,
            num_blocks=num_blocks,
            num_threads=num_threads,
        )
    torch.cuda.synchronize()
    warmup_time = time.time() - t0
    print(f"  Warmup done in {warmup_time:.2f}s (avg {warmup_time/warmup*1000:.1f}ms incl alloc)")

    # Benchmark with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for i in range(iters):
        result = metadata_preprocess_cute(
            routing_data=routing_data,
            num_of_tokens_per_rank=num_tokens,
            num_of_experts_per_rank=E,
            num_of_ranks_per_node=R,
            num_of_nodes=N,
            node_rank=0,
            local_rank=0,
            topk=topk,
            num_blocks=num_blocks,
            num_threads=num_threads,
        )
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    print(f"  CuTe DSL scan: {avg_ms:.3f} ms avg ({iters} iters)")
    print(f"  num_dispatched: {result['num_dispatched_tokens'].item()}")

    # Also measure kernel-only time (pre-allocate outputs to exclude alloc)
    # Pre-allocate all output tensors
    rdma_pad = ((num_tokens - 1) // 16 + 1) * 16
    sparse_to_dense_map = torch.empty(
        (num_tokens * N, R), dtype=torch.int32, device=device,
    )
    rdma_to_attn_map = torch.empty(
        (rdma_pad, N), dtype=torch.bool, device=device,
    )
    num_dispatched_tokens = torch.empty(1, dtype=torch.int32, device=device)
    local_expert_routing_map = torch.empty(
        (total_tokens, E), dtype=torch.bool, device=device,
    )
    tmp = torch.zeros(num_blocks * R, dtype=torch.int64, device=device)

    from deep_ep.cute_kernels.scan import scan_kernel_cute

    # Warmup kernel-only
    for _ in range(warmup):
        tmp.zero_()
        scan_kernel_cute(
            routing_data=routing_data,
            tmp=tmp,
            sparse_to_dense_map=sparse_to_dense_map,
            rdma_to_attn_map=rdma_to_attn_map,
            num_dispatched_tokens=num_dispatched_tokens,
            local_expert_routing_map=local_expert_routing_map,
            node_rank=0,
            local_rank=0,
            num_of_tokens_per_rank=num_tokens,
            num_of_experts_per_rank=E,
            num_of_ranks_per_node=R,
            num_of_nodes=N,
            topk=topk,
            num_blocks=num_blocks,
            num_threads=num_threads,
        )
    torch.cuda.synchronize()

    # Benchmark kernel-only
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        tmp.zero_()
        scan_kernel_cute(
            routing_data=routing_data,
            tmp=tmp,
            sparse_to_dense_map=sparse_to_dense_map,
            rdma_to_attn_map=rdma_to_attn_map,
            num_dispatched_tokens=num_dispatched_tokens,
            local_expert_routing_map=local_expert_routing_map,
            node_rank=0,
            local_rank=0,
            num_of_tokens_per_rank=num_tokens,
            num_of_experts_per_rank=E,
            num_of_ranks_per_node=R,
            num_of_nodes=N,
            topk=topk,
            num_blocks=num_blocks,
            num_threads=num_threads,
        )
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    print(f"  CuTe DSL kernel-only: {avg_ms:.3f} ms avg ({iters} iters)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--topk", type=int, default=36)
    parser.add_argument("--num-blocks", type=int, default=24)
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    bench_cute_scan(
        num_tokens=args.num_tokens,
        E=args.num_experts,
        R=args.num_ranks,
        topk=args.topk,
        num_blocks=args.num_blocks,
        num_threads=args.num_threads,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
