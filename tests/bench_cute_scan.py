# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Benchmark the CuTe DSL scan kernel with proper timing isolation.

Reports three measurements:
  1. Full call: metadata_preprocess_cute (includes alloc + DLPack + kernel)
  2. Kernel-only: pre-allocated outputs, pre-converted DLPack tensors,
     only the compiled kernel invocation is measured.
  3. Overhead: (1) - (2), showing Python/alloc/DLPack cost per call.

Usage:
    python tests/bench_cute_scan.py
    python tests/bench_cute_scan.py --num-tokens 8192 --num-experts 32 --num-ranks 8 --topk 36
"""

import argparse
import time
import torch

try:
    from deep_ep.cute_kernels.scan import (
        metadata_preprocess_cute,
        scan_kernel_cute,
        ScanKernel,
        _kernel_cache,
        VEC_WIDTH,
    )
    from cutlass.cute.runtime import from_dlpack
    import cutlass.cute as cute
    HAS_CUTE_DSL = True
except ImportError as e:
    HAS_CUTE_DSL = False
    CUTE_IMPORT_ERROR = str(e)


def generate_routing_data(total_tokens, total_experts, topk, device):
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
    warmup=50, iters=200,
):
    if not HAS_CUTE_DSL:
        print(f"SKIP: CuTe DSL not available: {CUTE_IMPORT_ERROR}")
        return

    device = torch.device("cuda:0")
    total_experts = E * R * N
    total_tokens = num_tokens * R * N

    # Clamp blocks to SM count (matching what the kernel does)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if num_blocks > sm_count:
        num_blocks = sm_count

    print(f"\nBenchmark: T={num_tokens}, E={E}, R={R}, K={topk}, N={N}, "
          f"blocks={num_blocks}, threads={num_threads}")
    print(f"  Total tokens: {total_tokens}, total experts: {total_experts}")

    routing_data = generate_routing_data(total_tokens, total_experts, topk, device)

    # ── 1. Full call (includes alloc + DLPack + kernel) ──
    # Warmup
    for _ in range(warmup):
        result = metadata_preprocess_cute(
            routing_data=routing_data,
            num_of_tokens_per_rank=num_tokens,
            num_of_experts_per_rank=E,
            num_of_ranks_per_node=R,
            num_of_nodes=N,
            node_rank=0, local_rank=0,
            topk=topk,
            num_blocks=num_blocks,
            num_threads=num_threads,
        )
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_ev.record()
    for _ in range(iters):
        result = metadata_preprocess_cute(
            routing_data=routing_data,
            num_of_tokens_per_rank=num_tokens,
            num_of_experts_per_rank=E,
            num_of_ranks_per_node=R,
            num_of_nodes=N,
            node_rank=0, local_rank=0,
            topk=topk,
            num_blocks=num_blocks,
            num_threads=num_threads,
        )
    end_ev.record()
    torch.cuda.synchronize()
    full_ms = start_ev.elapsed_time(end_ev) / iters
    print(f"  Full call:   {full_ms:.3f} ms avg")

    # ── 2. Kernel-only (pre-alloc + pre-convert, measure just compiled.__call__) ──
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

    # Pre-convert all tensors via from_dlpack (done once, not per iteration)
    routing_flat = routing_data.reshape(-1)
    tmp_flat = tmp.reshape(-1)
    s2d_flat = sparse_to_dense_map.reshape(-1)
    rdma_flat = rdma_to_attn_map.reshape(-1).view(torch.int8)
    nd_flat = num_dispatched_tokens.reshape(-1)
    le_flat = local_expert_routing_map.reshape(-1).view(torch.int8)

    routing_ct = from_dlpack(routing_flat)
    routing_ct.mark_layout_dynamic()
    tmp_ct = from_dlpack(tmp_flat)
    tmp_ct.mark_layout_dynamic()
    s2d_ct = from_dlpack(s2d_flat)
    s2d_ct.mark_layout_dynamic()
    rdma_ct = from_dlpack(rdma_flat)
    rdma_ct.mark_layout_dynamic()
    nd_ct = from_dlpack(nd_flat)
    nd_ct.mark_layout_dynamic()
    le_ct = from_dlpack(le_flat)
    le_ct.mark_layout_dynamic()

    # Get the compiled kernel from cache
    cache_key = (E, R, N, topk, num_blocks, num_threads)
    compiled = _kernel_cache[cache_key]

    # Warmup kernel-only
    for _ in range(warmup):
        tmp.zero_()
        compiled(
            routing_ct, tmp_ct, s2d_ct, rdma_ct, nd_ct, le_ct,
            0, 0, num_tokens,
        )
    torch.cuda.synchronize()

    # Benchmark kernel-only
    torch.cuda.synchronize()
    start_ev.record()
    for _ in range(iters):
        tmp.zero_()
        compiled(
            routing_ct, tmp_ct, s2d_ct, rdma_ct, nd_ct, le_ct,
            0, 0, num_tokens,
        )
    end_ev.record()
    torch.cuda.synchronize()
    kernel_with_zero_ms = start_ev.elapsed_time(end_ev) / iters

    # ── 3. Measure tmp.zero_() alone to subtract ──
    torch.cuda.synchronize()
    start_ev.record()
    for _ in range(iters):
        tmp.zero_()
    end_ev.record()
    torch.cuda.synchronize()
    zero_ms = start_ev.elapsed_time(end_ev) / iters

    kernel_ms = kernel_with_zero_ms - zero_ms
    overhead_ms = full_ms - kernel_with_zero_ms

    print(f"  Kernel+zero: {kernel_with_zero_ms:.3f} ms avg")
    print(f"  tmp.zero():  {zero_ms:.3f} ms avg")
    print(f"  Kernel-only: {kernel_ms:.3f} ms avg")
    print(f"  Overhead:    {overhead_ms:.3f} ms avg (alloc + DLPack + reshape)")
    print(f"  num_dispatched: {num_dispatched_tokens.item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--topk", type=int, default=36)
    parser.add_argument("--num-blocks", type=int, default=24)
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
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

    # Also bench sparse if topk > 0
    if args.topk > 0:
        bench_cute_scan(
            num_tokens=args.num_tokens,
            E=args.num_experts,
            R=args.num_ranks,
            topk=0,
            num_blocks=args.num_blocks,
            num_threads=args.num_threads,
            warmup=args.warmup,
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
