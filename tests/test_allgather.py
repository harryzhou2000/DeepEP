# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Dedicated test for the custom NVLink allgather kernel (ag_nvl_tma_kernel).

Tests correctness by comparing custom allgather output against torch.distributed.all_gather_into_tensor,
then benchmarks TMA vs legacy vs NCCL.

Usage:
    # 8-GPU test with default params:
    NUM_TOKENS_PER_RANK=8192 NUM_LOCAL_EXPERTS=32 TOPK=36 \
        python tests/test_allgather.py --num-processes 8

    # Force legacy kernel for comparison:
    HYBRID_EP_USE_AG_NVL_LEGACY=1 NUM_TOKENS_PER_RANK=8192 NUM_LOCAL_EXPERTS=32 TOPK=36 \
        python tests/test_allgather.py --num-processes 8
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from utils import bench, init_dist


# ---- Config from env ----
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
TOPK = int(os.environ.get("TOPK", 8))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
SEED = int(os.environ.get("SEED", 1025))
LOG_LABEL_WIDTH = 48

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def make_routing_data(num_tokens, topk, num_experts, dense):
    """Create routing data: dense int16 topk_idx or sparse bool routing_map."""
    topk_idx = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:topk] for _ in range(num_tokens)]
    ).to(torch.int64)

    if dense:
        routing_data = topk_idx.to(torch.int16).contiguous()
    else:
        routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool)
        for i in range(num_tokens):
            routing_map[i, topk_idx[i]] = True
        routing_data = routing_map

    return routing_data


def test_correctness(group, rank, world_size, num_tokens, topk, num_experts):
    """Test custom TMA allgather against NCCL by comparing full dispatch outputs."""
    import deep_ep

    if rank == 0:
        print(f"\n=== Allgather Correctness ({world_size} ranks) ===", flush=True)
        print(
            f"  Comparing dispatch(custom_allgather=True) vs dispatch(custom_allgather=False)",
            flush=True,
        )

    # Shared token data + routing for all sub-tests
    hidden = torch.randn(num_tokens, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    topk_idx_i64 = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:topk] for _ in range(num_tokens)]
    ).to(torch.int64)
    topk_weights = torch.ones(num_tokens, topk, device="cuda", dtype=torch.float32)

    # Build routing_map for sparse path
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool)
    for i in range(num_tokens):
        routing_map[i, topk_idx_i64[i]] = True

    buffer_custom = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=num_tokens,
        num_local_experts=NUM_LOCAL_EXPERTS,
        enable_custom_allgather=True,
    )
    buffer_nccl = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=num_tokens,
        num_local_experts=NUM_LOCAL_EXPERTS,
        enable_custom_allgather=False,
    )

    for mode, label in [("dense", "dense int16"), ("sparse", "sparse bool")]:
        if mode == "dense":
            dispatch_kwargs = dict(
                hidden=hidden,
                topk_idx=topk_idx_i64,
                topk_weights=topk_weights,
                num_of_experts=num_experts,
            )
            data_bytes = num_tokens * topk * 2
        else:
            dispatch_kwargs = dict(
                hidden=hidden,
                routing_map=routing_map,
                probs=torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.float32),
            )
            data_bytes = num_tokens * num_experts

        out_custom = buffer_custom.dispatch(**dispatch_kwargs)
        out_nccl = buffer_nccl.dispatch(**dispatch_kwargs)

        # Compare dispatched hidden tokens (bitwise)
        tokens_match = torch.equal(
            out_custom[0].contiguous().view(torch.uint8),
            out_nccl[0].contiguous().view(torch.uint8),
        )

        # Compare dispatched probs
        probs_match = True
        if out_custom[1] is not None and out_nccl[1] is not None:
            probs_match = torch.equal(
                out_custom[1].contiguous().view(torch.uint8),
                out_nccl[1].contiguous().view(torch.uint8),
            )

        status = "PASS" if (tokens_match and probs_match) else "FAIL"
        if rank == 0:
            print(
                f"  {label:<20s}: tokens={tokens_match}, probs={probs_match} "
                f"  [{status}]  (routing_data: {data_bytes / 1024:.1f} KB/rank)",
                flush=True,
            )
        if not (tokens_match and probs_match):
            if not tokens_match:
                diff_count = (out_custom[0] != out_nccl[0]).sum().item()
                print(
                    f"    [rank {rank}] token mismatch: {diff_count} elements differ",
                    flush=True,
                )
            if not probs_match:
                diff_count = (out_custom[1] != out_nccl[1]).sum().item()
                print(
                    f"    [rank {rank}] probs mismatch: {diff_count} elements differ",
                    flush=True,
                )

        dist.barrier()

    if rank == 0:
        print("", flush=True)


def test_benchmark(group, rank, world_size, num_tokens, topk, num_experts):
    """Benchmark custom TMA allgather vs legacy vs NCCL."""
    import deep_ep

    if rank == 0:
        print(f"=== Allgather Benchmark ({world_size} ranks) ===", flush=True)
        print(
            f"  T={num_tokens}, K={topk}, E={num_experts}, "
            f"E_per_rank={NUM_LOCAL_EXPERTS}, H={HIDDEN_DIM}",
            flush=True,
        )
        print("", flush=True)

    # Shared data across all sub-tests
    topk_idx_i64 = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:topk] for _ in range(num_tokens)]
    ).to(torch.int64)
    topk_weights = torch.ones(num_tokens, topk, device="cuda", dtype=torch.float32)
    hidden = torch.randn(num_tokens, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool)
    probs = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    for i in range(num_tokens):
        routing_map[i, topk_idx_i64[i]] = True
        probs[i, topk_idx_i64[i]] = topk_weights[i]

    for dense, label in [(True, "dense int16"), (False, "sparse bool")]:
        if dense:
            routing_data_bytes = num_tokens * topk * 2  # int16
            dispatch_kwargs = dict(
                hidden=hidden,
                topk_idx=topk_idx_i64,
                topk_weights=topk_weights,
                num_of_experts=num_experts,
            )
        else:
            routing_data_bytes = num_tokens * num_experts  # bool
            dispatch_kwargs = dict(
                hidden=hidden,
                routing_map=routing_map,
                probs=probs,
            )

        if rank == 0:
            print(
                f"  --- {label} (routing_data: {routing_data_bytes / 1024:.1f} KB/rank, "
                f"total AG: {routing_data_bytes * world_size / 1024:.1f} KB) ---",
                flush=True,
            )

        # ---- Benchmark: Full dispatch (includes allgather + scan + dispatch kernel) ----
        # This gives us end-to-end timing that includes allgather.

        # 1. Custom allgather (TMA)
        buffer_tma = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=HIDDEN_DIM,
            max_num_of_tokens_per_rank=num_tokens,
            num_local_experts=NUM_LOCAL_EXPERTS,
            enable_custom_allgather=True,
        )
        dist.barrier()
        avg_tma, min_tma, max_tma = bench(
            lambda: buffer_tma.dispatch(**dispatch_kwargs),
            num_warmups=20,
            num_tests=50,
        )

        # 2. NCCL allgather
        buffer_nccl = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=HIDDEN_DIM,
            max_num_of_tokens_per_rank=num_tokens,
            num_local_experts=NUM_LOCAL_EXPERTS,
            enable_custom_allgather=False,
        )
        dist.barrier()
        avg_nccl, min_nccl, max_nccl = bench(
            lambda: buffer_nccl.dispatch(**dispatch_kwargs),
            num_warmups=20,
            num_tests=50,
        )

        # 3. Legacy custom allgather
        os.environ["HYBRID_EP_USE_AG_NVL_LEGACY"] = "1"
        buffer_legacy = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=HIDDEN_DIM,
            max_num_of_tokens_per_rank=num_tokens,
            num_local_experts=NUM_LOCAL_EXPERTS,
            enable_custom_allgather=True,
        )
        dist.barrier()
        avg_legacy, min_legacy, max_legacy = bench(
            lambda: buffer_legacy.dispatch(**dispatch_kwargs),
            num_warmups=20,
            num_tests=50,
        )
        os.environ.pop("HYBRID_EP_USE_AG_NVL_LEGACY", None)

        if rank == 0:
            print(
                f"    {'dispatch (TMA AG)':<{LOG_LABEL_WIDTH}s} "
                f"avg={avg_tma * 1e6:8.1f} us  min={min_tma * 1e6:8.1f} us  max={max_tma * 1e6:8.1f} us",
                flush=True,
            )
            print(
                f"    {'dispatch (NCCL AG)':<{LOG_LABEL_WIDTH}s} "
                f"avg={avg_nccl * 1e6:8.1f} us  min={min_nccl * 1e6:8.1f} us  max={max_nccl * 1e6:8.1f} us",
                flush=True,
            )
            print(
                f"    {'dispatch (legacy AG)':<{LOG_LABEL_WIDTH}s} "
                f"avg={avg_legacy * 1e6:8.1f} us  min={min_legacy * 1e6:8.1f} us  max={max_legacy * 1e6:8.1f} us",
                flush=True,
            )
            print("", flush=True)

        dist.barrier()


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = NUM_LOCAL_EXPERTS * world_size

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        test_correctness(group, rank, world_size, NUM_TOKENS_PER_RANK, TOPK, num_experts)
        test_benchmark(group, rank, world_size, NUM_TOKENS_PER_RANK, TOPK, num_experts)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom allgather kernel")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="Run as a single process with the given local rank",
    )
    args = parser.parse_args()

    if args.local_rank is not None:
        test_main(args.local_rank, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(
            test_main, args=(args.num_processes, args), nprocs=args.num_processes
        )
