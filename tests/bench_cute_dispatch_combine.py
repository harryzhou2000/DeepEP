# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Benchmark CuTe DSL dispatch and combine kernels.

Usage:
    python tests/bench_cute_dispatch_combine.py
    python tests/bench_cute_dispatch_combine.py --hidden-dim 512 --num-tokens 4096
"""

import argparse
import time
import torch

try:
    from deep_ep.cute_kernels.dispatch import dispatch_cute, _dispatch_cache, _dispatch_launch
    from deep_ep.cute_kernels.combine import combine_cute, _combine_cache, _combine_launch
    from cutlass.cute.runtime import from_dlpack
    HAS_CUTE_DSL = True
except ImportError as e:
    HAS_CUTE_DSL = False
    CUTE_IMPORT_ERROR = str(e)


def generate_test_data(T, H, R, E, topk, max_buf, device, with_probs=False):
    """Generate routing data and buffers."""
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    probs = torch.randn(T, E * R, dtype=torch.float32, device=device) if with_probs else None

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

    dispatch_tokens = [torch.zeros(max_buf, H, dtype=torch.bfloat16, device=device) for _ in range(R)]
    dispatch_probs = [torch.zeros(max_buf, E * R, dtype=torch.float32, device=device) for _ in range(R)] if with_probs else None

    return hidden, probs, s2d_map, rdma_map, dispatch_tokens, dispatch_probs


def bench(T, H, R, E, topk, num_stages, num_blocks, warmup, iters, with_probs=False):
    if not HAS_CUTE_DSL:
        print(f"SKIP: {CUTE_IMPORT_ERROR}")
        return

    device = torch.device("cuda:0")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if num_blocks > sm_count:
        num_blocks = sm_count

    max_buf = T * 2

    prob_str = "w/ probs" if with_probs else "no probs"
    print(f"\nBenchmark ({prob_str}): T={T}, H={H}, R={R}, E={E}, K={topk}, "
          f"stages={num_stages}, blocks={num_blocks}")

    hidden, probs, s2d_map, rdma_map, dispatch_tokens, dispatch_probs = \
        generate_test_data(T, H, R, E, topk, max_buf, device, with_probs)

    # ── Dispatch ──
    print("  Compiling dispatch...", end="", flush=True)
    t0 = time.time()
    dispatch_cute(
        hidden=hidden, probs=probs,
        sparse_to_dense_map=s2d_map, rdma_to_attn_map=rdma_map,
        output_tokens=dispatch_tokens, output_probs=dispatch_probs,
        num_ranks=R, num_experts_per_rank=E,
        num_stages=num_stages, num_blocks=num_blocks,
    )
    torch.cuda.synchronize()
    print(f" {time.time()-t0:.1f}s")

    # Pre-build kernel-only args
    output_token_ptrs = torch.tensor(
        [t.data_ptr() for t in dispatch_tokens], dtype=torch.int64, device=device,
    )
    output_prob_ptrs = torch.tensor(
        [p.data_ptr() for p in dispatch_probs], dtype=torch.int64, device=device,
    ) if with_probs else torch.zeros(R, dtype=torch.int64, device=device)

    hidden_flat = hidden.reshape(-1)
    probs_flat = probs.reshape(-1) if probs is not None else torch.zeros(1, dtype=torch.float32, device=device)
    s2d_flat = s2d_map.reshape(-1)
    rdma_flat = rdma_map.reshape(-1).view(torch.int8)

    h_ct = from_dlpack(hidden_flat); h_ct.mark_layout_dynamic()
    p_ct = from_dlpack(probs_flat); p_ct.mark_layout_dynamic()
    s_ct = from_dlpack(s2d_flat); s_ct.mark_layout_dynamic()
    r_ct = from_dlpack(rdma_flat); r_ct.mark_layout_dynamic()
    tp_ct = from_dlpack(output_token_ptrs); tp_ct.mark_layout_dynamic()
    pp_ct = from_dlpack(output_prob_ptrs); pp_ct.mark_layout_dynamic()

    d_key = (H, R, E, num_stages, num_blocks, with_probs)
    d_compiled = _dispatch_cache[d_key]

    for _ in range(warmup):
        d_compiled(h_ct, p_ct, s_ct, r_ct, tp_ct, pp_ct, T)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        d_compiled(h_ct, p_ct, s_ct, r_ct, tp_ct, pp_ct, T)
    end.record()
    torch.cuda.synchronize()
    dispatch_ms = start.elapsed_time(end) / iters
    print(f"  Dispatch kernel: {dispatch_ms:.3f} ms")

    # ── Combine ──
    print("  Compiling combine...", end="", flush=True)
    t0 = time.time()
    combine_cute(
        input_tokens=dispatch_tokens, input_probs=dispatch_probs,
        sparse_to_dense_map=s2d_map, rdma_to_attn_map=rdma_map,
        num_tokens=T, num_ranks=R, num_experts_per_rank=E,
        hidden_dim=H, num_stages=num_stages, num_blocks=num_blocks,
    )
    torch.cuda.synchronize()
    print(f" {time.time()-t0:.1f}s")

    input_token_ptrs = torch.tensor(
        [t.data_ptr() for t in dispatch_tokens], dtype=torch.int64, device=device,
    )
    input_prob_ptrs = torch.tensor(
        [p.data_ptr() for p in dispatch_probs], dtype=torch.int64, device=device,
    ) if with_probs else torch.zeros(R, dtype=torch.int64, device=device)
    output_tokens = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
    output_probs = torch.zeros(T, E * R, dtype=torch.float32, device=device) if with_probs else torch.zeros(1, dtype=torch.float32, device=device)

    out_tok_flat = output_tokens.reshape(-1)
    out_prob_flat = output_probs.reshape(-1)

    it_ct = from_dlpack(input_token_ptrs); it_ct.mark_layout_dynamic()
    ip_ct = from_dlpack(input_prob_ptrs); ip_ct.mark_layout_dynamic()
    s2_ct = from_dlpack(s2d_flat); s2_ct.mark_layout_dynamic()
    r2_ct = from_dlpack(rdma_flat); r2_ct.mark_layout_dynamic()
    ot_ct = from_dlpack(out_tok_flat); ot_ct.mark_layout_dynamic()
    op_ct = from_dlpack(out_prob_flat); op_ct.mark_layout_dynamic()

    c_key = (H, R, E, num_stages, num_blocks, with_probs)
    c_compiled = _combine_cache[c_key]

    for _ in range(warmup):
        c_compiled(it_ct, ip_ct, s2_ct, r2_ct, ot_ct, op_ct, T)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        c_compiled(it_ct, ip_ct, s2_ct, r2_ct, ot_ct, op_ct, T)
    end.record()
    torch.cuda.synchronize()
    combine_ms = start.elapsed_time(end) / iters
    print(f"  Combine kernel: {combine_ms:.3f} ms")
    print(f"  Total:          {dispatch_ms + combine_ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    # Small config (fast compile)
    bench(T=256, H=128, R=4, E=2, topk=2, num_stages=8, num_blocks=4,
          warmup=args.warmup, iters=args.iters, with_probs=False)

    # User-specified config (no probs for faster compile)
    bench(T=args.num_tokens, H=args.hidden_dim, R=args.num_ranks,
          E=args.num_experts, topk=args.topk,
          num_stages=args.num_stages, num_blocks=args.num_blocks,
          warmup=args.warmup, iters=args.iters, with_probs=False)

    # User-specified config with probs
    bench(T=args.num_tokens, H=args.hidden_dim, R=args.num_ranks,
          E=args.num_experts, topk=args.topk,
          num_stages=args.num_stages, num_blocks=args.num_blocks,
          warmup=args.warmup, iters=args.iters, with_probs=True)


if __name__ == "__main__":
    main()
