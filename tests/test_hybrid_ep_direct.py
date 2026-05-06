# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Lightweight multi-GPU test for direct-permute dispatch.

Compares dispatch_with_permute(direct_permute=True) against the reference
non-direct path. Token data per expert should match (as sets, since ordering
within an expert region may differ).

Run: python tests/test_hybrid_ep_direct.py --num-processes 8
"""

import argparse
import os
import torch
import torch.distributed as dist
import deep_ep

from utils import init_dist

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 512))
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 8192))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", str(NUM_TOKENS_PER_RANK)))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 32))
TOPK = int(os.environ.get("TOPK", 36))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 1))
NUM_SMS_DISPATCH = int(os.environ.get("NUM_SMS_DISPATCH", 24))
NUM_SMS_COMBINE = int(os.environ.get("NUM_SMS_COMBINE", 24))


def init_tensor(hidden_dim, seq_len, topk, num_of_experts):
    """Generate test data: tokens + dense routing."""
    hidden = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.zeros(seq_len, topk, device="cuda", dtype=torch.int64)
    topk_weights = torch.ones(seq_len, topk, device="cuda", dtype=torch.float32)

    for i in range(seq_len):
        selected_experts = torch.randperm(num_of_experts, device="cuda")[:topk]
        topk_idx[i, :] = selected_experts.to(torch.int64)

    # Build probs from topk
    probs = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.float32)
    probs.scatter_(1, topk_idx.long(), topk_weights)

    return hidden, probs, topk_idx, topk_weights


def test_direct_dispatch(buffer, group):
    """Compare direct-permute dispatch against a pure-torch reference.
    
    Reference: allgather hidden states, then locally group tokens by expert
    using the global routing map. This avoids relying on hybrid-ep internals.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = NUM_LOCAL_EXPERTS * world_size

    if rank == 0:
        print(f"\n=== Direct-Permute Dispatch Test ===")
        print(f"  H={HIDDEN_DIM}, T={NUM_TOKENS_PER_RANK}, E_local={NUM_LOCAL_EXPERTS}, "
              f"K={TOPK}, R={world_size}, pad={PAD_MULTIPLE}")

    # Generate test data
    torch.manual_seed(42 + rank)
    hidden, probs, topk_idx, topk_weights = init_tensor(
        HIDDEN_DIM, NUM_TOKENS_PER_RANK, TOPK, num_experts
    )
    dist.barrier()

    # === Pure-torch reference ===
    # Allgather hidden states from all ranks
    all_hidden = torch.empty(
        NUM_TOKENS_PER_RANK * world_size, HIDDEN_DIM,
        dtype=hidden.dtype, device="cuda"
    )
    dist.all_gather_into_tensor(all_hidden, hidden, group=group)

    # Allgather routing maps (as int16)
    local_routing = topk_idx.to(torch.int16).contiguous()
    global_routing = torch.empty(
        NUM_TOKENS_PER_RANK * world_size, TOPK,
        dtype=torch.int16, device="cuda"
    )
    dist.all_gather_into_tensor(
        global_routing.view(torch.int8),
        local_routing.view(torch.int8),
        group=group
    )

    # Allgather probs
    all_probs = torch.empty(
        NUM_TOKENS_PER_RANK * world_size, num_experts,
        dtype=torch.float32, device="cuda"
    )
    dist.all_gather_into_tensor(all_probs, probs, group=group)

    # Build reference expert-grouped output for local experts on this rank
    # Local experts: [rank * NUM_LOCAL_EXPERTS, (rank+1) * NUM_LOCAL_EXPERTS)
    local_expert_start = rank * NUM_LOCAL_EXPERTS
    local_expert_end = local_expert_start + NUM_LOCAL_EXPERTS

    # Collect (token_data, expert_id) for each local expert
    ref_tokens_per_expert = []
    ref_probs_per_expert = []
    for e in range(NUM_LOCAL_EXPERTS):
        global_expert_id = local_expert_start + e
        # Find all (token, topk_slot) pairs routed to this expert
        mask = (global_routing == global_expert_id)  # [T_total, TOPK]
        token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]  # tokens that hit this expert
        ref_tokens_per_expert.append(all_hidden[token_indices])
        # Prob for this expert from each token
        ref_probs_per_expert.append(all_probs[token_indices, global_expert_id])

    # Compute expected tokens_per_expert
    ref_tpe = torch.tensor([t.shape[0] for t in ref_tokens_per_expert], device="cuda", dtype=torch.int64)
    ref_total = ref_tpe.sum().item()

    if rank == 0:
        print(f"  Reference (pure-torch): total tokens={ref_total}, "
              f"tpe sum={ref_tpe.sum().item()}")

    # === Direct path ===
    # Compute num_permuted_tokens from reference (this is what the user would provide)
    num_permuted_tokens = int(ref_total)

    (
        direct_tokens, direct_probs, direct_scaling, direct_tpe, direct_handle
    ) = buffer.dispatch_with_permute(
        hidden=hidden,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_of_experts=num_experts,
        num_permuted_tokens=num_permuted_tokens,
        pad_multiple=PAD_MULTIPLE,
        dense_routing=True,
        direct_permute=True,
        global_routing_map=global_routing,
    )
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"  Direct dispatch done: tokens shape={direct_tokens.shape}, "
              f"tpe sum={direct_tpe.sum().item()}")

    # === Compare ===
    # 1. tokens_per_expert should match
    direct_tpe_cpu = direct_tpe.cpu()
    ref_tpe_cpu = ref_tpe.cpu()
    tpe_match = torch.equal(ref_tpe_cpu, direct_tpe_cpu)
    if rank == 0:
        print(f"  tokens_per_expert: {'PASS' if tpe_match else 'FAIL'}")
    assert tpe_match, f"Rank {rank}: tokens_per_expert mismatch\n  ref={ref_tpe_cpu}\n  direct={direct_tpe_cpu}"

    # 2. Token content per expert region should match (as multisets — order is non-deterministic)
    expert_starts = [0]
    for e in range(NUM_LOCAL_EXPERTS):
        expert_starts.append(expert_starts[-1] + direct_tpe_cpu[e].item())

    num_expert_match = 0
    for e in range(NUM_LOCAL_EXPERTS):
        start = expert_starts[e]
        end = expert_starts[e + 1]
        count = end - start
        if count == 0:
            num_expert_match += 1
            continue

        # Reference tokens for this expert (from pure-torch)
        ref_expert = ref_tokens_per_expert[e]
        # Direct output tokens for this expert
        direct_expert = direct_tokens[start:end]

        assert ref_expert.shape[0] == count, \
            f"Expert {e}: ref count {ref_expert.shape[0]} != direct count {count}"

        # Compare as multisets (order within expert is non-deterministic)
        ref_set = set(tuple(r.tolist()) for r in ref_expert.float())
        direct_set = set(tuple(r.tolist()) for r in direct_expert.float())
        if ref_set == direct_set:
            num_expert_match += 1
        else:
            missing = len(ref_set - direct_set)
            extra = len(direct_set - ref_set)
            if rank == 0:
                print(f"    Expert {e} MISMATCH (count={count}): "
                      f"missing={missing}, extra={extra}")

    if rank == 0:
        print(f"  Token content: {num_expert_match}/{NUM_LOCAL_EXPERTS} experts match")
        if num_expert_match == NUM_LOCAL_EXPERTS:
            print(f"\n  === ALL TESTS PASSED ===")
        else:
            print(f"\n  === {NUM_LOCAL_EXPERTS - num_expert_match} EXPERTS DIFFER ===")

    dist.barrier()
    return num_expert_match == NUM_LOCAL_EXPERTS


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)

    # Static budget for the direct-output buffer.
    # With random routing, actual token counts can slightly exceed T*TOPK due to non-uniform
    # distribution. Add 5% margin to account for statistical variation.
    num_permuted_tokens_direct = int(NUM_TOKENS_PER_RANK * TOPK * 1.05)

    buffer = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
        num_local_experts=NUM_LOCAL_EXPERTS,
        use_fp8=False,
        num_sms_dispatch_api=NUM_SMS_DISPATCH,
        num_sms_combine_api=NUM_SMS_COMBINE,
        num_permuted_tokens_direct=num_permuted_tokens_direct,
    )

    success = test_direct_dispatch(buffer, group)

    dist.barrier()
    dist.destroy_process_group()

    if not success and local_rank == 0:
        raise RuntimeError("Direct-permute dispatch test FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test direct-permute dispatch')
    parser.add_argument('--num-processes', type=int, default=8)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)
