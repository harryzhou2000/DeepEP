# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
CuTe DSL scan kernel for hybrid-ep metadata preprocessing.

Key optimizations:
  - Vectorized GMEM loads: 4 int16s per Int64 load (matching C++ uint2).
  - SMEM staging for TOPK indices (supports dynamic indexing).
  - SMEM per-thread per-rank counters (avoids register-spill from constexpr unrolling).
  - Dynamic range() for R and TOPK loops to keep predicate count low.
    Only the vectorized unpack (4 iterations) uses range_constexpr.
  - Cross-block polling via atomic_exch/atomic_add (relaxed GPU scope).
  - Uint32 lane_mask for UB-free warp exclusive scan.
  - SM count clamping to prevent grid deadlock.

Architecture: R-pass over tokens (one pass per rank). Each pass reads TOPK
from SMEM (loaded once), computes needed flag, does ballot+popc scan.
The R-pass approach avoids the 18K predicate register explosion from
constexpr(R) × TOPK unrolling, and each pass is identical code (no unroll).
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

WARP_SIZE = 32
SCAN_STATE_PRIV_SUM = 1
VEC_WIDTH = 4


def metadata_preprocess_cute(
    routing_data: torch.Tensor,
    num_of_tokens_per_rank: int,
    num_of_experts_per_rank: int,
    num_of_ranks_per_node: int,
    num_of_nodes: int,
    node_rank: int,
    local_rank: int,
    topk: int,
    num_blocks: int = 24,
    num_threads: int = 128,
) -> dict:
    """Python entry point — allocates outputs and launches the CuTe DSL scan."""
    assert num_of_nodes == 1, "CuTe DSL scan currently only supports single-node"
    assert num_threads % WARP_SIZE == 0

    device = routing_data.device
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device_idx).multi_processor_count
    if num_blocks > sm_count:
        num_blocks = sm_count

    total_tokens = num_of_tokens_per_rank * num_of_ranks_per_node * num_of_nodes
    rdma_pad = ((num_of_tokens_per_rank - 1) // 16 + 1) * 16

    routing_data = routing_data.contiguous()

    sparse_to_dense_map = torch.empty(
        (num_of_tokens_per_rank * num_of_nodes, num_of_ranks_per_node),
        dtype=torch.int32, device=device,
    )
    rdma_to_attn_map = torch.empty(
        (rdma_pad, num_of_nodes), dtype=torch.bool, device=device,
    )
    attn_to_rdma_map = torch.zeros(
        (num_of_tokens_per_rank, max(num_of_nodes - 1, 1)),
        dtype=torch.bool, device=device,
    )
    num_dispatched_tokens = torch.empty(1, dtype=torch.int32, device=device)
    local_expert_routing_map = torch.empty(
        (total_tokens, num_of_experts_per_rank), dtype=torch.bool, device=device,
    )
    tmp = torch.zeros(
        num_blocks * num_of_ranks_per_node, dtype=torch.int64, device=device,
    )

    scan_kernel_cute(
        routing_data=routing_data,
        tmp=tmp,
        sparse_to_dense_map=sparse_to_dense_map,
        rdma_to_attn_map=rdma_to_attn_map,
        num_dispatched_tokens=num_dispatched_tokens,
        local_expert_routing_map=local_expert_routing_map,
        node_rank=node_rank,
        local_rank=local_rank,
        num_of_tokens_per_rank=num_of_tokens_per_rank,
        num_of_experts_per_rank=num_of_experts_per_rank,
        num_of_ranks_per_node=num_of_ranks_per_node,
        num_of_nodes=num_of_nodes,
        topk=topk,
        num_blocks=num_blocks,
        num_threads=num_threads,
    )

    return {
        "sparse_to_dense_map": sparse_to_dense_map,
        "rdma_to_attn_map": rdma_to_attn_map,
        "attn_to_rdma_map": attn_to_rdma_map,
        "num_dispatched_tokens": num_dispatched_tokens,
        "local_expert_routing_map": local_expert_routing_map,
    }


class ScanKernel:
    """CuTe DSL scan — dynamic loops + SMEM staging to minimize register pressure."""

    def __init__(self, E, R, N, TOPK, NUM_BLOCKS, NUM_THREADS):
        self.E = E
        self.R = R
        self.N = N
        self.TOPK = TOPK
        self.NUM_BLOCKS = NUM_BLOCKS
        self.NUM_THREADS = NUM_THREADS

    @cute.jit
    def __call__(
        self,
        routing_data: cute.Tensor,
        tmp: cute.Tensor,
        s2d_map: cute.Tensor,
        rdma_map: cute.Tensor,
        num_dispatched: cute.Tensor,
        local_expert_map: cute.Tensor,
        node_rank: cutlass.Int32,
        local_rank: cutlass.Int32,
        num_of_tokens_per_rank: cutlass.Int32,
        E: cutlass.Constexpr,
        R: cutlass.Constexpr,
        N: cutlass.Constexpr,
        TOPK: cutlass.Constexpr,
        NUM_BLOCKS: cutlass.Constexpr,
        NUM_THREADS: cutlass.Constexpr,
    ):
        NUM_WARPS = NUM_THREADS // WARP_SIZE
        smem_warp = (NUM_WARPS * R + R) * 4  # warp_sums + prev_block_sum
        smem_topk = NUM_THREADS * TOPK * 2 if TOPK > 0 else 0  # int16 staging
        smem_size = smem_warp + smem_topk

        self.kernel(
            routing_data, tmp, s2d_map, rdma_map, num_dispatched, local_expert_map,
            node_rank, local_rank, num_of_tokens_per_rank,
            E, R, N, TOPK, NUM_BLOCKS, NUM_THREADS,
        ).launch(
            grid=[NUM_BLOCKS, 1, 1],
            block=[NUM_THREADS, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        routing_data: cute.Tensor,
        tmp: cute.Tensor,
        s2d_map: cute.Tensor,
        rdma_map: cute.Tensor,
        num_dispatched: cute.Tensor,
        local_expert_map: cute.Tensor,
        node_rank: cutlass.Int32,
        local_rank: cutlass.Int32,
        num_of_tokens_per_rank: cutlass.Int32,
        E: cutlass.Constexpr,
        R: cutlass.Constexpr,
        N: cutlass.Constexpr,
        TOPK: cutlass.Constexpr,
        NUM_BLOCKS: cutlass.Constexpr,
        NUM_THREADS: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        warp_id = tidx // WARP_SIZE
        lane_id = tidx % WARP_SIZE

        NUM_WARPS = NUM_THREADS // WARP_SIZE
        EXPERTS_PER_NODE = E * R

        num_total_tokens = num_of_tokens_per_rank * R * N
        total_threads = NUM_THREADS * NUM_BLOCKS
        tokens_per_thread = (num_total_tokens + total_threads - 1) // total_threads
        tokens_per_warp = tokens_per_thread * WARP_SIZE
        tokens_per_block = tokens_per_warp * NUM_WARPS

        block_start = bidx * tokens_per_block
        warp_start = block_start + warp_id * tokens_per_warp
        thread_start = warp_start + lane_id

        tokens_per_node = num_of_tokens_per_rank * R
        rdma_map_size_per_node = ((num_of_tokens_per_rank + 15) // 16) * 16

        # SMEM
        smem = utils.SmemAllocator()
        warp_sums = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((NUM_WARPS, R), stride=(R, 1)),
        )
        prev_block_sum = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((R,), stride=(1,)),
        )
        if cutlass.const_expr(TOPK > 0):
            topk_stage = smem.allocate_tensor(
                cutlass.Int16, cute.make_layout((NUM_THREADS, TOPK), stride=(TOPK, 1)),
            )
            VEC_LOADS = (TOPK + VEC_WIDTH - 1) // VEC_WIDTH

        # =================================================================
        # Step 0: Intra-block partial sums
        # =================================================================
        # R-pass: one pass per rank. Each pass:
        #   - Loads TOPK indices into SMEM staging (vectorized, once per token)
        #   - Checks if token goes to rank r
        #   - Warp-reduces and stores to warp_sums
        # On the first pass (r==0), also writes rdma_to_attn_map and loads staging.
        for r in range(R):
            thread_count = cutlass.Int32(0)

            for ti in range(tokens_per_thread):
                cur_token = thread_start + ti * WARP_SIZE
                if cur_token < num_total_tokens:
                    node_start = node_rank * EXPERTS_PER_NODE
                    needed = cutlass.Int32(0)

                    if cutlass.const_expr(TOPK > 0):
                        # Vectorized load into SMEM staging (every pass reloads
                        # because each pass iterates over the same tokens, but
                        # SMEM holds only the last-loaded token).
                        routing_base = routing_data.iterator + cur_token * TOPK
                        for v in cutlass.range_constexpr(VEC_LOADS):
                            eo = v * VEC_WIDTH
                            if cutlass.const_expr(eo + VEC_WIDTH <= TOPK):
                                pk = cute.arch.load(routing_base + eo, cutlass.Int64)
                                for u in cutlass.range_constexpr(VEC_WIDTH):
                                    sh = cutlass.Int64(u * 16)
                                    elem = ((pk >> sh) & cutlass.Int64(0xFFFF)).to(cutlass.Int16)
                                    topk_stage[tidx, eo + u] = elem
                            else:
                                for u in cutlass.range_constexpr(TOPK - eo):
                                    topk_stage[tidx, eo + u] = routing_data[
                                        cur_token * TOPK + eo + u
                                    ]

                        # Check if any expert maps to rank r
                        for k in range(TOPK):
                            eid = topk_stage[tidx, k].to(cutlass.Int32)
                            if eid >= node_start:
                                local_eid = eid - node_start
                                if local_eid < EXPERTS_PER_NODE:
                                    if local_eid // E == r:
                                        needed = cutlass.Int32(1)
                    else:
                        # Sparse bool mode
                        row_off = cur_token * (E * R * N) + node_rank * (E * R) + r * E
                        for e in range(E):
                            bv = routing_data[row_off + e]
                            if bv != cutlass.Int8(0):
                                needed = cutlass.Int32(1)

                    thread_count = thread_count + needed

                    # Write rdma_to_attn_map on first rank pass only
                    if r == 0:
                        # Compute node-needed: any rank on local node wants this token
                        node_needed = cutlass.Int32(0)
                        if cutlass.const_expr(TOPK > 0):
                            for kk in range(TOPK):
                                eid2 = topk_stage[tidx, kk].to(cutlass.Int32)
                                if eid2 >= node_start:
                                    if eid2 - node_start < EXPERTS_PER_NODE:
                                        node_needed = cutlass.Int32(1)
                        else:
                            row_off2 = cur_token * (E * R * N) + node_rank * (E * R)
                            for j2 in range(R):
                                for e2 in range(E):
                                    bv2 = routing_data[row_off2 + j2 * E + e2]
                                    if bv2 != cutlass.Int8(0):
                                        node_needed = cutlass.Int32(1)
                        cur_node_rank = cur_token // tokens_per_node
                        cur_rem0 = cur_token % tokens_per_node
                        cur_local_rank = cur_rem0 // num_of_tokens_per_rank
                        cur_local_id = cur_rem0 % num_of_tokens_per_rank
                        if cur_local_rank == local_rank:
                            rdma_off = cur_node_rank * rdma_map_size_per_node + cur_local_id
                            if rdma_off < cute.size(rdma_map):
                                rdma_map[rdma_off] = node_needed.to(cutlass.Int8)

            warp_total_r = cute.arch.warp_reduction_sum(thread_count)
            if lane_id == 0:
                warp_sums[warp_id, r] = warp_total_r

        cute.arch.sync_threads()

        # =================================================================
        # Step 1: Cross-block prefix sum (parallel: tidx < R)
        # =================================================================
        r_idx = tidx
        if r_idx < R:
            block_sum = cutlass.Int32(0)
            for w in range(NUM_WARPS):
                block_sum = block_sum + warp_sums[w, r_idx]

            packed = (block_sum.to(cutlass.Int64) << cutlass.Int64(32)) | cutlass.Int64(SCAN_STATE_PRIV_SUM)
            tmp_base = tmp.iterator
            _ = cute.arch.atomic_exch(
                tmp_base + (bidx * R + r_idx), val=packed,
                sem="relaxed", scope="gpu",
            )

            prev_sum = cutlass.Int32(0)
            for prev_blk in range(bidx):
                read_ptr = tmp_base + (prev_blk * R + r_idx)
                data = cute.arch.atomic_add(read_ptr, cutlass.Int64(0), sem="relaxed", scope="gpu")
                state = data & cutlass.Int64(0xFFFFFFFF)
                while state != cutlass.Int64(SCAN_STATE_PRIV_SUM):
                    data = cute.arch.atomic_add(read_ptr, cutlass.Int64(0), sem="relaxed", scope="gpu")
                    state = data & cutlass.Int64(0xFFFFFFFF)
                value = (data >> cutlass.Int64(32)).to(cutlass.Int32)
                prev_sum = prev_sum + value

            prev_block_sum[r_idx] = prev_sum

        cute.arch.sync_threads()

        # =================================================================
        # Step 2: Final scan (R-pass, reloads TOPK from GMEM each pass)
        # =================================================================
        lane_mask = (cutlass.Uint32(1) << lane_id.to(cutlass.Uint32)) - cutlass.Uint32(1)

        for r in range(R):
            # Prefix for this warp
            acc = prev_block_sum[r]
            for w in range(warp_id):
                acc = acc + warp_sums[w, r]
            prev_sum_r = acc

            for ti in range(tokens_per_thread):
                cur_token = thread_start + ti * WARP_SIZE
                token_oob = cutlass.Int32(1)
                if cur_token < num_total_tokens:
                    token_oob = cutlass.Int32(0)

                all_oob = cute.arch.vote_ballot_sync(token_oob != cutlass.Int32(0))
                if all_oob != 0xFFFFFFFF:
                    node_start = node_rank * EXPERTS_PER_NODE
                    needed = cutlass.Int32(0)

                    if token_oob == cutlass.Int32(0):
                        if cutlass.const_expr(TOPK > 0):
                            # Reload into SMEM staging (vectorized)
                            routing_base2 = routing_data.iterator + cur_token * TOPK
                            for v2 in cutlass.range_constexpr(VEC_LOADS):
                                eo2 = v2 * VEC_WIDTH
                                if cutlass.const_expr(eo2 + VEC_WIDTH <= TOPK):
                                    pk2 = cute.arch.load(routing_base2 + eo2, cutlass.Int64)
                                    for u2 in cutlass.range_constexpr(VEC_WIDTH):
                                        sh2 = cutlass.Int64(u2 * 16)
                                        elem2 = ((pk2 >> sh2) & cutlass.Int64(0xFFFF)).to(cutlass.Int16)
                                        topk_stage[tidx, eo2 + u2] = elem2
                                else:
                                    for u2 in cutlass.range_constexpr(TOPK - eo2):
                                        topk_stage[tidx, eo2 + u2] = routing_data[
                                            cur_token * TOPK + eo2 + u2
                                        ]

                            for k in range(TOPK):
                                eid = topk_stage[tidx, k].to(cutlass.Int32)
                                if eid >= node_start:
                                    local_eid = eid - node_start
                                    if local_eid < EXPERTS_PER_NODE:
                                        if local_eid // E == r:
                                            needed = cutlass.Int32(1)
                        else:
                            row_off = cur_token * (E * R * N) + node_rank * (E * R) + r * E
                            for e in range(E):
                                bv = routing_data[row_off + e]
                                if bv != cutlass.Int8(0):
                                    needed = cutlass.Int32(1)

                    vote = cute.arch.vote_ballot_sync(needed != cutlass.Int32(0))
                    tile_sum = cute.arch.popc(vote)
                    ex_scan = cute.arch.popc(vote.to(cutlass.Uint32) & lane_mask).to(cutlass.Int32)

                    final_pos = prev_sum_r + ex_scan
                    if needed == cutlass.Int32(0):
                        final_pos = cutlass.Int32(-1)

                    if token_oob == cutlass.Int32(0):
                        cur_node_rank = cur_token // tokens_per_node
                        cur_rem = cur_token % tokens_per_node
                        cur_lr = cur_rem // num_of_tokens_per_rank
                        cur_lid = cur_rem % num_of_tokens_per_rank

                        if cur_lr == local_rank:
                            s2d_off = (cur_node_rank * num_of_tokens_per_rank + cur_lid) * R + r
                            s2d_map[s2d_off] = final_pos

                        # Write local_expert_routing_map
                        if r == local_rank:
                            if needed != cutlass.Int32(0):
                                if cutlass.const_expr(TOPK > 0):
                                    ns = node_rank * EXPERTS_PER_NODE + local_rank * E
                                    for e in range(E):
                                        expert_found = cutlass.Int8(0)
                                        for kk in range(TOPK):
                                            eid3 = topk_stage[tidx, kk].to(cutlass.Int32)
                                            if eid3 == ns + e:
                                                expert_found = cutlass.Int8(1)
                                        local_expert_map[final_pos * E + e] = expert_found
                                else:
                                    row_off3 = cur_token * (E * R * N) + node_rank * (E * R) + local_rank * E
                                    for e in range(E):
                                        local_expert_map[final_pos * E + e] = routing_data[row_off3 + e]

                        if cur_token == num_total_tokens - 1:
                            if r == local_rank:
                                num_dispatched[0] = prev_sum_r + tile_sum

                    prev_sum_r = prev_sum_r + tile_sum


# ---------------------------------------------------------------------------
_kernel_cache = {}


def scan_kernel_cute(
    routing_data: torch.Tensor,
    tmp: torch.Tensor,
    sparse_to_dense_map: torch.Tensor,
    rdma_to_attn_map: torch.Tensor,
    num_dispatched_tokens: torch.Tensor,
    local_expert_routing_map: torch.Tensor,
    node_rank: int,
    local_rank: int,
    num_of_tokens_per_rank: int,
    num_of_experts_per_rank: int,
    num_of_ranks_per_node: int,
    num_of_nodes: int,
    topk: int,
    num_blocks: int,
    num_threads: int,
):
    """Launch the CuTe DSL scan kernel."""
    cache_key = (
        num_of_experts_per_rank, num_of_ranks_per_node, num_of_nodes,
        topk, num_blocks, num_threads,
    )

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

    kernel = ScanKernel(
        E=num_of_experts_per_rank, R=num_of_ranks_per_node,
        N=num_of_nodes, TOPK=topk,
        NUM_BLOCKS=num_blocks, NUM_THREADS=num_threads,
    )

    if cache_key not in _kernel_cache:
        compiled = cute.compile(
            kernel,
            routing_ct, tmp_ct, s2d_ct, rdma_ct, nd_ct, le_ct,
            node_rank, local_rank, num_of_tokens_per_rank,
            num_of_experts_per_rank, num_of_ranks_per_node, num_of_nodes,
            topk, num_blocks, num_threads,
        )
        _kernel_cache[cache_key] = compiled

    compiled = _kernel_cache[cache_key]
    compiled(
        routing_ct, tmp_ct, s2d_ct, rdma_ct, nd_ct, le_ct,
        node_rank, local_rank, num_of_tokens_per_rank,
    )
