# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
CuTe DSL dispatch kernel for hybrid-ep with TMA pipeline.

Architecture (single-node, BF16, 4 warps = 128 threads):
  Warp 0:   G2S producer — cp.async.bulk loads source tokens GMEM→SMEM
  Warps 1-3: S2G consumer — reads routing map, cp.async.bulk stores SMEM→dest
  Pipeline:  NUM_STAGES stages, mbarrier producer/consumer sync

Single-GPU proof-of-concept. Source and destination are local GPU memory.
For multi-GPU, destination pointers would be NVLink peer addresses.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

WARP_SIZE = 32


def dispatch_cute(
    hidden: torch.Tensor,               # [T, H] bf16
    probs: torch.Tensor,                # [T, E*R] f32 or None
    sparse_to_dense_map: torch.Tensor,  # [T, R] int32
    rdma_to_attn_map: torch.Tensor,     # [T] bool
    output_tokens: list,                # R tensors [max_out, H] bf16
    output_probs: list,                 # R tensors [max_out, E*R] f32 or None
    num_ranks: int,
    num_experts_per_rank: int,
    num_stages: int = 8,
    num_blocks: int = 24,
) -> None:
    """CuTe DSL dispatch with TMA pipeline."""
    T, H = hidden.shape
    R = num_ranks
    E = num_experts_per_rank
    with_probs = probs is not None

    device = hidden.device
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device_idx).multi_processor_count
    if num_blocks > sm_count:
        num_blocks = sm_count

    output_token_ptrs = torch.tensor(
        [t.data_ptr() for t in output_tokens], dtype=torch.int64, device=device,
    )
    if with_probs:
        output_prob_ptrs = torch.tensor(
            [p.data_ptr() for p in output_probs], dtype=torch.int64, device=device,
        )
    else:
        output_prob_ptrs = torch.zeros(R, dtype=torch.int64, device=device)

    _dispatch_launch(
        hidden, probs, sparse_to_dense_map, rdma_to_attn_map,
        output_token_ptrs, output_prob_ptrs,
        T, H, R, E, with_probs, num_stages, num_blocks,
    )


class DispatchTMAKernel:
    """Dispatch kernel with TMA G2S/S2G pipeline and warp specialization."""

    def __init__(self, H, R, E, NUM_STAGES, NUM_BLOCKS, WITH_PROBS):
        self.H = H
        self.R = R
        self.E = E
        self.NUM_STAGES = NUM_STAGES
        self.NUM_BLOCKS = NUM_BLOCKS
        self.WITH_PROBS = WITH_PROBS

    @cute.jit
    def __call__(
        self,
        hidden: cute.Tensor,
        probs: cute.Tensor,
        s2d_map: cute.Tensor,
        rdma_map: cute.Tensor,
        out_token_ptrs: cute.Tensor,
        out_prob_ptrs: cute.Tensor,
        num_tokens: cutlass.Int32,
        H: cutlass.Constexpr,
        R: cutlass.Constexpr,
        E: cutlass.Constexpr,
        NUM_STAGES: cutlass.Constexpr,
        NUM_BLOCKS: cutlass.Constexpr,
        WITH_PROBS: cutlass.Constexpr,
    ):
        NUM_THREADS = 128
        # SMEM: token staging [STAGES][H] bf16 + mbarrier [STAGES][2] u64
        smem_tokens = NUM_STAGES * H * 2
        smem_mbar = NUM_STAGES * 2 * 8
        smem_size = smem_tokens + smem_mbar + 256

        self.kernel(
            hidden, probs, s2d_map, rdma_map,
            out_token_ptrs, out_prob_ptrs,
            num_tokens, H, R, E, NUM_STAGES, NUM_BLOCKS, WITH_PROBS,
        ).launch(
            grid=[NUM_BLOCKS, 1, 1],
            block=[NUM_THREADS, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        hidden: cute.Tensor,
        probs: cute.Tensor,
        s2d_map: cute.Tensor,
        rdma_map: cute.Tensor,
        out_token_ptrs: cute.Tensor,
        out_prob_ptrs: cute.Tensor,
        num_tokens: cutlass.Int32,
        H: cutlass.Constexpr,
        R: cutlass.Constexpr,
        E: cutlass.Constexpr,
        NUM_STAGES: cutlass.Constexpr,
        NUM_BLOCKS: cutlass.Constexpr,
        WITH_PROBS: cutlass.Constexpr,
    ):
        tidx = arch.thread_idx()[0]
        bidx = arch.block_idx()[0]
        warp_id = tidx // WARP_SIZE

        ER = E * R
        tx_bytes = H * 2  # bf16 token size in bytes

        # SMEM allocation
        smem = utils.SmemAllocator()
        token_buf = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((NUM_STAGES, H), stride=(H, 1)),
        )
        mbar_storage = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((NUM_STAGES, 2), stride=(2, 1)),
        )
        mbar_base = mbar_storage.iterator
        token_buf_base = token_buf.iterator

        # Initialize mbarriers
        if tidx == 0:
            arch.mbarrier_init_fence()
            for s in range(NUM_STAGES):
                arch.mbarrier_init(mbar_base + s * 2, cutlass.Int32(1))      # prod->cons
                arch.mbarrier_init(mbar_base + s * 2 + 1, cutlass.Int32(1))  # cons->prod
            arch.fence_proxy(kind="async")

        arch.sync_threads()

        # Pre-signal consumer->producer (all stages start free)
        if tidx == 0:
            for s in range(NUM_STAGES):
                arch.mbarrier_arrive(mbar_base + s * 2 + 1)

        arch.sync_threads()

        # =====================================================================
        # Warp 0: G2S Producer
        # =====================================================================
        if warp_id == 0:
            with arch.elect_one():
                stage = cutlass.Int32(0)
                cons_phase = cutlass.Int32(0)

                for token_id in range(bidx, num_tokens, NUM_BLOCKS):
                    # Check if token is needed
                    needed = rdma_map[token_id]
                    if needed != cutlass.Int8(0):
                        # Wait for consumer to free this stage
                        arch.mbarrier_wait(mbar_base + stage * 2 + 1, cons_phase)

                        # TMA G2S: copy token from GMEM to SMEM
                        g2s_op = cpasync.CopyBulkG2SOp()
                        g2s_atom = cute.make_copy_atom(g2s_op, cutlass.BFloat16)

                        src_slice = cute.make_tensor(
                            hidden.iterator + token_id * H,
                            cute.make_layout((H,)),
                        )
                        dst_slice = cute.make_tensor(
                            token_buf_base + stage * H,
                            cute.make_layout((H,)),
                        )
                        cute.copy(g2s_atom, src_slice, dst_slice,
                                  mbar_ptr=mbar_base + stage * 2)

                        # Signal expected TX bytes
                        arch.mbarrier_arrive_and_expect_tx(
                            mbar_base + stage * 2, tx_bytes,
                        )

                        # Advance stage
                        stage = stage + cutlass.Int32(1)
                        if stage == NUM_STAGES:
                            stage = cutlass.Int32(0)
                            cons_phase = cons_phase ^ cutlass.Int32(1)

        # =====================================================================
        # Warps 1-3: S2G Consumer
        # =====================================================================
        if warp_id >= 1:
            # Only warp 1 issues S2G (warps 2-3 idle for now — in the C++
            # kernel they distribute across destination ranks, but for the
            # POC we use a single warp)
            if warp_id == 1:
                with arch.elect_one():
                    stage = cutlass.Int32(0)
                    prod_phase = cutlass.Int32(0)
                    in_flight = cutlass.Int32(0)

                    for token_id in range(bidx, num_tokens, NUM_BLOCKS):
                        needed = rdma_map[token_id]
                        if needed != cutlass.Int8(0):
                            # Wait for producer to fill this stage
                            arch.mbarrier_wait(mbar_base + stage * 2, prod_phase)

                            # Read routing map and scatter to each rank
                            for r in range(R):
                                dst_idx = s2d_map[token_id * R + r]
                                if dst_idx >= cutlass.Int32(0):
                                    # TMA S2G: copy token from SMEM to destination
                                    s2g_op = cpasync.CopyBulkS2GOp()
                                    s2g_atom = cute.make_copy_atom(
                                        s2g_op, cutlass.BFloat16,
                                    )

                                    smem_slice = cute.make_tensor(
                                        token_buf_base + stage * H,
                                        cute.make_layout((H,)),
                                    )
                                    dst_ptr_val = out_token_ptrs[r]
                                    dst_base = cute.make_ptr(
                                        cutlass.BFloat16, dst_ptr_val,
                                        cute.AddressSpace.gmem,
                                        assumed_align=128,
                                    )
                                    dst_slice = cute.make_tensor(
                                        dst_base + dst_idx * H,
                                        cute.make_layout((H,)),
                                    )
                                    cute.copy(s2g_atom, smem_slice, dst_slice)

                                    # Also copy probs (element-wise, not TMA —
                                    # prob size is small: E*R * 4 bytes)
                                    if cutlass.const_expr(WITH_PROBS):
                                        prob_dst_ptr = out_prob_ptrs[r]
                                        prob_base = cute.make_ptr(
                                            cutlass.Float32, prob_dst_ptr,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        )
                                        for pe in range(ER):
                                            prob_val = probs[token_id * ER + pe]
                                            prob_dst = cute.make_tensor(
                                                prob_base + dst_idx * ER + pe,
                                                cute.make_layout((1,)),
                                            )
                                            prob_dst[0] = prob_val

                            # Commit S2G group and track in-flight
                            arch.cp_async_bulk_commit_group()
                            in_flight = in_flight + cutlass.Int32(1)

                            # If too many in-flight, wait for oldest
                            if in_flight >= NUM_STAGES:
                                arch.cp_async_bulk_wait_group(
                                    cutlass.Int32(NUM_STAGES - 1), read=True,
                                )
                                in_flight = in_flight - cutlass.Int32(1)

                                # Release oldest stage for producer
                                old_stage = (stage - cutlass.Int32(NUM_STAGES - 1) + NUM_STAGES) % NUM_STAGES
                                arch.mbarrier_arrive(mbar_base + old_stage * 2 + 1)

                            # Advance stage
                            stage = stage + cutlass.Int32(1)
                            if stage == NUM_STAGES:
                                stage = cutlass.Int32(0)
                                prod_phase = prod_phase ^ cutlass.Int32(1)

                    # Drain remaining in-flight S2G
                    arch.cp_async_bulk_wait_group(cutlass.Int32(0))
                    # Release remaining stages
                    for remaining in range(NUM_STAGES):
                        arch.mbarrier_arrive(mbar_base + remaining * 2 + 1)


# ---------------------------------------------------------------------------
_dispatch_cache = {}


def _dispatch_launch(
    hidden, probs, sparse_to_dense_map, rdma_to_attn_map,
    output_token_ptrs, output_prob_ptrs,
    T, H, R, E, with_probs, num_stages, num_blocks,
):
    """Launch the TMA dispatch kernel."""
    cache_key = (H, R, E, num_stages, num_blocks, with_probs)

    hidden_flat = hidden.reshape(-1)
    probs_flat = probs.reshape(-1) if probs is not None else torch.zeros(1, dtype=torch.float32, device=hidden.device)
    s2d_flat = sparse_to_dense_map.reshape(-1)
    rdma_flat = rdma_to_attn_map.reshape(-1).view(torch.int8)

    hidden_ct = from_dlpack(hidden_flat)
    hidden_ct.mark_layout_dynamic()
    probs_ct = from_dlpack(probs_flat)
    probs_ct.mark_layout_dynamic()
    s2d_ct = from_dlpack(s2d_flat)
    s2d_ct.mark_layout_dynamic()
    rdma_ct = from_dlpack(rdma_flat)
    rdma_ct.mark_layout_dynamic()
    ptrs_ct = from_dlpack(output_token_ptrs)
    ptrs_ct.mark_layout_dynamic()
    prob_ptrs_ct = from_dlpack(output_prob_ptrs)
    prob_ptrs_ct.mark_layout_dynamic()

    kernel = DispatchTMAKernel(
        H=H, R=R, E=E, NUM_STAGES=num_stages,
        NUM_BLOCKS=num_blocks, WITH_PROBS=1 if with_probs else 0,
    )

    if cache_key not in _dispatch_cache:
        compiled = cute.compile(
            kernel,
            hidden_ct, probs_ct, s2d_ct, rdma_ct,
            ptrs_ct, prob_ptrs_ct,
            T, H, R, E, num_stages, num_blocks,
            1 if with_probs else 0,
        )
        _dispatch_cache[cache_key] = compiled

    compiled = _dispatch_cache[cache_key]
    compiled(
        hidden_ct, probs_ct, s2d_ct, rdma_ct,
        ptrs_ct, prob_ptrs_ct,
        T,
    )
