# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
CuTe DSL combine kernel for hybrid-ep with TMA pipeline.

Architecture (single-node, BF16, 6 warps = 192 threads):
  Warps 0-1: G2S producer — for each output token, TMA G2S from each
             source rank's buffer into SMEM FIFO stages.
  Warps 2-5: Reduction consumer — wait for G2S data, accumulate BF16→FP32,
             then TMA S2G the result to output.

Pipeline: NUM_STAGES_G2S stages for the G2S FIFO, 1 S2G stage for output.
Each G2S stage holds one source contribution (one rank's token data).
The consumer serially accumulates all sources for each output token.

Single-GPU proof-of-concept.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

WARP_SIZE = 32


def combine_cute(
    input_tokens: list,              # R tensors [max_in, H] bf16
    input_probs: list,               # R tensors [max_in, E*R] f32 or None
    sparse_to_dense_map: torch.Tensor,  # [T, R] int32
    rdma_to_attn_map: torch.Tensor,     # [T] bool
    num_tokens: int,
    num_ranks: int,
    num_experts_per_rank: int,
    hidden_dim: int,
    num_stages: int = 8,
    num_blocks: int = 24,
) -> tuple:
    """CuTe DSL combine with TMA pipeline."""
    T = num_tokens
    H = hidden_dim
    R = num_ranks
    E = num_experts_per_rank
    with_probs = input_probs is not None and input_probs[0] is not None

    device = sparse_to_dense_map.device
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device_idx).multi_processor_count
    if num_blocks > sm_count:
        num_blocks = sm_count

    input_token_ptrs = torch.tensor(
        [t.data_ptr() for t in input_tokens], dtype=torch.int64, device=device,
    )
    if with_probs:
        input_prob_ptrs = torch.tensor(
            [p.data_ptr() for p in input_probs], dtype=torch.int64, device=device,
        )
    else:
        input_prob_ptrs = torch.zeros(R, dtype=torch.int64, device=device)

    output_tokens = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
    output_probs = torch.zeros(T, E * R, dtype=torch.float32, device=device) if with_probs else None

    _combine_launch(
        input_token_ptrs, input_prob_ptrs,
        sparse_to_dense_map, rdma_to_attn_map,
        output_tokens, output_probs,
        T, H, R, E, with_probs, num_stages, num_blocks,
    )

    return output_tokens, output_probs


class CombineTMAKernel:
    """Combine kernel with TMA G2S pipeline + cooperative FP32 reduction."""

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
        in_token_ptrs: cute.Tensor,
        in_prob_ptrs: cute.Tensor,
        s2d_map: cute.Tensor,
        rdma_map: cute.Tensor,
        out_tokens: cute.Tensor,
        out_probs: cute.Tensor,
        num_tokens: cutlass.Int32,
        H: cutlass.Constexpr,
        R: cutlass.Constexpr,
        E: cutlass.Constexpr,
        NUM_STAGES: cutlass.Constexpr,
        NUM_BLOCKS: cutlass.Constexpr,
        WITH_PROBS: cutlass.Constexpr,
    ):
        # 4 warps for reduction + 1 warp for G2S = 5 warps = 160 threads
        NUM_THREADS = 160
        ER = E * R
        # SMEM: G2S staging [STAGES][H] bf16 + accumulator [H] f32
        #       + S2G staging [H] bf16 + mbarrier [STAGES][2] u64
        #       + flag per stage (is_last_source) [STAGES] i8
        smem_g2s = NUM_STAGES * H * 2
        smem_acc = H * 4        # f32 accumulator
        smem_s2g = H * 2        # bf16 output staging for TMA S2G
        smem_mbar = NUM_STAGES * 2 * 8
        smem_prob_acc = ER * 4 if WITH_PROBS else 0
        smem_size = smem_g2s + smem_acc + smem_s2g + smem_mbar + smem_prob_acc + 512

        self.kernel(
            in_token_ptrs, in_prob_ptrs, s2d_map, rdma_map,
            out_tokens, out_probs,
            num_tokens, H, R, E, NUM_STAGES, NUM_BLOCKS, WITH_PROBS,
        ).launch(
            grid=[NUM_BLOCKS, 1, 1],
            block=[NUM_THREADS, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        in_token_ptrs: cute.Tensor,
        in_prob_ptrs: cute.Tensor,
        s2d_map: cute.Tensor,
        rdma_map: cute.Tensor,
        out_tokens: cute.Tensor,
        out_probs: cute.Tensor,
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
        tx_bytes = H * 2  # bf16 token

        # Warp layout: warp 0 = G2S producer, warps 1-4 = reduction consumer
        G2S_WARP = 0
        RED_WARP_START = 1
        RED_WARP_COUNT = 4
        RED_THREADS = RED_WARP_COUNT * WARP_SIZE  # 128

        # SMEM allocation
        smem = utils.SmemAllocator()
        g2s_buf = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((NUM_STAGES, H), stride=(H, 1)),
        )
        acc_buf = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((H,), stride=(1,)),
        )
        s2g_buf = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((H,), stride=(1,)),
        )
        mbar_storage = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((NUM_STAGES, 2), stride=(2, 1)),
        )
        if cutlass.const_expr(WITH_PROBS):
            prob_acc_buf = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((ER,), stride=(1,)),
            )

        mbar_base = mbar_storage.iterator
        g2s_base = g2s_buf.iterator

        # Initialize mbarriers
        if tidx == 0:
            arch.mbarrier_init_fence()
            for s in range(NUM_STAGES):
                arch.mbarrier_init(mbar_base + s * 2, cutlass.Int32(1))      # prod->cons
                arch.mbarrier_init(mbar_base + s * 2 + 1, cutlass.Int32(1))  # cons->prod
            arch.fence_proxy(kind="async")

        arch.sync_threads()

        # Pre-signal consumer->producer for all stages
        if tidx == 0:
            for s in range(NUM_STAGES):
                arch.mbarrier_arrive(mbar_base + s * 2 + 1)

        arch.sync_threads()

        # Thread-local identity within reduction group
        red_tidx = tidx - RED_WARP_START * WARP_SIZE  # 0..127
        elems_per_thread = (H + RED_THREADS - 1) // RED_THREADS

        # =====================================================================
        # Warp 0: G2S Producer — load source contributions for each output token
        # =====================================================================
        if warp_id == G2S_WARP:
            with arch.elect_one():
                stage = cutlass.Int32(0)
                cons_phase = cutlass.Int32(0)

                for token_id in range(bidx, num_tokens, NUM_BLOCKS):
                    needed = rdma_map[token_id]
                    if needed != cutlass.Int8(0):
                        # Load each source rank's contribution
                        for r in range(R):
                            src_idx = s2d_map[token_id * R + r]
                            if src_idx >= cutlass.Int32(0):
                                # Wait for consumer to free stage
                                arch.mbarrier_wait(mbar_base + stage * 2 + 1, cons_phase)

                                # TMA G2S from source rank's buffer
                                g2s_op = cpasync.CopyBulkG2SOp()
                                g2s_atom = cute.make_copy_atom(g2s_op, cutlass.BFloat16)

                                src_ptr_val = in_token_ptrs[r]
                                src_base = cute.make_ptr(
                                    cutlass.BFloat16, src_ptr_val,
                                    cute.AddressSpace.gmem, assumed_align=128,
                                )
                                src_slice = cute.make_tensor(
                                    src_base + src_idx * H,
                                    cute.make_layout((H,)),
                                )
                                dst_slice = cute.make_tensor(
                                    g2s_base + stage * H,
                                    cute.make_layout((H,)),
                                )
                                cute.copy(g2s_atom, src_slice, dst_slice,
                                          mbar_ptr=mbar_base + stage * 2)
                                arch.mbarrier_arrive_and_expect_tx(
                                    mbar_base + stage * 2, tx_bytes,
                                )

                                # Advance
                                stage = stage + cutlass.Int32(1)
                                if stage == NUM_STAGES:
                                    stage = cutlass.Int32(0)
                                    cons_phase = cons_phase ^ cutlass.Int32(1)

        # =====================================================================
        # Warps 1-4: Reduction Consumer — accumulate sources, store output
        # =====================================================================
        if warp_id >= RED_WARP_START:
            stage = cutlass.Int32(0)
            prod_phase = cutlass.Int32(0)

            for token_id in range(bidx, num_tokens, NUM_BLOCKS):
                needed = rdma_map[token_id]
                if needed != cutlass.Int8(0):
                    # Initialize FP32 accumulator to zero
                    for ei in range(elems_per_thread):
                        h_idx = red_tidx + ei * RED_THREADS
                        if h_idx < H:
                            acc_buf[h_idx] = cutlass.Float32(0.0)

                    if cutlass.const_expr(WITH_PROBS):
                        prob_elems = (ER + RED_THREADS - 1) // RED_THREADS
                        for pe in range(prob_elems):
                            p_idx = red_tidx + pe * RED_THREADS
                            if p_idx < ER:
                                prob_acc_buf[p_idx] = cutlass.Float32(0.0)

                    # Accumulate from each source rank
                    for r in range(R):
                        src_idx = s2d_map[token_id * R + r]
                        if src_idx >= cutlass.Int32(0):
                            # Wait for G2S to fill this stage
                            arch.mbarrier_wait(mbar_base + stage * 2, prod_phase)

                            # Accumulate tokens: BF16 → FP32
                            for ei in range(elems_per_thread):
                                h_idx = red_tidx + ei * RED_THREADS
                                if h_idx < H:
                                    src_val = g2s_buf[stage, h_idx].to(cutlass.Float32)
                                    acc_buf[h_idx] = acc_buf[h_idx] + src_val

                            # Accumulate probs (element-wise from source buffer)
                            if cutlass.const_expr(WITH_PROBS):
                                prob_src_ptr = in_prob_ptrs[r]
                                prob_base = cute.make_ptr(
                                    cutlass.Float32, prob_src_ptr,
                                    cute.AddressSpace.gmem, assumed_align=16,
                                )
                                prob_elems = (ER + RED_THREADS - 1) // RED_THREADS
                                for pe in range(prob_elems):
                                    p_idx = red_tidx + pe * RED_THREADS
                                    if p_idx < ER:
                                        prob_src = cute.make_tensor(
                                            prob_base + src_idx * ER + p_idx,
                                            cute.make_layout((1,)),
                                        )
                                        prob_acc_buf[p_idx] = prob_acc_buf[p_idx] + prob_src[0]

                            # Release stage for producer (one thread only)
                            if red_tidx == 0:
                                arch.mbarrier_arrive(mbar_base + stage * 2 + 1)

                            # Advance
                            stage = stage + cutlass.Int32(1)
                            if stage == NUM_STAGES:
                                stage = cutlass.Int32(0)
                                prod_phase = prod_phase ^ cutlass.Int32(1)

                    # Convert FP32 → BF16 and store to output
                    for ei in range(elems_per_thread):
                        h_idx = red_tidx + ei * RED_THREADS
                        if h_idx < H:
                            out_tokens[token_id * H + h_idx] = acc_buf[h_idx].to(cutlass.BFloat16)

                    if cutlass.const_expr(WITH_PROBS):
                        prob_elems = (ER + RED_THREADS - 1) // RED_THREADS
                        for pe in range(prob_elems):
                            p_idx = red_tidx + pe * RED_THREADS
                            if p_idx < ER:
                                out_probs[token_id * ER + p_idx] = prob_acc_buf[p_idx]


# ---------------------------------------------------------------------------
_combine_cache = {}


def _combine_launch(
    input_token_ptrs, input_prob_ptrs,
    sparse_to_dense_map, rdma_to_attn_map,
    output_tokens, output_probs,
    T, H, R, E, with_probs, num_stages, num_blocks,
):
    """Launch the TMA combine kernel."""
    cache_key = (H, R, E, num_stages, num_blocks, with_probs)

    s2d_flat = sparse_to_dense_map.reshape(-1)
    rdma_flat = rdma_to_attn_map.reshape(-1).view(torch.int8)
    out_tok_flat = output_tokens.reshape(-1)
    out_prob_flat = output_probs.reshape(-1) if with_probs else torch.zeros(1, dtype=torch.float32, device=output_tokens.device)

    ptrs_ct = from_dlpack(input_token_ptrs)
    ptrs_ct.mark_layout_dynamic()
    prob_ptrs_ct = from_dlpack(input_prob_ptrs)
    prob_ptrs_ct.mark_layout_dynamic()
    s2d_ct = from_dlpack(s2d_flat)
    s2d_ct.mark_layout_dynamic()
    rdma_ct = from_dlpack(rdma_flat)
    rdma_ct.mark_layout_dynamic()
    out_ct = from_dlpack(out_tok_flat)
    out_ct.mark_layout_dynamic()
    out_prob_ct = from_dlpack(out_prob_flat)
    out_prob_ct.mark_layout_dynamic()

    kernel = CombineTMAKernel(
        H=H, R=R, E=E, NUM_STAGES=num_stages,
        NUM_BLOCKS=num_blocks, WITH_PROBS=1 if with_probs else 0,
    )

    if cache_key not in _combine_cache:
        compiled = cute.compile(
            kernel,
            ptrs_ct, prob_ptrs_ct, s2d_ct, rdma_ct,
            out_ct, out_prob_ct,
            T, H, R, E, num_stages, num_blocks,
            1 if with_probs else 0,
        )
        _combine_cache[cache_key] = compiled

    compiled = _combine_cache[cache_key]
    compiled(
        ptrs_ct, prob_ptrs_ct, s2d_ct, rdma_ct,
        out_ct, out_prob_ct,
        T,
    )
