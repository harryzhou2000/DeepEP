# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
CuTe DSL combine kernel for hybrid-ep (single-node, BF16, no unpermute fusion).

Architecture matching the C++ JIT combine kernel:
  - Gather: for each output token, read contributions from up to R source ranks
  - Reduce: accumulate in FP32 (BF16→FP32 for numerics, back to BF16 on store)
  - Routing: sparse_to_dense_map[token, rank] -> source position per rank

Single-GPU proof-of-concept. Source buffers are local GPU memory.
For real multi-GPU, these would be NVLink peer addresses (each rank's
dispatched buffer, written by the dispatch kernel).
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

WARP_SIZE = 32


def combine_cute(
    input_tokens: list,           # R tensors, each [max_in, H] bf16 (source per rank)
    input_probs: list,            # R tensors, each [max_in, E*R] f32 or None
    sparse_to_dense_map: torch.Tensor,  # [T, R] int32 (-1 = no contribution)
    rdma_to_attn_map: torch.Tensor,     # [T] bool
    num_tokens: int,
    num_ranks: int,
    num_experts_per_rank: int,
    hidden_dim: int,
    num_blocks: int = 24,
) -> tuple:
    """
    CuTe DSL combine: gather tokens from per-rank source buffers and reduce.

    Returns:
        output_tokens: [T, H] bf16 — accumulated token hidden states
        output_probs: [T, E*R] f32 or None — accumulated probs
    """
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

    # Pack source buffer pointers
    input_token_ptrs = torch.tensor(
        [t.data_ptr() for t in input_tokens], dtype=torch.int64, device=device,
    )
    if with_probs:
        input_prob_ptrs = torch.tensor(
            [p.data_ptr() for p in input_probs], dtype=torch.int64, device=device,
        )
    else:
        input_prob_ptrs = torch.zeros(R, dtype=torch.int64, device=device)

    # Allocate outputs
    output_tokens = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
    output_probs = torch.zeros(T, E * R, dtype=torch.float32, device=device) if with_probs else None

    _combine_kernel_launch(
        input_token_ptrs=input_token_ptrs,
        input_prob_ptrs=input_prob_ptrs,
        sparse_to_dense_map=sparse_to_dense_map,
        rdma_to_attn_map=rdma_to_attn_map,
        output_tokens=output_tokens,
        output_probs=output_probs,
        T=T, H=H, R=R, E=E,
        with_probs=with_probs,
        num_blocks=num_blocks,
    )

    return output_tokens, output_probs


class CombineKernel:
    """
    CuTe DSL combine kernel: gather + reduce.

    All 128 threads cooperate on each output token:
    - For each source rank with data, load H elements into FP32 accumulators
    - After all sources accumulated, convert FP32→BF16 and store to output
    """

    def __init__(self, H, R, E, NUM_BLOCKS, WITH_PROBS):
        self.H = H
        self.R = R
        self.E = E
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
        NUM_BLOCKS: cutlass.Constexpr,
        WITH_PROBS: cutlass.Constexpr,
    ):
        NUM_THREADS = 128
        # SMEM: source token staging [H] bf16 + accumulator [H] f32
        smem_src = H * 2     # bf16 staging
        smem_acc = H * 4     # f32 accumulator
        smem_prob = E * R * 4 if WITH_PROBS else 0  # f32 prob accumulator
        smem_size = smem_src + smem_acc + smem_prob + 128

        self.kernel(
            in_token_ptrs, in_prob_ptrs, s2d_map, rdma_map,
            out_tokens, out_probs,
            num_tokens,
            H, R, E, NUM_BLOCKS, WITH_PROBS,
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
        NUM_BLOCKS: cutlass.Constexpr,
        WITH_PROBS: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]

        ER = E * R
        NUM_THREADS = cutlass.Int32(128)
        elems_per_thread = (H + 127) // 128

        # SMEM
        smem = utils.SmemAllocator()
        # Source token staging buffer [H] bf16
        src_buf = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((H,), stride=(1,)),
        )
        # FP32 accumulator [H] f32
        acc_buf = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((H,), stride=(1,)),
        )
        if cutlass.const_expr(WITH_PROBS):
            prob_acc_buf = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((ER,), stride=(1,)),
            )

        for token_id in range(bidx, num_tokens, NUM_BLOCKS):
            needed = rdma_map[token_id]
            if needed != cutlass.Int8(0):
                # Initialize accumulator to zero
                for elem_idx in range(elems_per_thread):
                    h_idx = tidx + elem_idx * 128
                    if h_idx < H:
                        acc_buf[h_idx] = cutlass.Float32(0.0)

                if cutlass.const_expr(WITH_PROBS):
                    prob_elems = (ER + 127) // 128
                    for pe in range(prob_elems):
                        p_idx = tidx + pe * 128
                        if p_idx < ER:
                            prob_acc_buf[p_idx] = cutlass.Float32(0.0)

                cute.arch.sync_threads()

                # Accumulate from each source rank
                for r in range(R):
                    src_idx = s2d_map[token_id * R + r]
                    if src_idx >= cutlass.Int32(0):
                        # Load source token into SMEM staging
                        src_ptr_val = in_token_ptrs[r]
                        src_base = cute.make_ptr(
                            cutlass.BFloat16, src_ptr_val,
                            cute.AddressSpace.gmem, assumed_align=128,
                        )

                        # Cooperative load: all threads load a portion of H elements
                        for elem_idx in range(elems_per_thread):
                            h_idx = tidx + elem_idx * 128
                            if h_idx < H:
                                src_tensor = cute.make_tensor(
                                    src_base + src_idx * H + h_idx,
                                    cute.make_layout((1,)),
                                )
                                src_buf[h_idx] = src_tensor[0]

                        cute.arch.sync_threads()

                        # Accumulate: BF16 → FP32 add
                        for elem_idx in range(elems_per_thread):
                            h_idx = tidx + elem_idx * 128
                            if h_idx < H:
                                src_val = src_buf[h_idx].to(cutlass.Float32)
                                acc_buf[h_idx] = acc_buf[h_idx] + src_val

                        # Accumulate probs
                        if cutlass.const_expr(WITH_PROBS):
                            prob_src_ptr = in_prob_ptrs[r]
                            prob_base = cute.make_ptr(
                                cutlass.Float32, prob_src_ptr,
                                cute.AddressSpace.gmem, assumed_align=16,
                            )
                            prob_elems = (ER + 127) // 128
                            for pe in range(prob_elems):
                                p_idx = tidx + pe * 128
                                if p_idx < ER:
                                    prob_src = cute.make_tensor(
                                        prob_base + src_idx * ER + p_idx,
                                        cute.make_layout((1,)),
                                    )
                                    prob_acc_buf[p_idx] = prob_acc_buf[p_idx] + prob_src[0]

                        cute.arch.sync_threads()

                # Store: FP32 → BF16, write to output
                for elem_idx in range(elems_per_thread):
                    h_idx = tidx + elem_idx * 128
                    if h_idx < H:
                        out_tokens[token_id * H + h_idx] = acc_buf[h_idx].to(cutlass.BFloat16)

                if cutlass.const_expr(WITH_PROBS):
                    prob_elems = (ER + 127) // 128
                    for pe in range(prob_elems):
                        p_idx = tidx + pe * 128
                        if p_idx < ER:
                            out_probs[token_id * ER + p_idx] = prob_acc_buf[p_idx]

                cute.arch.sync_threads()


# ---------------------------------------------------------------------------
_combine_cache = {}


def _combine_kernel_launch(
    input_token_ptrs, input_prob_ptrs,
    sparse_to_dense_map, rdma_to_attn_map,
    output_tokens, output_probs,
    T, H, R, E, with_probs, num_blocks,
):
    """Launch the combine kernel."""
    cache_key = (H, R, E, num_blocks, with_probs)

    s2d_flat = sparse_to_dense_map.reshape(-1)
    rdma_flat = rdma_to_attn_map.reshape(-1).view(torch.int8)
    out_tok_flat = output_tokens.reshape(-1)
    if with_probs:
        out_prob_flat = output_probs.reshape(-1)
    else:
        out_prob_flat = torch.zeros(1, dtype=torch.float32, device=output_tokens.device)

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

    kernel = CombineKernel(
        H=H, R=R, E=E,
        NUM_BLOCKS=num_blocks,
        WITH_PROBS=1 if with_probs else 0,
    )

    if cache_key not in _combine_cache:
        compiled = cute.compile(
            kernel,
            ptrs_ct, prob_ptrs_ct, s2d_ct, rdma_ct,
            out_ct, out_prob_ct,
            T,
            H, R, E, num_blocks,
            1 if with_probs else 0,
        )
        _combine_cache[cache_key] = compiled

    compiled = _combine_cache[cache_key]
    compiled(
        ptrs_ct, prob_ptrs_ct, s2d_ct, rdma_ct,
        out_ct, out_prob_ct,
        T,
    )
