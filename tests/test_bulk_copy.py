"""Test raw cp.async.bulk G2S and S2G with mbarrier in CuTe DSL."""
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.runtime import from_dlpack
import torch

WARP_SIZE = 32
H = 128  # Hidden dim (elements, bf16)
NUM_STAGES = 4
NUM_TOKENS = 8


@cute.kernel
def bulk_copy_kernel(
    src: cute.Tensor,    # flat bf16 [NUM_TOKENS * H]
    dst: cute.Tensor,    # flat bf16 [NUM_TOKENS * H]
    num_tokens: cutlass.Int32,
    H: cutlass.Constexpr,
    NUM_STAGES: cutlass.Constexpr,
):
    """
    Test kernel: G2S bulk copy + S2G bulk copy with mbarrier pipeline.

    Warp 0 (producer): TMA G2S from src to SMEM
    Warp 1 (consumer): TMA S2G from SMEM to dst

    This proves the cp.async.bulk + mbarrier pattern works in CuTe DSL.
    """
    tidx = cute.arch.thread_idx()[0]
    warp_id = tidx // WARP_SIZE

    # SMEM
    smem = utils.SmemAllocator()
    # Token staging buffer: [NUM_STAGES, H] bf16
    token_buf = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((NUM_STAGES, H), stride=(H, 1)),
    )
    # mbarrier array: [NUM_STAGES, 2] — [stage][0]=producer->consumer, [stage][1]=consumer->producer
    mbar_storage = smem.allocate_tensor(
        cutlass.Int64,
        cute.make_layout((NUM_STAGES, 2), stride=(2, 1)),
    )

    # Get base pointer for mbarrier array
    mbar_base = mbar_storage.iterator  # pointer to [NUM_STAGES * 2] int64 in SMEM

    # Initialize mbarriers (thread 0 only)
    if tidx == 0:
        arch.mbarrier_init_fence()
        for s in range(NUM_STAGES):
            # mbar_base + s*2 + 0 = producer->consumer
            # mbar_base + s*2 + 1 = consumer->producer
            arch.mbarrier_init(mbar_base + s * 2, cutlass.Int32(1))
            arch.mbarrier_init(mbar_base + s * 2 + 1, cutlass.Int32(1))
        arch.fence_proxy(kind="async")

    cute.arch.sync_threads()

    # Pre-signal consumer->producer for all stages (all slots start free)
    if tidx == 0:
        for s in range(NUM_STAGES):
            arch.mbarrier_arrive(mbar_base + s * 2 + 1)

    cute.arch.sync_threads()

    if warp_id == 0:
        # ===== PRODUCER: G2S bulk copy =====
        # Only elected thread issues TMA
        with arch.elect_one():
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            tx_bytes = H * 2  # bf16 = 2 bytes

            for token_id in range(num_tokens):
                # Wait for consumer to free this stage
                arch.mbarrier_wait(mbar_base + stage * 2 + 1, phase)

                # Issue bulk G2S copy
                g2s_op = cpasync.CopyBulkG2SOp()
                g2s_atom = cute.make_copy_atom(g2s_op, cutlass.BFloat16)

                src_slice = cute.make_tensor(
                    src.iterator + token_id * H,
                    cute.make_layout((H,)),
                )
                smem_stage_ptr = token_buf.iterator + stage * H
                dst_slice = cute.make_tensor(
                    smem_stage_ptr,
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
                    phase = phase ^ cutlass.Int32(1)

    if warp_id == 1:
        # ===== CONSUMER: S2G bulk copy =====
        with arch.elect_one():
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)

            for token_id in range(num_tokens):
                # Wait for producer to fill this stage
                arch.mbarrier_wait(mbar_base + stage * 2, phase)

                # Issue bulk S2G copy
                s2g_op = cpasync.CopyBulkS2GOp()
                s2g_atom = cute.make_copy_atom(s2g_op, cutlass.BFloat16)

                smem_stage_ptr = token_buf.iterator + stage * H
                src_slice = cute.make_tensor(
                    smem_stage_ptr,
                    cute.make_layout((H,)),
                )
                dst_slice = cute.make_tensor(
                    dst.iterator + token_id * H,
                    cute.make_layout((H,)),
                )
                cute.copy(s2g_atom, src_slice, dst_slice)
                arch.cp_async_bulk_commit_group()
                arch.cp_async_bulk_wait_group(cutlass.Int32(0))

                # Release stage for producer
                arch.mbarrier_arrive(mbar_base + stage * 2 + 1)

                # Advance stage
                stage = stage + cutlass.Int32(1)
                if stage == NUM_STAGES:
                    stage = cutlass.Int32(0)
                    phase = phase ^ cutlass.Int32(1)


@cute.jit
def launch(src: cute.Tensor, dst: cute.Tensor, num_tokens: cutlass.Int32,
           H: cutlass.Constexpr, NUM_STAGES: cutlass.Constexpr):
    bulk_copy_kernel(src, dst, num_tokens, H, NUM_STAGES).launch(
        grid=[1, 1, 1],
        block=[64, 1, 1],  # 2 warps
        smem=NUM_STAGES * H * 2 + NUM_STAGES * 2 * 8 + 256,
    )


# Test
device = torch.device("cuda:0")
src = torch.randn(NUM_TOKENS, H, dtype=torch.bfloat16, device=device)
dst = torch.zeros(NUM_TOKENS, H, dtype=torch.bfloat16, device=device)

src_flat = src.reshape(-1)
dst_flat = dst.reshape(-1)
src_ct = from_dlpack(src_flat)
src_ct.mark_layout_dynamic()
dst_ct = from_dlpack(dst_flat)
dst_ct.mark_layout_dynamic()

print(f"Testing bulk copy: {NUM_TOKENS} tokens, H={H}, stages={NUM_STAGES}")
import time
t0 = time.time()
launch(src_ct, dst_ct, NUM_TOKENS, H, NUM_STAGES)
torch.cuda.synchronize()
compile_ms = (time.time() - t0) * 1000
print(f"Compile + run: {compile_ms:.1f}ms")

match = torch.equal(src, dst)
if match:
    print("PASS: src == dst")
else:
    mismatches = (src != dst).sum().item()
    print(f"FAIL: {mismatches}/{src.numel()} mismatches")
    idx = (src != dst).nonzero()[:3]
    for i in range(min(3, len(idx))):
        r, c = idx[i].tolist()
        print(f"  [{r},{c}]: src={src[r,c].item():.4f}, dst={dst[r,c].item():.4f}")
