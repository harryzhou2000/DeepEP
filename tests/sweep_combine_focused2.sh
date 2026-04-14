#!/bin/bash
# sweep_combine_focused2.sh — Test smaller batch sizes and group=1 vs 2 with more SMs.
set -euo pipefail

cd /home/scratch.hhanyu_gpu/projects/moe/DeepEP

export HIDDEN_DIM=512
export NUM_TOKENS_PER_RANK=8192
export NUM_LOCAL_EXPERTS=32
export TOPK=36
export NUM_SMS_DISPATCH=32
export NUM_OF_STAGES_G2S_COMBINE_API=64
export NUM_OF_STAGES_S2G_COMBINE_API=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=64
export NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=64
export NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=64

OUTFILE="/home/scratch.hhanyu_gpu/projects/moe/data/combine_param_sweep2.csv"

echo "sms_combine,batch_size,group_size,chunk_size,dispatch_kernel_us,combine_wp_kernel_us,combine_np_kernel_us,dispatch_noprob_kernel_us,fused_dispatch_kernel_us,fused_combine_kernel_us" > "$OUTFILE"

PORT=30100

for SMS_COMBINE in 32 64 128; do
for BATCH_SIZE in 1 2 4 8 16 32; do
for GROUP_SIZE in 1 2 4; do
    PORT=$((PORT + 1))

    export NUM_SMS_COMBINE=$SMS_COMBINE
    export NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=$BATCH_SIZE
    export NUM_OF_TOKENS_PER_GROUP_COMBINE_API=$GROUP_SIZE
    export MASTER_PORT=$PORT

    echo "=== SMS=$SMS_COMBINE BATCH=$BATCH_SIZE GROUP=$GROUP_SIZE ===" >&2

    OUTPUT=$(timeout 300 python tests/test_hybrid_ep.py --num-processes 8 2>&1) || {
        echo "  FAILED" >&2
        pkill -f "test_hybrid_ep" 2>/dev/null || true
        sleep 2
        continue
    }

    dispatch_t=$(echo "$OUTPUT" | grep "^dispatch kernel (BF16)" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    combine_wp_t=$(echo "$OUTPUT" | grep "^combine kernel (w/ probs)" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    combine_np_t=$(echo "$OUTPUT" | grep "^combine kernel (no probs)" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    dispatch_np_t=$(echo "$OUTPUT" | grep "^dispatch no-prob kernel" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    fused_dispatch_t=$(echo "$OUTPUT" | grep "^fused dispatch" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    fused_combine_t=$(echo "$OUTPUT" | grep "^fused combine" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")

    echo "$SMS_COMBINE,$BATCH_SIZE,$GROUP_SIZE,64,$dispatch_t,$combine_wp_t,$combine_np_t,$dispatch_np_t,$fused_dispatch_t,$fused_combine_t" >> "$OUTFILE"
    echo "  combine_np=${combine_np_t}us combine_wp=${combine_wp_t}us" >&2

done
done
done

echo "Results saved to $OUTFILE" >&2
cat "$OUTFILE"
