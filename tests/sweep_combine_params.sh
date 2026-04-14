#!/bin/bash
# sweep_combine_params.sh — Sweep combine kernel parameters and record results.
# Run inside the container on the compute node.
# Usage: bash tests/sweep_combine_params.sh

set -euo pipefail

cd /home/scratch.hhanyu_gpu/projects/moe/DeepEP

# Fixed parameters
export HIDDEN_DIM=512
export NUM_TOKENS_PER_RANK=8192
export NUM_LOCAL_EXPERTS=32
export TOPK=36
export NUM_SMS_DISPATCH=32
export NUM_OF_STAGES_G2S_COMBINE_API=64
export NUM_OF_STAGES_S2G_COMBINE_API=8

OUTFILE="/home/scratch.hhanyu_gpu/projects/moe/data/combine_param_sweep.csv"

echo "sms_combine,batch_size,group_size,chunk_size,dispatch_kernel_us,combine_wp_kernel_us,combine_np_kernel_us,dispatch_noprob_kernel_us,fused_dispatch_kernel_us,fused_combine_kernel_us" > "$OUTFILE"

for SMS_COMBINE in 32 64; do
for BATCH_SIZE in 4 8 16 32; do
for GROUP_SIZE in 1 2 4; do
for CHUNK_SIZE in 16 32 64; do
    # group_size must divide chunk_size
    if (( CHUNK_SIZE % GROUP_SIZE != 0 )); then
        continue
    fi

    export NUM_SMS_COMBINE=$SMS_COMBINE
    export NUM_OF_COMBINE_REDUCE_BATCH_SIZE_API=$BATCH_SIZE
    export NUM_OF_TOKENS_PER_GROUP_COMBINE_API=$GROUP_SIZE
    export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=$CHUNK_SIZE
    export NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=$CHUNK_SIZE
    export NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=$CHUNK_SIZE

    echo "=== SMS=$SMS_COMBINE BATCH=$BATCH_SIZE GROUP=$GROUP_SIZE CHUNK=$CHUNK_SIZE ===" >&2

    OUTPUT=$(python tests/test_hybrid_ep.py --num-processes 8 2>&1) || {
        echo "  FAILED" >&2
        continue
    }

    # Extract kernel benchmark lines (avg_t values)
    dispatch_t=$(echo "$OUTPUT" | grep "^dispatch kernel (BF16)" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    combine_wp_t=$(echo "$OUTPUT" | grep "^combine kernel (w/ probs)" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    combine_np_t=$(echo "$OUTPUT" | grep "^combine kernel (no probs)" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    dispatch_np_t=$(echo "$OUTPUT" | grep "^dispatch no-prob kernel" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    fused_dispatch_t=$(echo "$OUTPUT" | grep "^fused dispatch" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")
    fused_combine_t=$(echo "$OUTPUT" | grep "^fused combine" | grep -oP 'avg_t=\K[0-9.]+' || echo "NA")

    echo "$SMS_COMBINE,$BATCH_SIZE,$GROUP_SIZE,$CHUNK_SIZE,$dispatch_t,$combine_wp_t,$combine_np_t,$dispatch_np_t,$fused_dispatch_t,$fused_combine_t" >> "$OUTFILE"
    echo "  dispatch=${dispatch_t}us combine_wp=${combine_wp_t}us combine_np=${combine_np_t}us fused_combine=${fused_combine_t}us" >&2

done
done
done
done

echo "Results saved to $OUTFILE" >&2
