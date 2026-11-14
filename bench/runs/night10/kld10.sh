#!/bin/bash
# kld10.sh ref ARTIFACT TEXT KLDFILE         the reference's top-32 per position (exact activations, BF16 KV)
# kld10.sh run ARTIFACT TEXT KLDFILE LABEL   KL divergence and top-1 agreement against it, with production's
#                                            numerics (4-bit activations on prefill, FP8 KV)
# context 4096, stride 2048, as night 7
P=/opt/NInferno-up/build/apps/ninferno-perplexity
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
cd /root/gbench/n10; mkdir -p kld
if [ "$1" = ref ]; then
    unset NINFERNO_GEMMA4_EMBED_HOST  # GGUF references tie the token table to the head
    N=ref_$(basename $4 .kld)
    timeout 2400 $P $2 --text $3 --context 4096 --stride 2048 --activations exact --kv-dtype bf16 --kld-out $4 > kld/$N.log 2>&1
    echo "REF $N exit $?: $(grep -E "scoring complete" kld/$N.log | tail -1)"
else
    timeout 1200 $P $2 --text $3 --context 4096 --stride 2048 --activations fast --kv-dtype fp8 --kld $4 > kld/$5.log 2>&1
    echo "KLD $5 exit $?: $(grep -E "mean KLD|top-1" kld/$5.log | tail -2 | tr '\n' ' ')"
fi
