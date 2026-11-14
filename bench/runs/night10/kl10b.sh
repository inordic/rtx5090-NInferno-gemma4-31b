#!/bin/bash
# kl10b.sh ref   ARTIFACT TEXT KLDFILE LABEL   record ARTIFACT's top-32 per position with exact (BF16) activations and BF16 KV
# kl10b.sh exact ARTIFACT TEXT KLDFILE LABEL   the same numerics against a recorded reference (sanity check)
# kl10b.sh run   ARTIFACT TEXT KLDFILE LABEL   production's numerics (4-bit activations on prefill, FP8 KV) against it
# context 4096, stride 2048, as night 7
P=/opt/NInferno-up/build/apps/ninferno-perplexity
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
cd /root/gbench/n10; mkdir -p kld
mode=$1; art=$2; text=$3; kld=$4; label=$5
case $mode in
    ref)   timeout 1500 $P $art --text $text --context 4096 --stride 2048 --activations exact --kv-dtype bf16 --kld-out $kld > kld/$label.log 2>&1 ;;
    exact) timeout 1500 $P $art --text $text --context 4096 --stride 2048 --activations exact --kv-dtype bf16 --kld $kld > kld/$label.log 2>&1 ;;
    run)   timeout 1200 $P $art --text $text --context 4096 --stride 2048 --activations fast --kv-dtype fp8 --kld $kld > kld/$label.log 2>&1 ;;
esac
rc=$?
echo "$mode $label exit $rc: $(grep -E "scoring complete|mean KLD|top-1" kld/$label.log | tail -2 | tr '\n' ' ' | cut -c1-260)"
