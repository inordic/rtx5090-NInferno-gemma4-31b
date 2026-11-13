#!/bin/bash
# kld.sh ref ARTIFACT KLDFILE           record the reference's top-32 per position (exact activations, BF16 KV)
# kld.sh run ARTIFACT KLDFILE LABEL [exact]  KL divergence and top-1 agreement against it
#                                       (default: production numerics, 4-bit activations and FP8 KV)
# text: n7/chat.txt (harmless prompts with production's answers in Gemma 4 chat format), context 4096, stride 2048
P=/opt/NInferno-up/build/apps/ninferno-perplexity; T=/root/gbench/n7/chat.txt
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
cd /root/gbench/n7; mkdir -p kld
/root/gbench/srv.sh stop > /dev/null 2>&1  # the GPU must be free of test servers
if [ "$1" = ref ]; then
    unset NINFERNO_GEMMA4_EMBED_HOST  # GGUF references tie the token table to the head
    N=ref_$(basename $3 .kld)
    timeout 2400 $P $2 --text $T --context 4096 --stride 2048 --activations exact --kv-dtype bf16 --kld-out $3 > kld/$N.log 2>&1
    echo "REF $N exit $?: $(grep -iE "perplexity|ppl|tokens" kld/$N.log | tail -2 | tr '\n' ' ' | cut -c1-300)"
else
    if [ "$5" = exact ]; then X="--activations exact --kv-dtype bf16"; else X="--activations fast --kv-dtype fp8"; fi
    timeout 2400 $P $2 --text $T --context 4096 --stride 2048 $X --kld $3 > kld/$4.log 2>&1
    echo "KLD $4 exit $?: $(grep -E "mean KLD|top-1" kld/$4.log | tail -2 | tr '\n' ' ')"
fi
