#!/bin/bash
# n2h_turbo: night 2, batch H - LilaRest NVFP4-turbo (from Google's BF16 weights) scored 171/280 on
# MMLU-Pro in batch D (production 147-152), but prefilled 7-9% slower: its query/key/value carry
# different NVFP4 divisors, so the converter kept them as three GEMMs. turbo-qkv is the same weights
# packed into one QKV plane under the query's divisor (the per-head norms cancel the factor).
# Prefill comes first (the user's rule: never trade prefill away), so H1 compares prefill of
# production, turbo and turbo-qkv in one session, all with the context levers (FP8 rings, host token
# table) and Google's regular assistant for the turbo builds (batch D could not fit it at 155K).
# After batch G, behind a 5-minute idle gate; the fast unit comes back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2G DONE" n2g_vision2.log 2>/dev/null; do sleep 30; done
until grep -q "CONV DONE" conv_turbo_fused.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
S=/opt/NInferno/build/apps-final5/ninferno-serve
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
T=/opt/models/dl3/turbo-nvfp4.r31.ninferno
TQ=/opt/models/dl3/turbo-qkv-nvfp4.r31.ninferno
[ -f $TQ ] || { step "turbo-qkv missing: its runs use turbo"; TQ=$T; }
DR=/opt/models/dl2/assistant.q8full.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
LV="--kv-dtype fp8 --sliding-kv-dtype fp8"
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_EMBED_HOST=1
pp() {  # pp <label> <artifact> <drafter>: prefill at 155K with the levers
    step "H1 prefill: $1"
    ./srv_final5.sh start H1-$1 $2 --max-context 155000 --prefill-chunk 1024 $LV --spec mtp --drafter $3 --draft-tokens 7 && \
        python3 gbench.py --label "H1 $1 levers 155K pp" --pp 2048,8192,32768,65536 --tg "" --skip-chat --reps 2
    ./srv.sh stop
}
pp production $QN $DQ
pp turbo $T $DR
pp turbo-qkv $TQ $DR
for d in "regular:$DR:7" "regular-k5:$DR:5" "qat:$DQ:7"; do
    IFS=: read name drafter k <<< "$d"
    step "H2 decode: turbo-qkv with the $name drafter (K$k)"
    ./srv_final5.sh start H2-$name $TQ --max-context 155000 --prefill-chunk 1024 $LV --spec mtp --drafter $drafter --draft-tokens $k && \
        python3 gbench.py --label "H2 turbo-qkv $name drafter K$k levers 155K" --pp "" --reps 1 && \
        python3 editbench.py --label "H2 turbo-qkv $name drafter K$k"
    ./srv.sh stop
done
step "H3 turbo-qkv: agent turns and needles, regular drafter"
./srv_final5.sh start H3-turbo-qkv $TQ --max-context 155000 --prefill-chunk 1024 $LV --spec mtp --drafter $DR --draft-tokens 7 && \
    python3 turnprobe.py http://127.0.0.1:8080 && \
    python3 needle.py --label "H3 turbo-qkv needles" --lengths 32000,140000 --depths 0.5
./srv.sh stop
step "H4 turbo-qkv quality (4 lanes)"
./srv_final5.sh start qH4-turbo-qkv $TQ --max-context 4096 --max-concurrency 4 $LV --spec mtp --drafter $DR --draft-tokens 7 && \
    python3 geval.py --label "alt-turbo-qkv" --suites gsm8k,mmlupro,mmlu,humaneval --conc 4
./srv.sh stop
for d in "regular:$DR" "qat:$DQ"; do
    printf "CEIL turbo-qkv levers drafter %-8s " ${d%%:*}
    $S $TQ --plan $LV --max-context 262144 --prefill-chunk 1024 --spec mtp --drafter ${d#*:} --draft-tokens 7 2>&1 \
        | grep -oE "with up to [0-9]+ tokens" | head -1; echo
done
step "restoring the fast unit"
./deploy_fast.sh
step "N2H DONE"
