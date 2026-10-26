#!/bin/bash
# n2a_context: night 2, batch A - context levers on the fast profile (apps-final5 = 6ff9931 + the
# host-mapped token table behind NINFERNO_GEMMA4_EMBED_HOST=1):
#  1. planner ceilings for base / FP8 sliding rings / host table / both, text and with vision;
#  2. speed, agent turns and the quality gate on both levers at the largest text context that fits.
# Starts after 22:00 UTC (23:00 CET) once 8080 has been idle 45 minutes; the fast unit comes back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
# started on request (2026-10-26 afternoon): no night wait
step "waiting for 5 idle minutes on 8080"
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
S=/opt/NInferno/build/apps-final5/ninferno-serve
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
QNV=/opt/models/dl2/qat-nvfp4-vision.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
ceiling() {  # ceiling <artifact> <chunk> [args] - the planner's largest context for 1 stream
    $S $1 --plan --kv-dtype fp8 --max-context 262144 --prefill-chunk $2 --spec mtp --drafter $DQ --draft-tokens 7 "${@:3}" 2>&1 \
        | grep -oE "with up to [0-9]+ tokens|rung [0-9]+ of [0-9]+: [0-9,]+ tokens x 1 context \([0-9,]+ KV tokens\), [0-9]+ cached" | head -2 | tr "\n" " "; echo
}
step "planner ceilings (chunk 1024 / 512)"
for ch in 1024 512; do
    for v in base rings host both; do
        case $v in
            base) E=0; A="";; rings) E=0; A="--sliding-kv-dtype fp8";; host) E=1; A="";; both) E=1; A="--sliding-kv-dtype fp8";;
        esac
        printf "CEIL text   chunk %s %-5s " $ch $v; NINFERNO_GEMMA4_EMBED_HOST=$E ceiling $QN $ch $A
        printf "CEIL vision chunk %s %-5s " $ch $v; NINFERNO_GEMMA4_EMBED_HOST=$E ceiling $QNV $ch $A --vision
    done
done
CTX=$(NINFERNO_GEMMA4_EMBED_HOST=1 $S $QN --plan --kv-dtype fp8 --sliding-kv-dtype fp8 --max-context 262144 --prefill-chunk 1024 \
      --spec mtp --drafter $DQ --draft-tokens 7 2>&1 | grep -oE "with up to [0-9]+ tokens" | head -1 | grep -oE "[0-9]+")
[ -z "$CTX" ] && CTX=262144
[ "$CTX" -gt 262144 ] && CTX=262144
step "B1 both levers at $CTX, chunk 1024: speed, turns"
sed "s#^S=.*#S=$S#" srv.sh > srv_final5.sh; chmod +x srv_final5.sh
NINFERNO_GEMMA4_EMBED_HOST=1 ./srv_final5.sh start B1-levers $QN --max-context $CTX --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 \
    --spec mtp --drafter $DQ --draft-tokens 7 && \
    grep -hE "plan \| rung|context cache|capacity \| KV" logs/B1-levers.log | sed -E "s/^.*INFO +//" | cut -c1-200 && \
    python3 gbench.py --label "B1 final5 rings-fp8 embed-host ctx$CTX chunk1024 K7" --pp 2048,8192,32768,65536 --reps 1 && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 turnprobe.py http://127.0.0.1:8080 && \
    python3 needle.py --label "B1 levers needles" --lengths 32000,72000,140000,$(( CTX * 9 / 10 )) --depths 0.5
./srv.sh stop
step "B2 quality gate: both levers (4 lanes)"
NINFERNO_GEMMA4_EMBED_HOST=1 ./srv_final5.sh start q-B2 $QN --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --sliding-kv-dtype fp8 \
    --spec mtp --drafter $DQ --draft-tokens 7 && \
    python3 geval.py --label nvfp4-final5-rings-fp8-embed-host --suites gsm8k,mmlupro,mmlu,humaneval --conc 4
./srv.sh stop
step "restoring the fast unit"
./deploy_fast.sh
step "N2A DONE"
