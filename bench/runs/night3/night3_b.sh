#!/bin/bash
# night3_b: after night3_a and the rebased build, behind the idle gate. Production's patches on
# upstream master (apps-up1) against apps-final10 at production flags (prefill, decode, agent turns,
# code edits, hard50), then both at the 262K unit's flags; then the fast unit back.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N3A DONE" night3_a.out 2>/dev/null && grep -qE "UP_DONE|BUILD_up1_FAILED|CMAKE_FAILED" build_up.out 2>/dev/null; do sleep 30; done
grep -q BUILD_up1_OK build_up.out || { step "rebased build failed: nothing to compare"; step "N3B DONE"; exit 1; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1
M=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
PROD="--vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja"
Q4=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
P262="--max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter /opt/models/dl2/qat-assistant.q40qat.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja"
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
mkdir -p hard50
efail() { grep -hE "Engine failed" logs/$1.log | sed -E "s/^.*ninferno: //" | cut -c1-300 | head -2; }

for b in final10 up1; do
    BIN=/opt/NInferno/build/apps-$b/ninferno-serve; [ $b = up1 ] && BIN=/opt/NInferno-up/build/apps-up1/ninferno-serve
    step "B $b at production flags"
    NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 SRV_BIN=$BIN ./srv_any.sh start B-$b $M $PROD && \
        python3 gbench.py --label "N3 $b production flags" --pp 2048,8192,32768,65536 --reps 2 && \
        python3 turnprobe.py http://127.0.0.1:8080 && \
        python3 editbench.py --label "N3 $b" && \
        python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag n3_$b --out hard50/n3_$b.json 2>&1 | tail -4
    efail B-$b
    ./srv.sh stop
done
for b in final10 up1; do
    BIN=/opt/NInferno/build/apps-$b/ninferno-serve; [ $b = up1 ] && BIN=/opt/NInferno-up/build/apps-up1/ninferno-serve
    step "Q $b at the 262K unit's flags (QAT-UD Q4)"
    SRV_BIN=$BIN ./srv_any.sh start Q-$b $Q4 $P262 && \
        python3 gbench.py --label "N3 $b 262K unit flags" --pp 8192,32768 --reps 1 && \
        python3 turnprobe.py http://127.0.0.1:8080
    efail Q-$b
    ./srv.sh stop
done
step "restoring the fast unit"
./deploy_fast.sh
step "N3B DONE"
