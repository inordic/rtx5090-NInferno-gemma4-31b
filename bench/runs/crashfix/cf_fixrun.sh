#!/bin/bash
# cf_fixrun: apps-up3 (kvshare-up2 + the capture fix + the underflow diagnostic) at production flags,
# in two windows behind the idle gate, the fast unit back after each; a window stops when the
# server is gone.
#   W1  GPU unit tests; night 4's A1 sequence (where #234 exited); the cache replay twice more
#   W2  quality suites, hard50, style chats, agent-mode stress (300 rounds)
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1
B=/opt/NInferno-up/build
LILA=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
TPL=/opt/models/templates/gemma4-agent.jinja
PROD="--vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template $TPL --auto-tools-shared-prefix --auto-shared-prefix-requested --host-state-slots 18 --host-kv-mib 8192"
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
mkdir -p cf logs hard50

take_gpu() {
    ./idle_gate_any.sh 5
    step "$1: idle, taking the GPU"
    systemctl stop gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
}
give_back() { ./srv.sh stop; step "restoring the fast unit"; ./deploy_fast.sh; }
alive() { pgrep -f "apps[^ ]*/ninferno-serve" > /dev/null && curl -sf -m 10 http://127.0.0.1:8080/v1/models > /dev/null; }
run() {
    alive || { echo "--- $1: server gone, window stopped"; return 1; }
    echo "--- $1"; shift; "$@" 2>&1 | tail -2
}
start() { NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 SRV_BIN=$B/apps-up3/ninferno-serve \
              ./srv_any.sh start $1 $LILA $PROD; }
report() {  # label
    if alive; then echo "SURVIVED $1"; else echo "EXITED $1"; fi
    echo "fallbacks: $(grep -c "publishes privately" logs/$1.log)"; grep -m3 "publishes privately" logs/$1.log | cut -c1-200
    grep -hE "Engine failed|FATAL" logs/$1.log | head -2 | cut -c1-700
}

take_gpu W1
(cd $B && for t in ninferno_qwen3_5_state_image_test ninferno_qwen3_5_context_store_test; do
    ./tests/$t > /root/gbench/cf/unit_up3_$t.log 2>&1 && echo "PASS $t" || { echo "FAIL $t"; tail -3 /root/gbench/cf/unit_up3_$t.log; }
done) > cf/UP3_W1.txt
start UP3_W1 >> cf/UP3_W1.txt 2>&1 && {
    run verify python3 verify.py && run textcall python3 textcalltest.py \
        && run vision python3 visiontest.py --label "CF up3" \
        && run speed python3 gbench.py --label "CF up3" --pp 2048,8192,32768,65536 --reps 2 \
        && run turns python3 turnprobe.py http://127.0.0.1:8080 \
        && run edit python3 editbench.py --label "CF up3" \
        && run agent python3 agenttest.py --label "CF up3" --runs 1 \
        && run replay1 python3 cachereplay.py --label "CF up3 r1" --slots 12 \
        && run replay2 python3 cachereplay.py --label "CF up3 r2" --slots 12 \
        && run replay3 python3 cachereplay.py --label "CF up3 r3" --slots 12
    report UP3_W1
} >> cf/UP3_W1.txt 2>&1
grep -hE "^(PASS|FAIL|SURVIVED|EXITED)|fallbacks|underflow" cf/UP3_W1.txt | cut -c1-300
give_back

if grep -q "^SURVIVED" cf/UP3_W1.txt; then
    take_gpu W2
    start UP3_W2 > cf/UP3_W2.txt 2>&1 && {
        run quality python3 geval.py --label "CF-up3" --suites mmlupro,mmlu,gsm8k,humaneval --conc 1 \
            && run hard50 python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag cf_up3 --out hard50/cf_up3.json \
            && run style python3 n4/n4style.py --label up3 \
            && run stress python3 stress_entitlement.py --mode agent --rounds 300 --seed 4
        report UP3_W2
    } >> cf/UP3_W2.txt 2>&1
    grep -hE "^(SURVIVED|EXITED)|fallbacks|underflow|PASSED|FAILED|mmlupro|correct" cf/UP3_W2.txt | cut -c1-300
    give_back
fi
curl -sf http://127.0.0.1:8080/v1/models > /dev/null && step "fast unit answering: $(systemctl is-active gemma4-31b-fast)"
step "CF_FIXRUN_DONE"
