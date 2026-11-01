#!/bin/bash
# night4: four batches behind the idle gate, the fast unit back after each; no batch starts after
# 05:30 CET. Outputs in n4/<label>.txt, server logs in logs/<label>.log.
#  B1  GPU unit tests of apps-up2; A0 = production as deployed (apps-up1 + LilaRest + cache flags)
#  B2  A1 = apps-up2 (+ #226 #246 #217 #234) at the same flags; agent-mode stress (the 27 Oct path)
#  B3  H1 = Humanlike NVFP4 (n4_prep.sh) on the better engine at the same flags; needles 140K/181K
#  B4  the 262K unit's flags: G0 = its QAT-UD Q4 model, G1 = Humanlike Q4_K_M (both on that engine)
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
LAST=$(date -u -d '2026-11-02 04:30' +%s)
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1
UP1=/opt/NInferno-up/build/apps-up1/ninferno-serve
UP2=/opt/NInferno-up/build/apps-up2/ninferno-serve
LILA=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
HLN=/opt/models/hl/humanlike-qkv-vision-nvfp4.r31.ninferno
HLQ=/opt/models/hl/humanlike-q4km.ninferno
Q4=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
TPL=/opt/models/templates/gemma4-agent.jinja
PROD="--vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template $TPL --auto-tools-shared-prefix --auto-shared-prefix-requested --host-state-slots 18 --host-kv-mib 8192"
P262="--max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --draft-tokens 7 --chat-template $TPL"
HS=/opt/NInferno-up/tools/bench/hard_set.py; SET=/opt/NInferno-up/eval/corpora/hard50/hard50.json
mkdir -p hard50 n4 logs

take_gpu() {  # batch name; false when it is too late to start one
    [ "$(date -u +%s)" -ge $LAST ] && { step "$1: too late to start (after 05:30 CET), skipped"; return 1; }
    ./idle_gate_any.sh 5
    [ "$(date -u +%s)" -ge $LAST ] && { step "$1: too late after the gate, skipped"; return 1; }
    step "$1: idle, taking the GPU"
    systemctl stop gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
}
give_back() { ./srv.sh stop; step "restoring the fast unit"; ./deploy_fast.sh; }
mtp() { grep -oE "mtp accepted [0-9]+/[0-9]+" logs/$1.log | tr '/' ' ' |
        awk -v L=$1 '{x += $3; y += $4} END {if (y) printf "MTP %s accepted %d/%d (%.1f%%)\n", L, x, y, 100 * x / y}'; }
efail() { grep -hE "Engine failed|ERROR|FATAL" logs/$1.log | sed -E "s/^.*(ninferno: |ERROR +|FATAL +)//" | cut -c1-300 | head -3; }
full() {  # the fast profile's battery against the server on 8080
    local L=$1
    echo "--- verify";   python3 verify.py 2>&1 | tail -4
    echo "--- textcall"; python3 textcalltest.py 2>&1 | tail -1
    echo "--- vision";   python3 visiontest.py --label "N4 $L" 2>&1 | tail -1
    echo "--- speed";    python3 gbench.py --label "N4 $L" --pp 2048,8192,32768,65536 --reps 2 2>&1 | tail -25
    echo "--- turns";    python3 turnprobe.py http://127.0.0.1:8080 2>&1 | tail -12
    echo "--- edit";     python3 editbench.py --label "N4 $L" 2>&1 | tail -2
    echo "--- agent";    python3 agenttest.py --label "N4 $L" --runs 1 2>&1 | tail -1
    echo "--- replay";   python3 cachereplay.py --label "N4 $L" --slots 12 2>&1 | tail -1
    echo "--- quality";  python3 geval.py --label "N4-$L" --suites mmlupro,mmlu,gsm8k,humaneval --conc 1 2>&1 | tail -6
    echo "--- hard50";   python3 $HS --url http://127.0.0.1:8080/v1 --set $SET --tag n4_$L --out hard50/n4_$L.json 2>&1 | tail -4
    echo "--- style";    python3 n4/n4style.py --label $L 2>&1 | tail -1
    mtp $L; efail $L
}
g262() {  # the 262K unit's battery
    local L=$1
    echo "--- textcall"; python3 textcalltest.py 2>&1 | tail -1
    echo "--- speed";    python3 gbench.py --label "N4 $L" --pp 2048,8192,32768 --reps 1 2>&1 | tail -20
    echo "--- turns";    python3 turnprobe.py http://127.0.0.1:8080 2>&1 | tail -12
    echo "--- agent";    python3 agenttest.py --label "N4 $L" --runs 1 2>&1 | tail -1
    echo "--- quality";  python3 geval.py --label "N4-$L" --suites mmlupro,gsm8k --conc 1 2>&1 | tail -4
    echo "--- needle";   python3 needle.py --label "N4 $L needle" --lengths 230000 --depths 0.5 2>&1 | tail -2
    mtp $L; efail $L
}

if take_gpu B1; then
    step "B1 GPU unit tests of apps-up2"
    (cd /opt/NInferno-up/build && for t in ninferno_qwen3_5_state_image_test ninferno_qwen3_5_context_store_test \
        ninferno_linear_nvfp4_a4_test ninferno_linear_add_nvfp4_test ninferno_linear_gemma4_nvfp4_test; do
        ./tests/$t > /root/gbench/n4/unit_$t.log 2>&1 && echo "PASS $t" || { echo "FAIL $t"; tail -3 /root/gbench/n4/unit_$t.log; }
    done) | tee n4/unit_gpu.txt
    step "B1 A0 production as deployed (apps-up1)"
    NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 SRV_BIN=$UP1 ./srv_any.sh start A0 $LILA $PROD > n4/A0.txt 2>&1 \
        && full A0 >> n4/A0.txt 2>&1
    give_back
fi
if take_gpu B2; then
    step "B2 A1 apps-up2 at production flags"
    NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 SRV_BIN=$UP2 ./srv_any.sh start A1 $LILA $PROD > n4/A1.txt 2>&1 \
        && full A1 >> n4/A1.txt 2>&1 \
        && { echo "--- stress"; python3 stress_entitlement.py --mode agent --rounds 300 --seed 4 2>&1 | tail -1; } >> n4/A1.txt 2>&1
    efail A1 >> n4/A1.txt
    give_back
fi
E=$UP2; EN=up2
if ! grep -q "^PASS" n4/unit_gpu.txt 2>/dev/null || grep -q "^FAIL" n4/unit_gpu.txt || ! grep -q "^PASSED" n4/A1.txt \
   || grep -q "Engine failed" logs/A1.log 2>/dev/null || ! grep -q "7/7" n4/A1.txt; then E=$UP1; EN=up1; fi
step "engine for the model tests: $EN"

until grep -qE "NVFP4_DONE|NVFP4_FAILED|PREP_SKIPPED|BF16 SHA FAILED" n4/prep.out 2>/dev/null || [ "$(date -u +%s)" -ge $LAST ]; do sleep 60; done
if grep -q NVFP4_DONE n4/prep.out && take_gpu B3; then
    step "B3 H1 Humanlike NVFP4 on $EN at production flags"
    NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 SRV_BIN=$E ./srv_any.sh start H1 $HLN $PROD > n4/H1.txt 2>&1 \
        && full H1 >> n4/H1.txt 2>&1 \
        && { echo "--- needles"; python3 needle.py --label "N4 H1 needles" --lengths 140000,181000 --depths 0.5 2>&1 | tail -2; } >> n4/H1.txt 2>&1
    give_back
else step "B3 skipped: $(grep -hE 'NVFP4_FAILED|PREP_SKIPPED|SHA FAILED' n4/prep.out | tail -1)"; fi

until grep -qE "GGUF_CONVERT_(DONE|FAILED)|PREP_SKIPPED|PREP_DONE" n4/prep.out n4/conv_gguf.log 2>/dev/null || [ "$(date -u +%s)" -ge $LAST ]; do sleep 60; done
if grep -q GGUF_CONVERT_DONE n4/conv_gguf.log 2>/dev/null && take_gpu B4; then
    step "B4 G0 the 262K unit's QAT-UD Q4 on $EN"
    SRV_BIN=$E ./srv_any.sh start G0 $Q4 $P262 --drafter $DQ > n4/G0.txt 2>&1 && g262 G0 >> n4/G0.txt 2>&1
    ./srv.sh stop
    step "B4 G1 Humanlike Q4_K_M on $EN at the 262K flags, regular assistant"
    SRV_BIN=$E ./srv_any.sh start G1 $HLQ $P262 --drafter $DR > n4/G1.txt 2>&1 && g262 G1 >> n4/G1.txt 2>&1
    give_back
else step "B4 skipped"; fi

curl -sf http://127.0.0.1:8080/v1/models > /dev/null && step "fast unit answering: $(systemctl is-active gemma4-31b-fast)"
step "N4 DONE"
