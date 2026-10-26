#!/bin/bash
# queue13: after queue12 + build_a8 - is the NVFP4 knowledge drop the W4A4 prefill or the NVFP4
# weights? W4A16 (exact) and W4A8 (int8, new for Gemma 4) on mmlupro + mmlu; W4A8 prefill speed.
cd /root/gbench
until grep -q "QUEUE12 DONE" queue12.log 2>/dev/null && grep -qE "BUILD_A8_OK|BUILD_FAILED" build_a8.out; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
grep -q BUILD_A8_OK build_a8.out || { step "A8 BUILD FAILED"; exit 1; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
sed "s#^S=.*#S=/opt/NInferno/build/apps-a8/ninferno-serve#" srv.sh > srv_a8.sh; chmod +x srv_a8.sh
km() {  # km <label> <artifact> [serve args...]
    local label=$1 art=$2; shift 2
    NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv_a8.sh start "k-$label" $art --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 "$@" || return
    python3 geval.py --label "$label" --suites mmlupro,mmlu --conc 4
    ./srv.sh stop
}
step "knowledge: NVFP4 W4A8 (int8)"; km nvfp4-a8 $QN --activations int8
step "PP: NVFP4 W4A8 + INT8 attention, K8, 90K chunk2048"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv_a8.sh start P8-nvfp4-a8 $QN --activations int8 --max-context 90000 --prefill-chunk 2048 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 gbench.py --label "P8 QAT-NVFP4 W4A8 INT8attn K8 shared2 90K chunk2048" --pp 2048,8192,32768 --reps 1 --skip-chat
./srv.sh stop
step "knowledge: NVFP4 W4A16 (exact)"; km nvfp4-exact $QN --activations exact
step "QUEUE13 DONE"
