#!/bin/bash
# queue22: after queue21 + build_qk8x2 - split-Q FP8 scores: 32K perplexity (pv8b 77.99, qk8 81.13)
# and prefill at 8K / 32K / 64K (pv8b 4,986 / 3,522 / 2,542; qk8 5,116 / 3,760 / 2,811).
cd /root/gbench
until grep -qE "QUEUE21 DONE" queue21.log 2>/dev/null && grep -qE "BUILD_QK8X2_OK|BUILD_FAILED" build_qk8x2.out; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
grep -q BUILD_QK8X2_OK build_qk8x2.out || { step "QK8X2 BUILD FAILED"; step "QUEUE22 DONE"; exit 1; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
G=/opt/NInferno/build/apps-qk8x2/ninferno-gemma4
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
step "score qk8x2 window 32768"
$G $QN score --tokens /root/gbench/ppl64k.i32 --window 32768 --kv-dtype fp8 --context 32768 --dump /root/gbench/score_qk8x2_32768.bin > /dev/null 2>&1
python3 ppl.py /root/gbench/score_qk8x2_32768.bin
for p in 8192 32768 65536; do
    step "bench qk8x2 prefill $p"
    $G $QN bench --tokens /root/gbench/prompt128k.i32 --prompt $p --decode 4 --kv-dtype fp8 --context $((p + 2048)) --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+" ; echo
done
step "QUEUE22 DONE"
