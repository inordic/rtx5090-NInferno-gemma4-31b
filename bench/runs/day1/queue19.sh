#!/bin/bash
# queue19: after queue16 + build_pv8b - FP8 P V in the lab tool (non-shared V, gain 2^13): 32K
# perplexity against a8 (78.5567, BF16 P V) and prefill at 32K / 64K (a8: 2,992 / 2,034).
cd /root/gbench
until grep -q "QUEUE16 DONE" queue16.log 2>/dev/null && grep -qE "BUILD_PV8B_OK|BUILD_FAILED" build_pv8b.out; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
grep -q BUILD_PV8B_OK build_pv8b.out || { step "PV8B BUILD FAILED"; exit 1; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
G=/opt/NInferno/build/apps-pv8b/ninferno-gemma4
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
step "score pv8b window 32768"
$G $QN score --tokens /root/gbench/ppl64k.i32 --window 32768 --kv-dtype fp8 --context 32768 --dump /root/gbench/score_pv8b_32768.bin > /dev/null 2>&1
python3 ppl.py /root/gbench/score_pv8b_32768.bin /root/gbench/score_s2_a8_32768.bin
for p in 32768 65536; do
    step "bench pv8b prefill $p"
    $G $QN bench --tokens /root/gbench/prompt128k.i32 --prompt $p --decode 4 --kv-dtype fp8 --context $((p + 2048)) --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+|\"decode_tok_s\":[0-9.]+" | tr "\n" " "; echo
done
step "QUEUE19 DONE"
