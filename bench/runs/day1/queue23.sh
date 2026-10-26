#!/bin/bash
# queue23: after queue20 + build_qk8p - promoted FP8 scores: 32K perplexity (split 76.96, pv8b
# 77.99, qk8 81.13) and prefill at 8K / 32K / 64K (pv8b 4,986 / 3,522 / 2,542).
cd /root/gbench
until grep -qE "QUEUE20 DONE" queue20.log 2>/dev/null && grep -q BUILD_QK8P_DONE build_qk8p.out; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
for v in qk8p1 qk8p2; do
    grep -q BUILD_${v}_OK build_qk8p.out || continue
    G=/opt/NInferno/build/apps-$v/ninferno-gemma4
    step "score $v window 32768"
    $G $QN score --tokens /root/gbench/ppl64k.i32 --window 32768 --kv-dtype fp8 --context 32768 --dump /root/gbench/score_${v}_32768.bin > /dev/null 2>&1
    python3 ppl.py /root/gbench/score_${v}_32768.bin
    for p in 8192 32768 65536; do
        step "bench $v prefill $p"
        $G $QN bench --tokens /root/gbench/prompt128k.i32 --prompt $p --decode 4 --kv-dtype fp8 --context $((p + 2048)) --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+" ; echo
    done
done
step "QUEUE23 DONE"
