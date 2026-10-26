#!/bin/bash
# queue26: after queue25 - lab prefill rows for the report: BF16 attention (split) and INT8 + BF16 P V
# (a8) at 8K / 32K / 64K, and the final binary at the same lengths.
cd /root/gbench
until grep -qE "QUEUE25 DONE" queue25.log 2>/dev/null; do sleep 10; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
for v in split a8 final; do
    for p in 8192 32768 65536; do
        printf "LAB %s %s " $v $p
        /opt/NInferno/build/apps-$v/ninferno-gemma4 $QN bench --tokens /root/gbench/prompt128k.i32 --prompt $p --decode 4 --kv-dtype fp8 --context $((p + 2048)) --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+"
    done
done
step "QUEUE26 DONE"
