#!/bin/bash
# queue15: after queue14 + build_pv8 - FP8 P V (pv8) against BF16 P V (a8 = i8c + W4A8 option) and
# no INT8 (split): perplexity at a 32K window, then lab prefill at 32K and 64K.
cd /root/gbench
until grep -q "QUEUE14 DONE" queue14.log 2>/dev/null && grep -qE "BUILD_PV8_OK|BUILD_FAILED" build_pv8.out; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
grep -q BUILD_PV8_OK build_pv8.out || { step "PV8 BUILD FAILED"; exit 1; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 NINFERNO_GEMMA4_SHARED_VALUES=2
for v in split a8 pv8; do
    step "score $v window 32768 (shared2)"
    /opt/NInferno/build/apps-$v/ninferno-gemma4 $QN score --tokens /root/gbench/ppl64k.i32 --window 32768 \
        --kv-dtype fp8 --context 32768 --dump /root/gbench/score_s2_${v}_32768.bin > /dev/null 2>&1
    python3 ppl.py /root/gbench/score_s2_${v}_32768.bin
done
for v in a8 pv8; do
    for p in 32768 65536; do
        step "bench $v prefill $p"
        /opt/NInferno/build/apps-$v/ninferno-gemma4 $QN bench --tokens /root/gbench/prompt128k.i32 --prompt $p \
            --decode 4 --kv-dtype fp8 --context $((p + 2048)) --chunk 2048 2>&1 | grep -oE "\"prefill_tok_s\":[0-9.]+|\"decode_tok_s\":[0-9.]+" | tr "\n" " "; echo
    done
done
step "QUEUE15 DONE"
